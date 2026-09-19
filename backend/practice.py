"""Account-owned learning lifecycle. Model work always runs outside write transactions."""
import hashlib
import json
import math
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Blueprint, current_app, g, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from backend.database import get_db, init_db, transaction, utcnow
from backend.study_agent import AgentLoopError, run_study_agent

bp = Blueprint('practice', __name__, url_prefix='/api')
CONFIDENCES = {'high', 'medium', 'low', 'unknown'}
ERROR_TYPES = {'concept_missing', 'confusion', 'method', 'calculation', 'reading', 'expression', 'unknown'}
QUESTION_TYPES = {'single_choice', 'multiple_choice', 'fill_blank', 'short_answer'}


class APIError(Exception):
    def __init__(self, message, status=400):
        self.message, self.status = message, status


class Expired(APIError):
    def __init__(self, question_id):
        super().__init__('原题缓存已失效，请生成新题。', 410)
        self.question_id = question_id


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise APIError('请求必须是 JSON 对象。')
    return data


def integer(value, name, low=0, high=2**53-1):
    if type(value) is not int or not low <= value <= high:
        raise APIError(f'{name} 必须是 {low}～{high} 的整数。')
    return value


def string(value, name, limit=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise APIError(f'{name} 不能为空且不能超过 {limit} 字符。')
    return value.strip()


def password_text(data):
    value = data.get('password')
    if not isinstance(value, str) or not 8 <= len(value) <= 256:
        raise APIError('密码长度应为 8～256 个字符。')
    return value


def owned(table, record_id):
    # table names only come from constants in this module.
    if table in ('hint_records', 'error_records'):
        row = get_db().execute(f'SELECT r.* FROM {table} r JOIN question_attempts a ON a.id=r.attempt_id WHERE r.id=? AND a.user_id=?', (record_id, g.user['id'])).fetchone()
    else:
        row = get_db().execute(f'SELECT * FROM {table} WHERE id=? AND user_id=?', (record_id, g.user['id'])).fetchone()
    if row is None:
        raise APIError('记录不存在。', 404)
    return dict(row)


def insert(db, table, values):
    return db.execute(f"INSERT INTO {table} ({','.join(values)}) VALUES ({','.join('?' for _ in values)})", tuple(values.values())).lastrowid


def storage_path(relative):
    root = Path(current_app.config['STORAGE_ROOT']).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise APIError('无效存储路径。')
    return path


def delete_cache(question_id):
    try:
        storage_path(f'cache/questions/{question_id}.json').unlink(missing_ok=True)
    except OSError:
        current_app.logger.warning('Could not remove question cache %s', question_id)


def expire_caches(user_id=None):
    sql = "SELECT id,content_expires_at FROM questions WHERE status='ready' AND content_snapshot_json IS NULL"
    args = ()
    if user_id is not None:
        sql += ' AND user_id=?'
        args = (user_id,)
    candidates = get_db().execute(sql, args).fetchall()
    expired = []
    with transaction() as db:
        for q in candidates:
            if q['content_expires_at'] <= utcnow() or not storage_path(f"cache/questions/{q['id']}.json").is_file():
                # Re-check the snapshot under the write lock: a concurrent submit may have saved it.
                changed = db.execute("UPDATE questions SET status='expired' WHERE id=? AND content_snapshot_json IS NULL AND status='ready'", (q['id'],)).rowcount
                if changed:
                    db.execute("UPDATE question_attempts SET status='expired',last_activity_at=?,updated_at=? WHERE question_id=? AND status='in_progress'", (utcnow(), utcnow(), q['id']))
                    expired.append(q['id'])
    for qid in expired:
        delete_cache(qid)
    return len(expired)


def provider():
    return current_app.extensions['learning_provider']


def content(question):
    if question['status'] != 'ready':
        raise APIError('题目已失效。', 410)
    if question['content_snapshot_json']:
        return json.loads(question['content_snapshot_json'])
    path = storage_path(f"cache/questions/{question['id']}.json")
    if not question['content_expires_at'] or question['content_expires_at'] <= utcnow() or not path.is_file():
        raise Expired(question['id'])
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        raise Expired(question['id'])


def public_question(question):
    snapshot = content(question)
    return {k: question[k] for k in ('id', 'material_id', 'primary_knowledge_point_id', 'question_type', 'difficulty_level', 'question_summary', 'status', 'created_at')} | {'question': snapshot['question'], 'options': snapshot.get('options', [])}


def public_user(row):
    return {k: row[k] for k in ('id', 'username', 'display_name', 'status', 'created_at', 'last_login_at')}


def init_app(app, learning_provider=None):
    root = Path(__file__).parent / 'storage'
    app.config.setdefault('STORAGE_ROOT', str(root))
    app.config.setdefault('DATABASE', str(Path(app.config['STORAGE_ROOT']) / 'ctrlpass.sqlite3'))
    app.config.setdefault('QUESTION_CACHE_SECONDS', 86400)
    app.config.setdefault('AGENT_MAX_TOOL_STEPS', 4)
    app.config.setdefault('MAX_CONTENT_LENGTH', 32 * 1024 * 1024)
    if app.config['MAX_CONTENT_LENGTH'] is None:
        app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
    if not app.secret_key:
        secret_file = Path(app.config['STORAGE_ROOT']) / '.session-secret'
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with secret_file.open('x', encoding='utf-8') as f:
                f.write(secrets.token_hex(32))
        except FileExistsError:
            pass
        app.secret_key = os.getenv('CTRLPASS_SECRET_KEY') or secret_file.read_text(encoding='utf-8')
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
    init_db(app)
    if learning_provider is None:
        from backend.learning_provider import LearningProvider
        learning_provider = LearningProvider()
    app.extensions['learning_provider'] = learning_provider
    app.register_blueprint(bp)

    @app.cli.command('cleanup-question-cache')
    def cleanup_question_cache():
        """Expire unsubmitted questions and remove their temporary content."""
        import click
        count = expire_caches()
        for row in get_db().execute("SELECT id FROM questions WHERE status!='ready' OR content_snapshot_json IS NOT NULL"):
            delete_cache(row['id'])
        click.echo(f'Expired {count} questions; removed obsolete caches.')

    @app.before_request
    def authenticate():
        if not request.path.startswith('/api/') or request.method == 'OPTIONS':
            return
        if request.method in ('POST', 'PATCH', 'DELETE', 'PUT'):
            origin = request.headers.get('Origin')
            allowed = {request.host_url.rstrip('/'), *filter(None, os.getenv('CTRLPASS_ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(','))}
            if request.headers.get('X-CtrlPass-Request') != '1' or (origin and origin not in allowed):
                raise APIError('请求来源验证失败。', 403)
        if request.path in ('/api/health', '/api/auth/register', '/api/auth/login'):
            return
        row = get_db().execute('SELECT * FROM users WHERE id=? AND status=?', (session.get('user_id'), 'active')).fetchone()
        if row is None:
            session.clear()
            raise APIError('请先登录。', 401)
        g.user = dict(row)

    @app.errorhandler(APIError)
    def api_error(exc):
        if isinstance(exc, Expired):
            with transaction() as db:
                changed = db.execute("UPDATE questions SET status='expired' WHERE id=? AND content_snapshot_json IS NULL", (exc.question_id,)).rowcount
                if changed:
                    db.execute("UPDATE question_attempts SET status='expired',last_activity_at=?,updated_at=? WHERE question_id=? AND status='in_progress'", (utcnow(), utcnow(), exc.question_id))
            if changed:
                delete_cache(exc.question_id)
        return jsonify(error=exc.message), exc.status

    @app.errorhandler(sqlite3.IntegrityError)
    def conflict(_exc):
        return jsonify(error='记录冲突或关联数据无效，请刷新后重试。'), 409


@bp.post('/auth/register')
def register():
    data = body()
    username = string(data.get('username'), '登录名', 80)
    password = password_text(data)
    with transaction() as db:
        uid = insert(db, 'users', dict(username=username, password_hash=generate_password_hash(password), display_name=string(data.get('display_name') or username, '显示名称', 80), created_at=utcnow()))
    return jsonify(user=public_user(get_db().execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone())), 201


@bp.post('/auth/login')
def login():
    data = body()
    username = string(data.get('username'), '登录名', 80)
    password = password_text(data)
    row = get_db().execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
    if row is None or not check_password_hash(row['password_hash'], password) or row['status'] != 'active':
        raise APIError('账号或密码错误，或账号已停用。', 401)
    with transaction() as db:
        db.execute('UPDATE users SET last_login_at=? WHERE id=?', (utcnow(), row['id']))
    session.clear()
    session['user_id'] = row['id']
    session['workspace_key'] = secrets.token_hex(24)
    return jsonify(user=public_user(get_db().execute('SELECT * FROM users WHERE id=?', (row['id'],)).fetchone()))


@bp.post('/auth/logout')
def logout():
    session.clear()
    return jsonify(success=True)


@bp.get('/auth/me')
def me():
    return jsonify(user=public_user(g.user))


@bp.post('/materials/upload')
def upload_material():
    upload = request.files.get('file')
    if upload is None or not upload.filename:
        raise APIError('请选择文件。')
    filename = upload.filename.replace('\\', '/').split('/')[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.txt', '.pdf', '.docx'):
        raise APIError('仅支持 TXT、PDF、DOCX。')
    raw = upload.read()
    if not raw:
        raise APIError('文件为空。')
    digest = hashlib.sha256(raw).hexdigest()
    with transaction() as db:
        duplicate = db.execute('SELECT id FROM learning_materials WHERE user_id=? AND content_hash=?', (g.user['id'], digest)).fetchone()
        mid = insert(db, 'learning_materials', dict(user_id=g.user['id'], original_filename=filename, file_type=suffix[1:], file_size=len(raw), file_path='', content_hash=digest, created_at=utcnow()))
        relative = f'materials/{mid}/source{suffix}'
        path = storage_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        db.execute('UPDATE learning_materials SET file_path=? WHERE id=?', (relative, mid))
    return jsonify(material=owned('learning_materials', mid), duplicate_material_id=duplicate['id'] if duplicate else None, file_id=mid, filename=filename, file_path=relative), 201


@bp.get('/materials')
def materials():
    return jsonify(materials=[dict(row) for row in get_db().execute('SELECT * FROM learning_materials WHERE user_id=? ORDER BY id DESC', (g.user['id'],))])


@bp.post('/materials/<int:material_id>/process')
def process_material(material_id):
    data = body()
    size = integer(data.get('chunk_size', 500), '分段大小', 100, 10000)
    overlap = integer(data.get('chunk_overlap', 100), '重叠长度', 0, size-1)
    model_splitter = data.get('use_model_splitter', False)
    if type(model_splitter) is not bool:
        raise APIError('use_model_splitter 必须是布尔值。')
    config = dict(chunk_size=size, chunk_overlap=overlap, use_model_splitter=model_splitter)
    with transaction() as db:
        material = owned('learning_materials', material_id)
        previous_config = json.loads(material['processing_config_json'])
        processing_start = previous_config.get('_processing_started_at')
        if material['status'] == 'processing' and processing_start and (datetime.now(timezone.utc)-datetime.fromisoformat(processing_start.replace('Z', '+00:00'))).total_seconds() < 1800:
            raise APIError('资料正在处理中；中断超过 30 分钟后可重试。', 409)
        if material['status'] == 'ready':
            return jsonify(success=True, material=material, chunks=material['chunk_count'], message='资料已可使用。')
        config['_processing_key'] = secrets.token_hex(12)
        config['_processing_started_at'] = utcnow()
        config_json = dump(config)
        db.execute("UPDATE learning_materials SET status='processing',processing_config_json=?,error_message=NULL WHERE id=?", (config_json, material_id))
    try:
        result = provider().process(material, config, storage_path(''))
        integer(result.get('chunk_count'), '文本片段数', 1)
        storage_path(string(result.get('vector_store_path'), '向量路径', 500))
        names = result['knowledge_points']
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError('Invalid knowledge points')
        # Preserve any prior IDs. New names are appended, never renumbered.
        points = json.loads(material['knowledge_points_json'])
        for name in dict.fromkeys(names):
            if not any(p['name'] == name for p in points):
                points.append({'id': max((p['id'] for p in points), default=0)+1, 'name': name})
        with transaction() as db:
            changed = db.execute("UPDATE learning_materials SET status='ready',chunk_count=?,vector_store_path=?,knowledge_points_json=?,processed_at=? WHERE id=? AND status='processing' AND processing_config_json=?", (result['chunk_count'], result['vector_store_path'], dump(points), utcnow(), material_id, config_json)).rowcount
            if not changed:
                raise APIError('处理任务已被更新，请刷新资料。', 409)
    except Exception:
        current_app.logger.exception('Material processing failed: %s', material_id)
        with transaction() as db:
            db.execute("UPDATE learning_materials SET status='failed',error_message=?,processed_at=? WHERE id=? AND status='processing' AND processing_config_json=?", ('资料处理失败，请检查模型配置和文件内容后重试。', utcnow(), material_id, config_json))
        raise APIError('资料处理失败，可以重试。', 502)
    return jsonify(success=True, material=owned('learning_materials', material_id), chunks=result['chunk_count'], message='资料处理完成。')


@bp.patch('/materials/<int:material_id>/knowledge-points/<int:point_id>')
def rename_point(material_id, point_id):
    name = string(body().get('name'), '知识点名称', 200)
    with transaction() as db:
        material = owned('learning_materials', material_id)
        if material['status'] != 'ready':
            raise APIError('请等待资料处理完成。', 409)
        points = json.loads(material['knowledge_points_json'])
        point = next((p for p in points if p['id'] == point_id), None)
        if point is None:
            raise APIError('知识点不存在。', 404)
        if any(p['id'] != point_id and p['name'] == name for p in points):
            raise APIError('知识点名称重复。', 409)
        point['name'] = name
        db.execute('UPDATE learning_materials SET knowledge_points_json=? WHERE id=?', (dump(points), material_id))
    return jsonify(knowledge_points=points)


def history_context(material_id):
    rows = get_db().execute('''SELECT a.id,a.answer_summary,a.score,a.grading_confidence,q.question_summary,q.difficulty_level,
     q.primary_knowledge_point_id,q.question_type,
     (SELECT COUNT(*) FROM hint_records h WHERE h.attempt_id=a.id AND h.status='viewed' AND h.hint_type!='answer') AS hint_count,
     (SELECT COUNT(*) FROM hint_records h WHERE h.attempt_id=a.id AND h.status='viewed' AND h.hint_type='answer') AS answer_view_count
     FROM question_attempts a JOIN questions q ON q.id=a.question_id
     WHERE a.user_id=? AND q.material_id=? AND a.status='completed' ORDER BY a.id DESC LIMIT 10''', (g.user['id'], material_id)).fetchall()
    history = []
    for row in rows:
        item = dict(row)
        item['errors'] = [dict(e) for e in get_db().execute("SELECT knowledge_point_id,error_type,description,confidence,review_status FROM error_records WHERE attempt_id=? AND review_status!='rejected'", (row['id'],))]
        history.append(item)
    return history


def validate_snapshot(raw, question_type):
    if not isinstance(raw, dict):
        raise APIError('模型返回的题目格式无效。', 502)
    question = string(raw.get('question'), '题干', 16000)
    rubric = string(raw.get('scoring_rubric'), '评分标准', 4000)
    answer = raw.get('answer')
    options = raw.get('options', [])
    if question_type in ('single_choice', 'multiple_choice'):
        if not isinstance(options, list) or not 2 <= len(options) <= 12:
            raise APIError('模型返回的选项无效。', 502)
        ids, normalized_options = [], []
        for opt in options:
            if not isinstance(opt, dict):
                raise APIError('模型返回的选项无效。', 502)
            ids.append(string(opt.get('id'), '选项编号', 10))
            normalized_options.append({'id': ids[-1], 'text': string(opt.get('text'), '选项内容', 4000)})
        if len(set(ids)) != len(ids):
            raise APIError('模型返回重复选项编号。', 502)
        valid = isinstance(answer, str) and answer in ids if question_type == 'single_choice' else isinstance(answer, list) and len(answer) > 0 and all(isinstance(a, str) and a in ids for a in answer) and len(set(answer)) == len(answer)
        if not valid:
            raise APIError('模型返回的标准答案无效。', 502)
        options = normalized_options
    else:
        string(answer, '标准答案', 8000)
        options = []
    return dict(question=question, options=options, answer=answer, scoring_rubric=rubric)


def create_question(material, kp, qtype, difficulty, goal, context_extra=None):
    """Generate, validate and persist one question for manual or agent mode."""
    points = json.loads(material['knowledge_points_json'])
    context = dict(
        goal=goal,
        history=history_context(material['id']),
        requested_difficulty=difficulty,
        adjustment_reason='结合学习历史选择考查方式；不因低可信度错因大幅调整难度。',
    )
    if context_extra:
        context.update(context_extra)
    try:
        raw = provider().generate(material, kp, qtype, difficulty, context, storage_path(''))
        snapshot = validate_snapshot(raw, qtype)
        summary = string(raw.get('question_summary'), '题目摘要', 500)
        skill_type = string(raw.get('skill_type', 'understanding'), '考查能力', 80)
        secondary = raw.get('secondary_knowledge_point_ids', [])
        if not isinstance(secondary, list) or any(type(p) is not int or p not in {v['id'] for v in points} or p == kp for p in secondary):
            raise APIError('辅助知识点无效。', 502)
        refs = raw.get('source_refs', [])
        if not isinstance(refs, list) or not refs:
            raise APIError('题目缺少来源。', 502)
    except APIError:
        raise
    except Exception:
        current_app.logger.exception('Question generation failed')
        raise APIError('出题失败，请检查模型服务后重试。', 502)
    now = utcnow()
    expires = (datetime.now(timezone.utc)+timedelta(seconds=current_app.config['QUESTION_CACHE_SECONDS'])).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    with transaction() as db:
        qid = insert(db, 'questions', dict(user_id=g.user['id'], material_id=material['id'], primary_knowledge_point_id=kp, secondary_knowledge_point_ids_json=dump(secondary), question_type=qtype, difficulty_level=difficulty, skill_type=skill_type, question_summary=summary, source_refs_json=dump(refs), generation_context_json=dump(context), model_name=provider().model_name, prompt_version='v1', policy_version='v1', content_expires_at=expires, created_at=now))
        path = storage_path(f'cache/questions/{qid}.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump(snapshot), encoding='utf-8')
    return public_question(owned('questions', qid))


@bp.post('/questions/generate')
def generate():
    data = body()
    expire_caches(g.user['id'])
    material = owned('learning_materials', integer(data.get('material_id'), '资料编号', 1))
    if material['status'] != 'ready':
        raise APIError('资料尚未处理完成。', 409)
    points = json.loads(material['knowledge_points_json'])
    kp = integer(data.get('primary_knowledge_point_id'), '知识点编号', 1)
    if kp not in {p['id'] for p in points}:
        raise APIError('知识点不属于该资料。')
    qtype = data.get('question_type', 'single_choice')
    if not isinstance(qtype, str) or qtype not in QUESTION_TYPES:
        raise APIError('题型无效。')
    difficulty = integer(data.get('difficulty_level', 3), '预计难度', 1, 5)
    goal = string(data.get('goal') or '巩固当前知识点', '出题目标', 500)
    return jsonify(question=create_question(material, kp, qtype, difficulty, goal)), 201


def learning_stats(points, history):
    """Build deterministic evidence that the agent can inspect as a tool result."""
    stats = {point['id']: {
        'knowledge_point_id': point['id'],
        'name': point['name'],
        'attempt_count': 0,
        'average_score': None,
        'latest_score': None,
        'confirmed_error_count': 0,
    } for point in points}
    scores = {point['id']: [] for point in points}
    for attempt in history:
        point_id = attempt['primary_knowledge_point_id']
        if point_id not in stats:
            continue
        stats[point_id]['attempt_count'] += 1
        if attempt['score'] is not None:
            scores[point_id].append(attempt['score'])
            if stats[point_id]['latest_score'] is None:
                stats[point_id]['latest_score'] = attempt['score']
        for error in attempt['errors']:
            error_point = error.get('knowledge_point_id')
            if error_point in stats and error.get('review_status') == 'confirmed':
                stats[error_point]['confirmed_error_count'] += 1
    for point_id, values in scores.items():
        if values:
            stats[point_id]['average_score'] = round(sum(values) / len(values), 3)
    return list(stats.values())


@bp.post('/agent/run')
def run_agent():
    """Let the model observe, choose tools and re-observe until it has a result."""
    data = body()
    expire_caches(g.user['id'])
    material = owned('learning_materials', integer(data.get('material_id'), '资料编号', 1))
    if material['status'] != 'ready':
        raise APIError('资料尚未处理完成。', 409)
    task = data.get('task', 'auto')
    if task not in ('auto', 'practice', 'qa', 'knowledge_graph'):
        raise APIError('Agent 任务类型无效。')
    defaults = {
        'auto': '根据资料和学习记录选择最合适的下一项学习任务',
        'practice': '根据我的学习记录安排下一道练习题',
        'qa': '概括这份资料的核心内容',
        'knowledge_graph': '生成整份资料的核心知识图谱',
    }
    goal = string(data.get('goal') or defaults[task], '学习目标或问题', 1000)
    points = json.loads(material['knowledge_points_json'])
    history = history_context(material['id'])
    completed = {}
    base_observation = {
        'material': {'id': material['id'], 'name': material['original_filename']},
        'knowledge_points': points,
        'recent_history': history,
        'requested_task': task,
    }

    def observe_agent(state):
        # Every turn re-reads the stable context and includes the newest tool result.
        return base_observation | state.get('last_tool_result', {})

    def decide_agent(state):
        return provider().decide_next(
            goal,
            state['observation'],
            state.get('trace', []),
            'result' in state,
        )

    def inspect_history(_arguments):
        return {'point_stats': learning_stats(points, history)}

    def generate_practice(arguments):
        if 'practice' in completed:
            return completed['practice'] | {'reused': True}
        kp = arguments.get('primary_knowledge_point_id')
        qtype = arguments.get('question_type')
        difficulty = arguments.get('difficulty_level')
        focus = arguments.get('focus') or goal
        if type(kp) is not int or kp not in {point['id'] for point in points}:
            raise AgentLoopError('Agent selected an invalid knowledge point')
        if not isinstance(qtype, str) or qtype not in QUESTION_TYPES:
            raise AgentLoopError('Agent selected an invalid question type')
        if type(difficulty) is not int or not 1 <= difficulty <= 5:
            raise AgentLoopError('Agent selected an invalid difficulty')
        if not isinstance(focus, str) or not focus.strip() or len(focus) > 500:
            raise AgentLoopError('Agent selected an invalid practice focus')
        question = create_question(
            material,
            kp,
            qtype,
            difficulty,
            focus.strip(),
            context_extra={'agent_mode': True, 'agent_goal': goal},
        )
        completed['practice'] = {'type': 'practice', 'question': question}
        return completed['practice'] | {'reused': False}

    def answer_from_material(arguments):
        if 'qa' in completed:
            return completed['qa'] | {'reused': True}
        question = arguments.get('question') or goal
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise AgentLoopError('Agent produced an invalid material question')
        answer = provider().answer(material, question.strip(), storage_path(''))
        completed['qa'] = {'type': 'qa', **answer}
        return completed['qa'] | {'reused': False}

    def generate_graph(arguments):
        if 'knowledge_graph' in completed:
            return completed['knowledge_graph'] | {'reused': True}
        focus = arguments.get('focus') or goal
        if not isinstance(focus, str) or not focus.strip() or len(focus) > 1000:
            raise AgentLoopError('Agent produced an invalid graph focus')
        graph = provider().knowledge_graph(material, focus.strip(), storage_path(''))
        completed['knowledge_graph'] = {'type': 'knowledge_graph', 'graph': graph}
        return completed['knowledge_graph'] | {'reused': False}

    try:
        result = run_study_agent(
            goal,
            observe_agent,
            decide_agent,
            {
                'inspect_learning_history': inspect_history,
                'generate_practice_question': generate_practice,
                'answer_from_material': answer_from_material,
                'generate_knowledge_graph': generate_graph,
            },
            result_actions={'generate_practice_question', 'answer_from_material', 'generate_knowledge_graph'},
            max_tool_steps=current_app.config['AGENT_MAX_TOOL_STEPS'],
        )
    except APIError:
        raise
    except AgentLoopError as exc:
        current_app.logger.warning('Study agent stopped: %s', exc)
        raise APIError(f'学习 Agent 未能完成任务：{exc}', 502)
    except Exception:
        current_app.logger.exception('Study agent failed')
        raise APIError('学习 Agent 运行失败，请检查模型服务后重试。', 502)
    output = result['result']
    response = {
        'result': output,
        'agent': {
            'goal': goal,
            'message': result['final_message'],
            'tool_steps': result['tool_steps'],
            'trace': result['trace'],
        },
    }
    if output['type'] == 'practice':
        response['question'] = output['question']
    return jsonify(response), 201


@bp.post('/materials/<int:material_id>/ask')
def ask_material(material_id):
    data = body()
    material = owned('learning_materials', material_id)
    if material['status'] != 'ready':
        raise APIError('资料尚未处理完成。', 409)
    question = string(data.get('question'), '问题', 1000)
    try:
        return jsonify(provider().answer(material, question, storage_path('')))
    except Exception:
        current_app.logger.exception('Material question answering failed')
        raise APIError('资料问答失败，请检查模型服务后重试。', 502)


@bp.post('/materials/<int:material_id>/knowledge-graph')
def material_knowledge_graph(material_id):
    data = body()
    material = owned('learning_materials', material_id)
    if material['status'] != 'ready':
        raise APIError('资料尚未处理完成。', 409)
    focus = data.get('focus') or '整份资料的核心概念与关系'
    focus = string(focus, '知识图谱范围', 1000)
    try:
        return jsonify(graph=provider().knowledge_graph(material, focus, storage_path('')))
    except Exception:
        current_app.logger.exception('Knowledge graph generation failed')
        raise APIError('知识图谱生成失败，请检查模型服务后重试。', 502)


@bp.get('/questions/<int:question_id>')
def get_question(question_id):
    return jsonify(question=public_question(owned('questions', question_id)))


@bp.post('/questions/<int:question_id>/attempts')
def start_attempt(question_id):
    data = body()
    hidden = integer(data.get('is_page_hidden', 0), '页面隐藏状态', 0, 1)
    with transaction() as db:
        question = owned('questions', question_id)
        content(question)
        row = db.execute("SELECT id FROM question_attempts WHERE question_id=? AND user_id=? AND status='in_progress'", (question_id, g.user['id'])).fetchone()
        if row:
            aid = row['id']
            db.execute('UPDATE question_attempts SET last_activity_at=? WHERE id=?', (utcnow(), aid))
        else:
            number = db.execute('SELECT COALESCE(MAX(attempt_no),0)+1 FROM question_attempts WHERE user_id=? AND question_id=?', (g.user['id'], question_id)).fetchone()[0]
            now = utcnow()
            aid = insert(db, 'question_attempts', dict(user_id=g.user['id'], question_id=question_id, attempt_no=number, started_at=now, last_activity_at=now, timing_updated_at=now, is_page_hidden=hidden, created_at=now, updated_at=now))
    return jsonify(attempt=owned('question_attempts', aid)), 201


def progress(db, attempt, data):
    seq = integer(data.get('progress_seq'), '进度序号', 1)
    total = integer(data.get('active_duration_ms'), '累计用时')
    paused = integer(data.get('is_manually_paused'), '暂停状态', 0, 1)
    hidden = integer(data.get('is_page_hidden'), '页面隐藏状态', 0, 1)
    if seq <= attempt['progress_seq']:
        return False
    now = utcnow()
    elapsed = max(0, (datetime.fromisoformat(now.replace('Z', '+00:00'))-datetime.fromisoformat(attempt['timing_updated_at'].replace('Z', '+00:00'))).total_seconds()*1000)
    delta = total-attempt['active_duration_ms']
    # A small tolerance accommodates network jitter, but each sync is bounded.
    max_delta = min(elapsed+2000, 120000)
    if attempt['is_manually_paused'] or attempt['is_page_hidden']:
        max_delta = 0
    if delta < 0 or delta > max_delta:
        raise APIError('计时异常，请重新读取作答进度。', 409)
    changed = paused != attempt['is_manually_paused'] or hidden != attempt['is_page_hidden']
    db.execute('UPDATE question_attempts SET progress_seq=?,active_duration_ms=?,is_manually_paused=?,is_page_hidden=?,timing_updated_at=?,last_activity_at=?,updated_at=? WHERE id=?', (seq, total, paused, hidden, now, now if changed else attempt['last_activity_at'], now, attempt['id']))
    return True


@bp.patch('/attempts/<int:attempt_id>/progress')
def save_progress(attempt_id):
    data = body()
    with transaction() as db:
        attempt = owned('question_attempts', attempt_id)
        if attempt['status'] != 'in_progress':
            raise APIError('作答已结束。', 409)
        content(owned('questions', attempt['question_id']))
        applied = progress(db, attempt, data)
    return jsonify(attempt=owned('question_attempts', attempt_id), applied=applied)


def valid_answer(answer, question, snapshot):
    qtype = question['question_type']
    ids = {o['id'] for o in snapshot['options']}
    if qtype == 'single_choice':
        valid = isinstance(answer, str) and answer in ids
    elif qtype == 'multiple_choice':
        valid = isinstance(answer, list) and bool(answer) and all(isinstance(a, str) and a in ids for a in answer) and len(answer) == len(set(answer))
    else:
        valid = isinstance(answer, str) and bool(answer.strip()) and len(answer) <= 20000
    if not valid:
        raise APIError('请提交符合题型的非空答案。')


def grade(attempt_id):
    attempt = owned('question_attempts', attempt_id)
    question = owned('questions', attempt['question_id'])
    snapshot = json.loads(question['content_snapshot_json'])
    answer = json.loads(attempt['answer_json'])
    try:
        if question['question_type'] in ('single_choice', 'multiple_choice'):
            correct = answer == snapshot['answer'] if question['question_type'] == 'single_choice' else set(answer) == set(snapshot['answer'])
            selected = [answer] if isinstance(answer, str) else answer
            summary = '选择了：' + '；'.join(o['text'] for o in snapshot['options'] if o['id'] in selected)
            result = dict(score=float(correct), feedback_summary='按选项与标准答案比对。', answer_summary=summary[:1000], confidence='high', errors=[])
            method = 'rule'
            # Wrong choices alone do not justify a specific cognitive diagnosis.
            if not correct:
                result['errors'] = [dict(knowledge_point_id=question['primary_knowledge_point_id'], error_type='unknown', description='选项与标准答案不一致，具体原因待确认。', evidence_summary=f'提交选项：{dump(answer)}', confidence='unknown', source='rule')]
        else:
            material = owned('learning_materials', question['material_id'])
            result = provider().grade(question, snapshot, answer, json.loads(material['knowledge_points_json']))
            method = 'model'
        score = result['score']
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1 or result.get('confidence') not in CONFIDENCES:
            raise ValueError('Invalid grading result')
        feedback = string(result.get('feedback_summary'), '判分反馈', 2000)
        summary = string(result.get('answer_summary'), '答案摘要', 1000)
        errors = result.get('errors', [])
        if not isinstance(errors, list) or len(errors) > 10:
            raise ValueError('Invalid errors')
        allowed_points = {p['id'] for p in json.loads(owned('learning_materials', question['material_id'])['knowledge_points_json'])}
        normalized = []
        for error in errors:
            if not isinstance(error, dict) or type(error.get('knowledge_point_id')) is not int or error['knowledge_point_id'] not in allowed_points or error.get('error_type') not in ERROR_TYPES or error.get('confidence') not in CONFIDENCES:
                raise ValueError('Invalid error inference')
            normalized.append(dict(knowledge_point_id=error['knowledge_point_id'], error_type=error['error_type'], description=string(error.get('description'), '错因', 1000), evidence_summary=string(error.get('evidence_summary'), '证据', 1000), confidence=error['confidence'], source=method, suggestion=error.get('suggestion')))
        with transaction() as db:
            # updated_at acts as a grading lease; an older response cannot overwrite a retry.
            changed = db.execute("UPDATE question_attempts SET status='completed',score=?,is_correct=?,answer_summary=?,feedback_summary=?,grading_method=?,grading_version='v1',grading_confidence=?,updated_at=? WHERE id=? AND status='pending_grading' AND updated_at=?", (score, int(score == 1), summary, feedback, method, result['confidence'], utcnow(), attempt_id, attempt['updated_at'])).rowcount
            if changed:
                for error in normalized:
                    now = utcnow()
                    insert(db, 'error_records', dict(attempt_id=attempt_id, **error, created_at=now, updated_at=now))
    except Exception:
        current_app.logger.exception('Grading failed for attempt %s', attempt_id)
        with transaction() as db:
            db.execute("UPDATE question_attempts SET status='grading_failed',feedback_summary=?,updated_at=? WHERE id=? AND status='pending_grading' AND updated_at=?", ('判分失败，请重试；本次未记为答错。', utcnow(), attempt_id, attempt['updated_at']))


@bp.post('/attempts/<int:attempt_id>/submit')
def submit(attempt_id):
    data = body()
    key = string(data.get('submission_key'), '提交防重标识', 128)
    with transaction() as db:
        attempt = owned('question_attempts', attempt_id)
        if attempt['submission_key']:
            if attempt['submission_key'] != key or json.loads(attempt['answer_json']) != data.get('answer'):
                raise APIError('已提交，不能覆盖答案。', 409)
            return jsonify(attempt=attempt)
        if attempt['status'] != 'in_progress':
            raise APIError('作答已结束。', 409)
        question = owned('questions', attempt['question_id'])
        snapshot = content(question)
        valid_answer(data.get('answer'), question, snapshot)
        if 'progress' in data:
            if not isinstance(data['progress'], dict):
                raise APIError('进度格式无效。')
            progress(db, attempt, data['progress'])
        now = utcnow()
        db.execute('UPDATE questions SET content_snapshot_json=?,content_expires_at=NULL WHERE id=? AND content_snapshot_json IS NULL', (dump(snapshot), question['id']))
        db.execute("UPDATE question_attempts SET answer_json=?,submission_key=?,status='pending_grading',submitted_at=?,last_activity_at=?,updated_at=? WHERE id=?", (dump(data['answer']), key, now, now, now, attempt_id))
    grade(attempt_id)
    delete_cache(question['id'])
    return jsonify(attempt=owned('question_attempts', attempt_id))


@bp.post('/attempts/<int:attempt_id>/retry-grading')
def retry_grading(attempt_id):
    with transaction() as db:
        attempt = owned('question_attempts', attempt_id)
        stale = attempt['status'] == 'pending_grading' and (datetime.now(timezone.utc)-datetime.fromisoformat(attempt['updated_at'].replace('Z', '+00:00'))).total_seconds() > 180
        if attempt['status'] != 'grading_failed' and not stale:
            raise APIError('当前状态不能重试判分。', 409)
        db.execute("UPDATE question_attempts SET status='pending_grading',updated_at=? WHERE id=?", (utcnow(), attempt_id))
    grade(attempt_id)
    return jsonify(attempt=owned('question_attempts', attempt_id))


@bp.post('/attempts/<int:attempt_id>/abandon')
def abandon(attempt_id):
    data = body()
    with transaction() as db:
        attempt = owned('question_attempts', attempt_id)
        if attempt['status'] == 'abandoned':
            return jsonify(attempt=attempt)
        if attempt['status'] != 'in_progress':
            raise APIError('作答已结束。', 409)
        if 'progress' in data:
            progress(db, attempt, data['progress'])
        db.execute("UPDATE question_attempts SET status='abandoned',last_activity_at=?,updated_at=? WHERE id=?", (utcnow(), utcnow(), attempt_id))
    return jsonify(attempt=owned('question_attempts', attempt_id))


@bp.post('/attempts/<int:attempt_id>/hints')
def request_hint(attempt_id):
    data = body()
    key = string(data.get('request_key'), '提示防重标识', 128)
    kind = data.get('hint_type', 'concept')
    if kind not in ('concept', 'approach', 'step', 'answer'):
        raise APIError('提示类型无效。')
    level = integer(data.get('hint_level', 1), '提示强度', 1, 3)
    help_text = data.get('request_text')
    if help_text is not None:
        help_text = string(help_text, '求助内容', 1000)
    with transaction() as db:
        attempt = owned('question_attempts', attempt_id)
        prior = db.execute('SELECT * FROM hint_records WHERE attempt_id=? AND request_key=?', (attempt_id, key)).fetchone()
        if prior and (prior['hint_type'] != kind or prior['hint_level'] != level or prior['request_text'] != help_text):
            raise APIError('相同防重标识不能用于不同请求。', 409)
        if prior and prior['status'] != 'failed':
            if prior['status'] == 'requested' and (datetime.now(timezone.utc)-datetime.fromisoformat(prior['requested_at'].replace('Z', '+00:00'))).total_seconds() > 180:
                db.execute("UPDATE hint_records SET status='failed' WHERE id=?", (prior['id'],))
            else:
                return jsonify(hint=dict(prior))
        if attempt['status'] != 'in_progress' and not (kind == 'answer' and attempt['status'] in ('completed','grading_failed','pending_grading')):
            raise APIError('当前状态不能请求提示。', 409)
        question = owned('questions', attempt['question_id'])
        snapshot = content(question)
        if 'progress' in data and attempt['status'] == 'in_progress':
            progress(db, attempt, data['progress'])
            attempt = owned('question_attempts', attempt_id)
        if prior:
            hid = prior['id']
            db.execute("UPDATE hint_records SET status='requested',requested_at=? WHERE id=?", (utcnow(), hid))
        else:
            number = db.execute('SELECT COALESCE(MAX(sequence_no),0)+1 FROM hint_records WHERE attempt_id=?', (attempt_id,)).fetchone()[0]
            hid = insert(db, 'hint_records', dict(attempt_id=attempt_id, sequence_no=number, hint_type=kind, hint_level=level, request_text=help_text, status='requested', requested_at=utcnow(), active_elapsed_ms=attempt['active_duration_ms'], request_key=key, model_name=provider().model_name if kind != 'answer' else None, prompt_version='v1'))
        lease = owned('hint_records', hid)['requested_at']
    try:
        summary = dump(snapshot['answer']) if kind == 'answer' else string(provider().hint(snapshot, kind, level, help_text), '提示内容', 2000)
        with transaction() as db:
            db.execute("UPDATE hint_records SET status='generated',hint_summary=?,delivered_at=? WHERE id=? AND status='requested' AND requested_at=?", (summary, utcnow(), hid, lease))
    except Exception:
        current_app.logger.exception('Hint generation failed')
        with transaction() as db:
            db.execute("UPDATE hint_records SET status='failed' WHERE id=? AND status='requested' AND requested_at=?", (hid, lease))
    return jsonify(hint=owned('hint_records', hid))


@bp.post('/hints/<int:hint_id>/viewed')
def view_hint(hint_id):
    with transaction() as db:
        hint = owned('hint_records', hint_id)
        if hint['status'] not in ('generated', 'viewed'):
            raise APIError('提示尚未生成。', 409)
        now = utcnow()
        db.execute("UPDATE hint_records SET status='viewed',viewed_at=COALESCE(viewed_at,?) WHERE id=?", (now, hint_id))
        if hint['hint_type'] == 'answer':
            db.execute('UPDATE question_attempts SET answer_revealed_at=COALESCE(answer_revealed_at,?) WHERE id=?', (now, hint['attempt_id']))
    return jsonify(hint=owned('hint_records', hint_id))


@bp.get('/attempts/<int:attempt_id>')
def attempt_detail(attempt_id):
    attempt = owned('question_attempts', attempt_id)
    question = owned('questions', attempt['question_id'])
    public = public_question(question) if attempt['status'] == 'in_progress' or question['content_snapshot_json'] else None
    hints = [dict(r) for r in get_db().execute('SELECT * FROM hint_records WHERE attempt_id=? ORDER BY sequence_no', (attempt_id,))]
    # Full answer text is returned only by an explicit hint request.
    for hint in hints:
        if hint['hint_type'] == 'answer':
            hint['hint_summary'] = None
    return jsonify(attempt=attempt, question=public, hints=hints, errors=[dict(r) for r in get_db().execute('SELECT * FROM error_records WHERE attempt_id=? ORDER BY id', (attempt_id,))])


@bp.get('/attempts')
def attempts():
    try:
        limit = integer(int(request.args.get('limit', 50)), '数量', 1, 100)
        offset = integer(int(request.args.get('offset', 0)), '偏移量')
    except ValueError:
        raise APIError('分页参数无效。')
    expire_caches(g.user['id'])
    rows = get_db().execute('''SELECT a.*,q.question_summary,q.material_id,
      (SELECT COUNT(*) FROM hint_records h WHERE h.attempt_id=a.id AND h.status='viewed' AND h.hint_type!='answer') AS hint_count,
      (SELECT COUNT(*) FROM hint_records h WHERE h.attempt_id=a.id AND h.status='viewed' AND h.hint_type='answer') AS answer_view_count
      FROM question_attempts a JOIN questions q ON q.id=a.question_id WHERE a.user_id=? ORDER BY a.started_at DESC,a.id DESC LIMIT ? OFFSET ?''', (g.user['id'], limit, offset)).fetchall()
    return jsonify(attempts=[dict(r) for r in rows])


@bp.post('/attempts/<int:attempt_id>/errors')
def user_error(attempt_id):
    data = body()
    attempt = owned('question_attempts', attempt_id)
    if attempt['submitted_at'] is None:
        raise APIError('请先提交答案。', 409)
    question = owned('questions', attempt['question_id'])
    kp = integer(data.get('knowledge_point_id', question['primary_knowledge_point_id']), '知识点编号', 1)
    material = owned('learning_materials', question['material_id'])
    if kp not in {p['id'] for p in json.loads(material['knowledge_points_json'])}:
        raise APIError('知识点不属于来源资料。')
    kind = data.get('error_type', 'unknown')
    if not isinstance(kind, str) or kind not in ERROR_TYPES:
        raise APIError('错因类型无效。')
    with transaction() as db:
        now = utcnow()
        eid = insert(db, 'error_records', dict(attempt_id=attempt_id, knowledge_point_id=kp, error_type=kind, description=string(data.get('description'), '错因', 1000), evidence_summary=string(data.get('evidence_summary'), '作答证据', 1000), source='user', confidence='unknown', review_status='confirmed', created_at=now, updated_at=now))
    return jsonify(error_record=owned('error_records', eid)), 201


@bp.patch('/errors/<int:error_id>')
def review_error(error_id):
    data = body()
    owned('error_records', error_id)
    status = data.get('review_status')
    if status not in ('pending', 'confirmed', 'rejected'):
        raise APIError('确认状态无效。')
    with transaction() as db:
        db.execute('UPDATE error_records SET review_status=?,updated_at=? WHERE id=?', (status, utcnow(), error_id))
    return jsonify(error_record=owned('error_records', error_id))
