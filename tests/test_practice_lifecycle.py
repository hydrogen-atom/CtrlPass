"""Offline acceptance tests: real SQLite and Flask, deterministic model boundary."""
import io
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask
from backend.database import connect
from backend.practice import init_app

HEADERS = {'X-CtrlPass-Request': '1'}


class FakeProvider:
    model_name = 'test-model'
    fail_grade = False
    fail_hint = False

    def decide_next(self, goal, observation, trace, has_result):
        self.agent_observations = getattr(self, 'agent_observations', []) + [observation]
        if has_result:
            return dict(action='finish', arguments={}, reason='已观察到目标结果生成成功。', final_message='已根据资料完成任务。')
        if observation.get('requested_task') == 'qa':
            return dict(action='answer_from_material', arguments={'question': goal}, reason='用户需要资料问答。', final_message='')
        if observation.get('requested_task') == 'knowledge_graph':
            return dict(action='generate_knowledge_graph', arguments={'focus': goal}, reason='用户需要梳理知识关系。', final_message='')
        if 'point_stats' not in observation:
            return dict(action='inspect_learning_history', arguments={}, reason='先比较各知识点的历史表现。', final_message='')
        return dict(action='generate_practice_question', arguments={
            'primary_knowledge_point_id': 1,
            'question_type': 'single_choice',
            'difficulty_level': 3,
            'focus': '巩固二分查找的前置条件',
        }, reason='知识点一需要继续巩固。', final_message='')

    def process(self, material, config, root):
        return dict(chunk_count=3, vector_store_path=f'materials/{material["id"]}/vectors', knowledge_points=['二分查找', '边界条件'])

    def generate(self, material, kp, qtype, difficulty, context, root):
        self.context = context
        return dict(question='二分查找是否需要有序数据？', options=[{'id': 'A', 'text': '需要'}, {'id': 'B', 'text': '不需要'}],
                    answer=['A'] if qtype == 'multiple_choice' else 'A' if qtype == 'single_choice' else '需要有序数据',
                    scoring_rubric='指出有序要求', question_summary='考查二分查找的前置条件', source_refs=[{'chunk_id': 1, 'page': None}])

    def grade(self, question, snapshot, answer, points):
        if self.fail_grade:
            raise RuntimeError('Simulated model outage')
        return dict(score=0.5, feedback_summary='部分正确', answer_summary='提到了查找，但未说明有序要求', confidence='medium',
                    errors=[dict(knowledge_point_id=1, error_type='concept_missing', description='未说明有序要求', evidence_summary='回答仅提到查找', confidence='low')])

    def hint(self, snapshot, kind, level, text):
        if self.fail_hint:
            raise RuntimeError('Simulated hint outage')
        return '想一想如何排除一半的候选范围。'

    def answer(self, material, question, root):
        return {'answer': '二分查找要求数据有序。[片段1]', 'source_refs': [{'chunk_id': 1, 'page': None}]}

    def knowledge_graph(self, material, focus, root):
        return {
            'nodes': [
                {'id': 'binary-search', 'label': '二分查找', 'category': '算法', 'description': '逐步缩小范围'},
                {'id': 'sorted-data', 'label': '有序数据', 'category': '前提', 'description': '数据保持有序'},
            ],
            'edges': [
                {'source': 'binary-search', 'target': 'sorted-data', 'relation': '要求', 'evidence': '二分查找依赖有序数据'},
            ],
            'source_refs': [{'chunk_id': 1, 'page': None}],
        }


class PracticeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.app = Flask(__name__)
        self.app.config.update(TESTING=True, SECRET_KEY='test-secret', STORAGE_ROOT=str(self.root), DATABASE=str(self.root / 'test.sqlite3'))
        self.provider = FakeProvider()
        init_app(self.app, self.provider)
        self.client = self.app.test_client()
        self.register_login(self.client, 'alice')

    def tearDown(self):
        self.tmp.cleanup()

    def post(self, path, data=None, client=None):
        return (client or self.client).post('/api'+path, json={} if data is None else data, headers=HEADERS)

    def patch(self, path, data, client=None):
        return (client or self.client).patch('/api'+path, json=data, headers=HEADERS)

    def register_login(self, client, name):
        self.assertEqual(self.post('/auth/register', dict(username=name, password='password123'), client).status_code, 201)
        self.assertEqual(self.post('/auth/login', dict(username=name, password='password123'), client).status_code, 200)

    def sql(self, query, args=()):
        db = connect(self.root / 'test.sqlite3')
        try:
            return [dict(r) for r in db.execute(query, args).fetchall()]
        finally:
            db.close()

    def material(self):
        response = self.client.post('/api/materials/upload', data={'file': (io.BytesIO(b'hello binary search'), '../../sample.txt')}, headers=HEADERS)
        self.assertEqual(response.status_code, 201, response.json)
        material = response.json['material']
        self.assertEqual(self.post(f'/materials/{material["id"]}/process').status_code, 200)
        return material['id']

    def question(self, qtype='single_choice', mid=None):
        response = self.post('/questions/generate', dict(material_id=mid or self.material(), primary_knowledge_point_id=1, question_type=qtype))
        self.assertEqual(response.status_code, 201, response.json)
        self.assertNotIn('answer', response.json['question'])
        self.assertNotIn('content_snapshot_json', response.json['question'])
        return response.json['question']['id']

    def start(self, qid=None, qtype='single_choice'):
        qid = qid or self.question(qtype)
        response = self.post(f'/questions/{qid}/attempts')
        self.assertEqual(response.status_code, 201)
        return qid, response.json['attempt']['id']

    def submit(self, aid, answer='A', key='submission-one'):
        return self.post(f'/attempts/{aid}/submit', dict(answer=answer, submission_key=key))

    def progress(self, aid, seq, total=0, paused=0, hidden=0):
        return self.patch(f'/attempts/{aid}/progress', dict(progress_seq=seq, active_duration_ms=total, is_manually_paused=paused, is_page_hidden=hidden))

    def test_schema_has_six_business_tables_autoincrement_and_foreign_keys(self):
        tables = self.sql("SELECT name,sql FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence'")
        self.assertEqual(len(tables), 6)
        self.assertTrue(all('INTEGER PRIMARY KEY AUTOINCREMENT' in t['sql'] for t in tables))
        with self.assertRaises(sqlite3.IntegrityError):
            self.sql("INSERT INTO learning_materials(user_id,original_filename,file_type,file_size,file_path,content_hash,created_at) VALUES(999,'x','txt',1,'x','x','x')")

    def test_agent_observes_decides_calls_tools_and_observes_again(self):
        mid = self.material()
        response = self.post('/agent/run', dict(material_id=mid, goal='安排下一道练习'))
        self.assertEqual(response.status_code, 201, response.json)
        self.assertNotIn('answer', response.json['question'])
        self.assertEqual(response.json['agent']['tool_steps'], 2)
        phases = [event['phase'] for event in response.json['agent']['trace']]
        self.assertEqual(phases, ['observe', 'decide', 'act', 'observe', 'decide', 'act', 'observe', 'decide'])
        self.assertNotIn('point_stats', self.provider.agent_observations[0])
        self.assertIn('point_stats', self.provider.agent_observations[1])
        self.assertEqual(self.provider.agent_observations[2]['question']['id'], response.json['question']['id'])
        saved = self.sql('SELECT generation_context_json FROM questions WHERE id=?', (response.json['question']['id'],))[0]
        self.assertTrue(json.loads(saved['generation_context_json'])['agent_mode'])

    def test_agent_rejects_unknown_model_selected_tool(self):
        mid = self.material()
        with patch.object(self.provider, 'decide_next', return_value={
            'action': 'delete_material', 'arguments': {}, 'reason': 'bad', 'final_message': '',
        }):
            response = self.post('/agent/run', dict(material_id=mid))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.sql('SELECT COUNT(*) AS count FROM questions')[0]['count'], 0)

    def test_agent_answers_from_material_and_observes_answer(self):
        mid = self.material()
        response = self.post('/agent/run', dict(material_id=mid, task='qa', goal='二分查找有什么前提？'))
        self.assertEqual(response.status_code, 201, response.json)
        self.assertEqual(response.json['result']['type'], 'qa')
        self.assertIn('有序', response.json['result']['answer'])
        self.assertEqual(response.json['agent']['tool_steps'], 1)
        self.assertIn('answer', self.provider.agent_observations[-1])
        self.assertNotIn('question', response.json)

    def test_agent_generates_grounded_knowledge_graph(self):
        mid = self.material()
        response = self.post('/agent/run', dict(material_id=mid, task='knowledge_graph'))
        self.assertEqual(response.status_code, 201, response.json)
        graph = response.json['result']['graph']
        self.assertEqual(response.json['result']['type'], 'knowledge_graph')
        self.assertEqual(len(graph['nodes']), 2)
        self.assertEqual(graph['edges'][0]['relation'], '要求')
        self.assertIn('graph', self.provider.agent_observations[-1])

    def test_direct_material_qa_and_graph_endpoints(self):
        mid = self.material()
        answer = self.post(f'/materials/{mid}/ask', {'question': '二分查找的前提是什么？'})
        graph = self.post(f'/materials/{mid}/knowledge-graph', {'focus': '算法前提'})
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(answer.json['source_refs'][0]['chunk_id'], 1)
        self.assertEqual(graph.status_code, 200)
        self.assertEqual(graph.json['graph']['nodes'][0]['id'], 'binary-search')

    def test_login_timestamp_only_changes_on_successful_login(self):
        before = self.sql('SELECT * FROM users')[0]
        self.assertNotEqual(before['password_hash'], 'password123')
        self.assertTrue(before['last_login_at'].endswith('Z'))
        self.client.get('/api/materials')
        self.post('/auth/login', dict(username='alice', password='wrong-password'))
        self.assertEqual(self.sql('SELECT last_login_at FROM users')[0]['last_login_at'], before['last_login_at'])
        self.post('/auth/register', dict(username='bob', password='password123'))
        self.assertIsNone(self.sql("SELECT last_login_at FROM users WHERE username='bob'")[0]['last_login_at'])

    def test_auth_logout_disabled_and_csrf(self):
        self.assertEqual(self.client.post('/api/auth/logout', json={}).status_code, 403)
        self.assertEqual(self.client.post('/api/auth/logout', json={}, headers=HEADERS | {'Origin': 'https://evil.example'}).status_code, 403)
        self.sql("UPDATE users SET status='disabled'")
        self.assertEqual(self.client.get('/api/materials').status_code, 401)
        self.sql("UPDATE users SET status='active'")
        self.post('/auth/login', dict(username='alice', password='password123'))
        self.post('/auth/logout')
        self.assertEqual(self.client.get('/api/materials').status_code, 401)

    def test_owner_is_checked_for_all_record_types(self):
        qid, aid = self.start(qtype='short_answer')
        hid = self.post(f'/attempts/{aid}/hints', dict(request_key='hint1')).json['hint']['id']
        self.submit(aid, '查找')
        eid = self.sql('SELECT id FROM error_records')[0]['id']
        bob = self.app.test_client()
        self.register_login(bob, 'bob')
        for path in (f'/questions/{qid}/attempts', f'/attempts/{aid}/submit', f'/attempts/{aid}/abandon', f'/hints/{hid}/viewed', '/materials/1/process'):
            data = {'submission_key': 'stolen', 'answer': 'A'} if path.endswith('/submit') else {}
            self.assertEqual(self.post(path, data, bob).status_code, 404, path)
        self.assertEqual(self.patch(f'/errors/{eid}', dict(review_status='confirmed'), bob).status_code, 404)
        self.assertEqual(bob.get(f'/api/attempts/{aid}').status_code, 404)
        self.assertEqual(bob.get('/api/attempts').json['attempts'], [])
        self.assertEqual(bob.get('/api/materials').json['materials'], [])

    def test_material_path_hash_and_knowledge_point_ids(self):
        mid = self.material()
        m = self.sql('SELECT * FROM learning_materials WHERE id=?', (mid,))[0]
        self.assertEqual(m['file_path'], f'materials/{mid}/source.txt')
        self.assertEqual(len(m['content_hash']), 64)
        self.assertEqual([p['id'] for p in json.loads(m['knowledge_points_json'])], [1, 2])
        self.assertEqual(self.post('/questions/generate', dict(material_id=mid, primary_knowledge_point_id=999)).status_code, 400)

    def test_start_hint_and_invalid_answer_do_not_save_snapshot(self):
        qid, aid = self.start()
        self.progress(aid, 1, 100)
        h = self.post(f'/attempts/{aid}/hints', dict(request_key='h1')).json['hint']
        self.post(f'/hints/{h["id"]}/viewed')
        self.assertEqual(self.submit(aid, '').status_code, 400)
        self.assertIsNone(self.sql('SELECT content_snapshot_json FROM questions WHERE id=?', (qid,))[0]['content_snapshot_json'])

    def test_first_submit_snapshot_and_redo_are_immutable(self):
        qid, aid = self.start()
        result = self.submit(aid)
        self.assertEqual(result.json['attempt']['status'], 'completed')
        snapshot = self.sql('SELECT content_snapshot_json FROM questions')[0]['content_snapshot_json']
        self.assertIsNotNone(snapshot)
        path = self.root / f'cache/questions/{qid}.json'
        if path.exists():
            path.unlink()
        _, second = self.start(qid)
        self.assertNotEqual(aid, second)
        self.assertEqual(self.sql('SELECT attempt_no FROM question_attempts WHERE id=?', (second,))[0]['attempt_no'], 2)
        self.submit(second, key='submission-two')
        self.assertEqual(self.sql('SELECT content_snapshot_json FROM questions')[0]['content_snapshot_json'], snapshot)
        with self.assertRaises(sqlite3.IntegrityError):
            self.sql("UPDATE questions SET content_snapshot_json='{}' WHERE id=?", (qid,))

    def test_atomic_submit_rolls_back_snapshot_if_answer_write_fails(self):
        qid, aid = self.start()
        self.sql("CREATE TRIGGER test_failure BEFORE UPDATE OF answer_json ON question_attempts BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        self.assertEqual(self.submit(aid).status_code, 409)
        self.assertIsNone(self.sql('SELECT content_snapshot_json FROM questions')[0]['content_snapshot_json'])
        self.assertEqual(self.sql('SELECT status FROM question_attempts')[0]['status'], 'in_progress')

    def test_idempotent_submit_and_conflicting_keys(self):
        _, aid = self.start()
        self.assertEqual(self.submit(aid).status_code, 200)
        self.assertEqual(self.submit(aid).status_code, 200)
        self.assertEqual(self.submit(aid, 'B').status_code, 409)
        self.assertEqual(self.submit(aid, key='different').status_code, 409)
        _, aid2 = self.start()
        self.assertEqual(self.submit(aid2).status_code, 409)
        q2 = self.sql('SELECT question_id FROM question_attempts WHERE id=?', (aid2,))[0]['question_id']
        self.assertIsNone(self.sql('SELECT content_snapshot_json FROM questions WHERE id=?', (q2,))[0]['content_snapshot_json'])

    def test_model_failure_preserves_snapshot_and_ungraded_nulls_then_retry(self):
        _, aid = self.start(qtype='short_answer')
        self.provider.fail_grade = True
        result = self.submit(aid, '查找').json['attempt']
        self.assertEqual(result['status'], 'grading_failed')
        self.assertIsNone(result['score']); self.assertIsNone(result['is_correct'])
        self.assertIsNotNone(self.sql('SELECT content_snapshot_json FROM questions')[0]['content_snapshot_json'])
        self.provider.fail_grade = False
        self.assertEqual(self.post(f'/attempts/{aid}/retry-grading').json['attempt']['status'], 'completed')
        self.assertEqual(len(self.sql('SELECT * FROM error_records')), 1)
        self.assertEqual(self.post(f'/attempts/{aid}/retry-grading').status_code, 409)

    def test_progress_order_pause_hidden_and_anomalies(self):
        _, aid = self.start()
        self.assertTrue(self.progress(aid, 2, 100, paused=1).json['applied'])
        self.assertFalse(self.progress(aid, 1, 0).json['applied'])
        self.assertFalse(self.progress(aid, 2, 999).json['applied'])
        self.assertEqual(self.progress(aid, 3, 101, paused=1).status_code, 409)
        self.assertEqual(self.progress(aid, 3, 100, paused=1, hidden=1).status_code, 200)
        self.assertEqual(self.progress(aid, 4, 100, paused=0, hidden=1).status_code, 200)
        self.assertEqual(self.progress(aid, 5, 101, paused=0, hidden=1).status_code, 409)
        self.assertEqual(self.progress(aid, 5, 100, paused=0, hidden=0).status_code, 200)
        self.assertEqual(self.progress(aid, 6, 1000000).status_code, 409)
        self.assertEqual(self.progress(aid, 6, 99).status_code, 409)
        self.assertEqual(self.progress(aid, 6, 150).status_code, 200)

    def test_heartbeat_does_not_update_activity_timestamp(self):
        _, aid = self.start()
        self.sql("UPDATE question_attempts SET last_activity_at='2020-01-01T00:00:00.000Z' WHERE id=?", (aid,))
        self.progress(aid, 1, 100)
        self.assertEqual(self.sql('SELECT last_activity_at FROM question_attempts')[0]['last_activity_at'], '2020-01-01T00:00:00.000Z')
        self.progress(aid, 2, 100, paused=1)
        self.assertNotEqual(self.sql('SELECT last_activity_at FROM question_attempts')[0]['last_activity_at'], '2020-01-01T00:00:00.000Z')

    def test_deleted_cache_expires_question_and_stops_attempt(self):
        qid, aid = self.start()
        (self.root / f'cache/questions/{qid}.json').unlink()
        self.assertEqual(self.submit(aid).status_code, 410)
        self.assertEqual(self.sql('SELECT status FROM questions')[0]['status'], 'expired')
        self.assertEqual(self.sql('SELECT status FROM question_attempts')[0]['status'], 'expired')
        self.assertEqual(self.post(f'/questions/{qid}/attempts').status_code, 410)

    def test_ttl_expiry_is_visible_in_history(self):
        qid, aid = self.start()
        self.sql("UPDATE questions SET content_expires_at='2000-01-01T00:00:00.000Z' WHERE id=?", (qid,))
        history = self.client.get('/api/attempts').json['attempts']
        self.assertEqual(history[0]['status'], 'expired')

    def test_hints_acknowledgement_failure_retry_and_answer_separate(self):
        qid, aid = self.start()
        self.provider.fail_hint = True
        req = dict(request_key='h1')
        failed = self.post(f'/attempts/{aid}/hints', req).json['hint']
        self.assertEqual(failed['status'], 'failed')
        self.provider.fail_hint = False
        hint = self.post(f'/attempts/{aid}/hints', req).json['hint']
        self.assertEqual(failed['id'], hint['id'])
        self.assertEqual(self.post(f'/attempts/{aid}/hints', req).json['hint']['id'], hint['id'])
        self.assertEqual(self.post(f'/attempts/{aid}/hints', req | {'hint_type': 'step'}).status_code, 409)
        self.assertEqual(self.client.get('/api/attempts').json['attempts'][0]['hint_count'], 0)
        self.post(f'/hints/{hint["id"]}/viewed')
        answer = self.post(f'/attempts/{aid}/hints', dict(request_key='answer', hint_type='answer')).json['hint']
        self.assertIsNone(self.sql('SELECT answer_revealed_at FROM question_attempts')[0]['answer_revealed_at'])
        self.post(f'/hints/{answer["id"]}/viewed')
        row = self.client.get('/api/attempts').json['attempts'][0]
        self.assertEqual((row['hint_count'], row['answer_view_count']), (1, 1))
        self.assertIsNotNone(row['answer_revealed_at'])
        self.assertIsNone(self.sql('SELECT content_snapshot_json FROM questions')[0]['content_snapshot_json'])

    def test_abandon_stops_progress_and_no_snapshot(self):
        _, aid = self.start()
        self.assertEqual(self.post(f'/attempts/{aid}/abandon').status_code, 200)
        self.assertEqual(self.post(f'/attempts/{aid}/abandon').status_code, 200)
        self.assertEqual(self.progress(aid, 1).status_code, 409)
        self.assertEqual(self.submit(aid).status_code, 409)
        self.assertIsNone(self.sql('SELECT content_snapshot_json FROM questions')[0]['content_snapshot_json'])

    def test_user_error_does_not_replace_inference_and_rejected_is_not_reused(self):
        qid, aid = self.start(qtype='short_answer')
        self.submit(aid, '查找')
        inferred = self.sql('SELECT id FROM error_records')[0]['id']
        self.patch(f'/errors/{inferred}', dict(review_status='rejected'))
        self.assertEqual(self.post(f'/attempts/{aid}/errors', dict(description='我没看清题', evidence_summary='遗漏了有序条件')).status_code, 201)
        self.assertEqual(len(self.sql('SELECT * FROM error_records')), 2)
        mid = self.sql('SELECT material_id FROM questions WHERE id=?', (qid,))[0]['material_id']
        self.question(mid=mid)
        self.assertEqual(len(self.provider.context['history']), 1)
        self.assertEqual(len(self.provider.context['history'][0]['errors']), 1)

    def test_multiple_choice_validation(self):
        _, aid = self.start(qtype='multiple_choice')
        for invalid in ([], ['A', 'A'], ['Z'], 'A', [1]):
            self.assertEqual(self.submit(aid, invalid).status_code, 400)
        self.assertEqual(self.submit(aid, ['A']).json['attempt']['score'], 1)

    def test_pending_grading_recovery_after_process_interruption(self):
        _, aid = self.start(qtype='short_answer')
        self.submit(aid, '查找')
        old = (datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat().replace('+00:00','Z')
        self.sql("UPDATE question_attempts SET status='pending_grading',updated_at=? WHERE id=?", (old, aid))
        self.sql('DELETE FROM error_records')
        self.assertEqual(self.post(f'/attempts/{aid}/retry-grading').json['attempt']['status'], 'completed')

    def test_concurrent_start_creates_one_open_attempt(self):
        qid = self.question()
        def start_in_client(_):
            with self.app.test_client() as client:
                self.post('/auth/login', dict(username='alice', password='password123'), client)
                return self.post(f'/questions/{qid}/attempts', client=client).json['attempt']['id']
        with ThreadPoolExecutor(max_workers=3) as pool:
            ids = list(pool.map(start_in_client, range(3)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(self.sql('SELECT * FROM question_attempts')), 1)

    def test_concurrent_submit_records_one_snapshot_and_one_grading(self):
        _, aid = self.start(qtype='short_answer')
        def submit_in_client(_):
            with self.app.test_client() as client:
                self.post('/auth/login', dict(username='alice', password='password123'), client)
                return self.post(f'/attempts/{aid}/submit', dict(answer='查找', submission_key='same-request'), client).status_code
        with ThreadPoolExecutor(max_workers=3) as pool:
            statuses = list(pool.map(submit_in_client, range(3)))
        self.assertEqual(statuses, [200, 200, 200])
        self.assertEqual(len(self.sql('SELECT * FROM error_records')), 1)

    def test_knowledge_point_rename_keeps_ids_and_question_relationship(self):
        qid, _ = self.start()
        result = self.patch('/materials/1/knowledge-points/1', dict(name='二分查找基础'))
        self.assertEqual(result.json['knowledge_points'][0], {'id': 1, 'name': '二分查找基础'})
        self.assertEqual(self.sql('SELECT primary_knowledge_point_id FROM questions WHERE id=?', (qid,))[0]['primary_knowledge_point_id'], 1)
        self.assertEqual(self.patch('/materials/1/knowledge-points/1', dict(name='边界条件')).status_code, 409)

    def test_cache_cleanup_removes_expired_content(self):
        qid = self.question()
        self.sql("UPDATE questions SET content_expires_at='2000-01-01T00:00:00.000Z' WHERE id=?", (qid,))
        result = self.app.test_cli_runner().invoke(args=['cleanup-question-cache'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse((self.root / f'cache/questions/{qid}.json').exists())

    def test_model_result_malformed_does_not_mark_answer_incorrect(self):
        _, aid = self.start(qtype='short_answer')
        with patch.object(self.provider, 'grade', return_value={'score': float('nan'), 'confidence': 'low'}):
            result = self.submit(aid, '查找').json['attempt']
        self.assertEqual(result['status'], 'grading_failed')
        self.assertIsNone(result['is_correct'])
        self.assertIsNone(result['score'])

    def test_answer_view_during_grading_does_not_invalidate_grading_lease(self):
        _, aid = self.start(qtype='short_answer')
        original = self.provider.grade
        def grade_and_reveal(*args):
            client = self.app.test_client()
            self.post('/auth/login', dict(username='alice', password='password123'), client)
            hint = self.post(f'/attempts/{aid}/hints', dict(request_key='reveal', hint_type='answer'), client).json['hint']
            self.post(f'/hints/{hint["id"]}/viewed', client=client)
            return original(*args)
        with patch.object(self.provider, 'grade', side_effect=grade_and_reveal):
            result = self.submit(aid, '查找')
        self.assertEqual(result.json['attempt']['status'], 'completed')

    def test_api_rejects_malformed_parameters(self):
        mid = self.material()
        self.assertEqual(self.post('/questions/generate', dict(material_id=mid, primary_knowledge_point_id=1, question_type=[])).status_code, 400)
        self.assertEqual(self.post(f'/materials/{mid}/process', dict(chunk_overlap=501)).status_code, 400)
        _, aid = self.start()
        self.assertEqual(self.progress(aid, True).status_code, 400)
        self.assertEqual(self.post(f'/attempts/{aid}/submit', dict(submission_key='x', answer='A', progress=[])).status_code, 400)

    def test_material_processing_failure_and_interrupted_job_retry(self):
        uploaded = self.client.post('/api/materials/upload', data={'file': (io.BytesIO(b'example'), 'example.txt')}, headers=HEADERS).json['material']
        mid = uploaded['id']
        with patch.object(self.provider, 'process', side_effect=RuntimeError('offline')):
            self.assertEqual(self.post(f'/materials/{mid}/process').status_code, 502)
        self.assertEqual(self.sql('SELECT status FROM learning_materials WHERE id=?', (mid,))[0]['status'], 'failed')
        recent = json.dumps({'_processing_started_at': datetime.now(timezone.utc).isoformat()})
        self.sql("UPDATE learning_materials SET status='processing',processing_config_json=? WHERE id=?", (recent, mid))
        self.assertEqual(self.post(f'/materials/{mid}/process').status_code, 409)
        stale = json.dumps({'_processing_started_at': (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()})
        self.sql('UPDATE learning_materials SET processing_config_json=? WHERE id=?', (stale, mid))
        self.assertEqual(self.post(f'/materials/{mid}/process').status_code, 200)

    def test_submit_failure_recovery_survives_new_application_instance(self):
        qid, aid = self.start(qtype='short_answer')
        self.provider.fail_grade = True
        self.submit(aid, '查找')
        other_app = Flask('restarted')
        other_app.config.update(TESTING=True, SECRET_KEY='test-secret', STORAGE_ROOT=str(self.root), DATABASE=str(self.root / 'test.sqlite3'))
        init_app(other_app, FakeProvider())
        client = other_app.test_client()
        self.post('/auth/login', dict(username='alice', password='password123'), client)
        self.assertEqual(client.get(f'/api/questions/{qid}').status_code, 200)
        self.assertEqual(self.post(f'/attempts/{aid}/retry-grading', client=client).json['attempt']['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
