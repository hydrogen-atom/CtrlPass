import { useCallback, useEffect, useRef, useState } from 'react';
import { call, labels, messageOf, RequestError, requestKey } from '../api/practice';
import type { Attempt, Detail, ErrorRecord, Hint, Material, Point, Progress, Question, User } from '../api/practice';
import './PracticeApp.css';

const duration = (ms: number) => `${Math.floor(ms / 60000)} 分 ${Math.floor(ms / 1000) % 60} 秒`;
const loadWorkspace = (offset: number) => Promise.all([
  call<{ materials: Material[] }>('/materials'), call<{ attempts: Attempt[] }>(`/attempts?offset=${offset}`),
]);

function Auth({ onLogin }: { onLogin: (user: User) => void }) {
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  return <form className="study-auth study-card" onSubmit={async event => {
    event.preventDefault(); setBusy(true); setError('');
    const data = Object.fromEntries(new FormData(event.currentTarget));
    try {
      if (register) await call('/auth/register', 'POST', data);
      const result = await call<{ user: User }>('/auth/login', 'POST', data);
      onLogin(result.user);
    } catch (e) { setError(messageOf(e)); } finally { setBusy(false); }
  }}>
    <p className="study-eyebrow">CtrlPass · 学习工作台</p>
    <h1>{register ? '创建学习账号' : '继续你的学习'}</h1>
    <p>登录后，资料、作答进度和学习记录会保存在你的账号下。</p>
    <label>登录名<input name="username" required maxLength={80} autoComplete="username" /></label>
    {register && <label>显示名称<input name="display_name" maxLength={80} /></label>}
    <label>密码<input name="password" type="password" required minLength={8} maxLength={256} autoComplete={register ? 'new-password' : 'current-password'} /></label>
    {error && <p role="alert" className="study-error">{error}</p>}
    <button disabled={busy}>{busy ? '请稍候…' : register ? '注册并登录' : '登录'}</button>
    <button type="button" className="secondary" disabled={busy} onClick={() => setRegister(!register)}>{register ? '已有账号，去登录' : '没有账号，创建账号'}</button>
    <small>这里使用独立学习账号，暂未连接 WebForms 账号。</small>
  </form>;
}

function AttemptView({ initial, onChanged, onClose }: { initial: Detail; onChanged: () => void; onClose: () => void }) {
  const [detail, setDetail] = useState(initial);
  const [answer, setAnswer] = useState<string | string[]>(initial.attempt.answer_json ? JSON.parse(initial.attempt.answer_json) : initial.question?.question_type === 'multiple_choice' ? [] : '');
  const [displayMs, setDisplayMs] = useState(initial.attempt.active_duration_ms);
  const [paused, setPaused] = useState(Boolean(initial.attempt.is_manually_paused));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [hintLevel, setHintLevel] = useState(1);
  const [helpText, setHelpText] = useState('');
  const [selfError, setSelfError] = useState('');
  const [selfEvidence, setSelfEvidence] = useState('');
  const attempt = detail.attempt;
  const question = detail.question;
  const active = attempt.status === 'in_progress';
  const [submissionKey] = useState(() => initial.attempt.submission_key || requestKey());
  const clock = useRef({ total: initial.attempt.active_duration_ms, seq: initial.attempt.progress_seq, paused: Boolean(initial.attempt.is_manually_paused), hidden: Boolean(initial.attempt.is_page_hidden), running: initial.attempt.status === 'in_progress', at: 0 });
  const queue = useRef<Promise<unknown>>(Promise.resolve());
  const alive = useRef(true);
  const enqueue = useCallback(<T,>(work: () => Promise<T>): Promise<T> => {
    const next = queue.current.then(work, work);
    queue.current = next.catch(() => undefined);
    return next;
  }, []);
  const collect = useCallback(() => {
    const c = clock.current;
    const now = performance.now();
    if (c.running && !c.paused && !c.hidden) c.total += Math.max(0, now - c.at);
    c.at = now;
    return c;
  }, []);
  const packet = useCallback((capture = true): Progress => {
    const c = capture ? collect() : clock.current;
    return { progress_seq: ++c.seq, active_duration_ms: Math.floor(c.total), is_manually_paused: Number(c.paused), is_page_hidden: Number(c.hidden) };
  }, [collect]);
  const recover = useCallback(async (e: unknown) => {
    if (alive.current) setError(messageOf(e));
    try {
      const result = await call<Detail>(`/attempts/${initial.attempt.id}`);
      const c = clock.current;
      c.total = result.attempt.active_duration_ms; c.seq = Math.max(c.seq, result.attempt.progress_seq);
      c.paused = Boolean(result.attempt.is_manually_paused); c.hidden = Boolean(result.attempt.is_page_hidden);
      c.running = result.attempt.status === 'in_progress'; c.at = performance.now();
      if (c.running && alive.current && c.hidden !== document.hidden) {
        await call(`/attempts/${initial.attempt.id}/progress`, 'PATCH', {
          progress_seq: ++c.seq, active_duration_ms: c.total, is_manually_paused: Number(c.paused), is_page_hidden: Number(document.hidden),
        });
        c.hidden = document.hidden; c.at = performance.now();
      }
      if (alive.current) {
        setDetail(result); setPaused(c.paused); setDisplayMs(c.total);
        if (result.attempt.answer_json !== null) setAnswer(JSON.parse(result.attempt.answer_json));
      }
    } catch (failure) {
      if (failure instanceof RequestError && failure.status === 410) {
        clock.current.running = false;
        if (alive.current) setDetail(d => ({ ...d, attempt: { ...d.attempt, status: 'expired' } }));
      }
    }
  }, [initial.attempt.id]);
  const sync = useCallback((capture = true) => {
    if (!clock.current.running) return Promise.resolve();
    const payload = packet(capture);
    return enqueue(async () => {
      try {
        await call(`/attempts/${initial.attempt.id}/progress`, 'PATCH', payload);
      } catch (e) { await recover(e); throw e; }
    });
  }, [enqueue, initial.attempt.id, packet, recover]);
  useEffect(() => {
    alive.current = true;
    const timerState = clock.current;
    // Start with server flags, then explicitly restore the browser visibility state.
    clock.current.hidden = document.hidden;
    clock.current.at = performance.now();
    void sync(false).catch(() => undefined);
    const update = () => { setDisplayMs(collect().total); };
    const tick = window.setInterval(update, 1000);
    const heartbeat = window.setInterval(() => { void sync().catch(() => undefined); }, 15000);
    const visibility = () => { collect(); clock.current.hidden = document.hidden; void sync(false).catch(() => undefined); };
    const pagehide = () => {
      collect(); clock.current.hidden = true;
      // Best effort, same-origin credentials and the usual CSRF header are retained.
      if (clock.current.running) void fetch(`${import.meta.env.VITE_API_URL || ''}/api/attempts/${initial.attempt.id}/progress`, {
        method: 'PATCH', credentials: 'include', keepalive: true,
        headers: { 'Content-Type': 'application/json', 'X-CtrlPass-Request': '1' }, body: JSON.stringify(packet()),
      }).catch(() => undefined);
    };
    document.addEventListener('visibilitychange', visibility);
    window.addEventListener('pagehide', pagehide);
    return () => {
      alive.current = false; clearInterval(tick); clearInterval(heartbeat);
      document.removeEventListener('visibilitychange', visibility); window.removeEventListener('pagehide', pagehide);
      collect(); timerState.hidden = true; void sync().catch(() => undefined);
    };
  }, [collect, initial.attempt.id, packet, sync]);
  useEffect(() => {
    // Acknowledge only after React has rendered the delivered help.
    const delivered = detail.hints.filter(h => h.status === 'generated' && h.hint_summary !== null);
    for (const hint of delivered) {
      void call<{ hint: Hint }>(`/hints/${hint.id}/viewed`, 'POST', {}).then(result => {
        setDetail(d => ({ ...d, hints: d.hints.map(h => h.id === hint.id ? result.hint : h) }));
      }).catch(e => setError(messageOf(e)));
    }
  }, [detail.hints]);

  const refresh = async () => {
    const result = await call<Detail>(`/attempts/${attempt.id}`);
    clock.current.running = result.attempt.status === 'in_progress';
    if (alive.current) setDetail(result);
    onChanged();
  };
  const action = async (work: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await work(); } catch (e) { await recover(e); } finally { if (alive.current) setBusy(false); }
  };
  const finish = (kind: 'submit' | 'abandon') => action(async () => {
    const progress = packet();
    clock.current.running = false;
    await enqueue(async () => {
      await call(`/attempts/${attempt.id}/${kind}`, 'POST', kind === 'submit' ? { submission_key: submissionKey, answer, progress } : { progress });
      await refresh();
    });
  });
  const hint = (kind: string, prior?: Hint) => action(async () => {
    const progress = active ? packet() : undefined;
    const result = await enqueue(() => call<{ hint: Hint }>(`/attempts/${attempt.id}/hints`, 'POST', {
      hint_type: kind, hint_level: prior?.hint_level || hintLevel, request_key: prior?.request_key || requestKey(),
      request_text: prior ? prior.request_text : helpText.trim() || null, ...(progress ? { progress } : {}),
    }));
    setDetail(d => ({ ...d, hints: [...d.hints.filter(h => h.id !== result.hint.id), result.hint].sort((a, b) => a.id - b.id) }));
  });
  return <section className="study-card study-attempt">
    <div className="study-row"><h2>第 {attempt.attempt_no} 次作答</h2><span className="study-tag">{labels[attempt.status]}</span></div>
    <div className="study-row"><span aria-live="off">有效用时：{duration(displayMs)} {active && paused ? ' · 已暂停' : ''}</span>
      {active && <button className="secondary" disabled={busy} onClick={() => {
        collect(); clock.current.paused = !clock.current.paused; setPaused(clock.current.paused); void sync(false).catch(() => undefined);
      }}>{paused ? '继续' : '暂停'}</button>}
      <button className="secondary" disabled={busy} onClick={() => void action(async () => {
        if (active) { collect(); clock.current.paused = true; setPaused(true); await sync(); }
        onChanged(); onClose();
      })}>返回列表{active ? '并暂停' : ''}</button>
    </div>
    {question && <>
      <p className="study-meta">{labels[question.question_type]} · 预计难度 {question.difficulty_level}/5（模型估计）</p>
      <h3 className="study-question">{question.question}</h3>
      <fieldset disabled={!active || busy || paused}>
        <legend>你的答案</legend>
        {question.options.length ? question.options.map(option => <label className="study-option" key={option.id}>
          <input type={question.question_type === 'multiple_choice' ? 'checkbox' : 'radio'} name="answer"
            checked={Array.isArray(answer) ? answer.includes(option.id) : answer === option.id}
            onChange={event => setAnswer(question.question_type === 'multiple_choice' ? event.target.checked ? [...(Array.isArray(answer) ? answer : []), option.id] : (Array.isArray(answer) ? answer : []).filter(id => id !== option.id) : option.id)} />
          <span>{option.id}. {option.text}</span>
        </label>) : <textarea aria-label="你的答案" rows={5} maxLength={20000} value={typeof answer === 'string' ? answer : ''} onChange={e => setAnswer(e.target.value)} placeholder="输入答案，提交前的草稿仅保留在当前页面" />}
      </fieldset>
    </>}
    {error && <p role="alert" className="study-error">{error}</p>}
    {active && <>
      <div className="study-row"><button disabled={busy || paused || !answer.length} onClick={() => void finish('submit')}>{busy ? '处理中…' : '提交答案'}</button><button className="secondary" disabled={busy} onClick={() => void finish('abandon')}>放弃本次作答</button></div>
      <div className="study-help"><h3>需要帮助</h3><input aria-label="求助内容" value={helpText} maxLength={1000} onChange={e => setHelpText(e.target.value)} placeholder="可选：描述你卡住的地方" />
        <div className="study-row"><label>提示强度 <select value={hintLevel} onChange={e => setHintLevel(Number(e.target.value))}>{[1, 2, 3].map(n => <option key={n} value={n}>{n}</option>)}</select></label>
          {['concept', 'approach', 'step'].map(kind => <button className="secondary" disabled={busy} key={kind} onClick={() => void hint(kind)}>{labels[kind]}</button>)}
        </div>
      </div>
    </>}
    {!['abandoned', 'expired'].includes(attempt.status) && <button className="secondary" disabled={busy} onClick={() => void hint('answer')}>查看完整答案（单独记录）</button>}
    {detail.hints.map(h => <div className="study-hint" key={h.id}>
      <strong>{labels[h.hint_type]} · {h.status === 'failed' ? '失败' : labels[h.status]}</strong>
      {h.hint_summary && <p>{h.hint_summary}</p>}
      {(h.status === 'failed' || h.status === 'requested') && <button className="secondary" disabled={busy} onClick={() => void hint(h.hint_type, h)}>重试 / 获取结果</button>}
    </div>)}
    {attempt.score !== null && <p className="study-score">得分：{Math.round(attempt.score * 100)}% · 判分可信程度：{labels[attempt.grading_confidence]}</p>}
    {attempt.feedback_summary && <p>{attempt.feedback_summary}</p>}
    {['grading_failed', 'pending_grading'].includes(attempt.status) && <div className="study-row"><button disabled={busy} onClick={() => void action(async () => { await call(`/attempts/${attempt.id}/retry-grading`, 'POST', {}); await refresh(); })}>重试判分</button><button className="secondary" disabled={busy} onClick={() => void action(refresh)}>刷新判分状态</button><small>待判分超过 3 分钟后可重试；失败不会记为答错。</small></div>}
    {detail.errors.map(record => <div className="study-hint" key={record.id}>
      <strong>{record.description}</strong><p>作答证据：{record.evidence_summary}</p>
      <small>{labels[record.source]} · 可信程度：{labels[record.confidence]} · {labels[record.review_status]}</small>
      <div className="study-row">{['confirmed', 'rejected', 'pending'].map(status => <button className="secondary" key={status} disabled={busy || record.review_status === status} onClick={() => void action(async () => {
        const result = await call<{ error_record: ErrorRecord }>(`/errors/${record.id}`, 'PATCH', { review_status: status });
        setDetail(d => ({ ...d, errors: d.errors.map(e => e.id === record.id ? result.error_record : e) }));
      })}>{labels[status]}</button>)}</div>
    </div>)}
    {attempt.submitted_at && <div className="study-help"><h3>补充自己的错因</h3><input aria-label="自己的错因" value={selfError} onChange={e => setSelfError(e.target.value)} maxLength={1000} placeholder="例如：我把结束边界看错了" /><input aria-label="支持错因的作答证据" value={selfEvidence} onChange={e => setSelfEvidence(e.target.value)} maxLength={1000} placeholder="作答中哪些内容支持这个判断" /><button disabled={busy || !selfError.trim() || !selfEvidence.trim()} onClick={() => void action(async () => {
      await call(`/attempts/${attempt.id}/errors`, 'POST', { description: selfError, evidence_summary: selfEvidence }); setSelfError(''); setSelfEvidence(''); await refresh();
    })}>保存自述</button></div>}
  </section>;
}

function Workspace() {
  const [materials, setMaterials] = useState<Material[]>([]);
  const [history, setHistory] = useState<Attempt[]>([]);
  const [selected, setSelected] = useState(0);
  const [point, setPoint] = useState(0);
  const [type, setType] = useState('single_choice');
  const [difficulty, setDifficulty] = useState(3);
  const [goal, setGoal] = useState('');
  const [detail, setDetail] = useState<Detail | null>(null);
  const [generated, setGenerated] = useState<Question | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [offset, setOffset] = useState(0);
  const [chunkSize, setChunkSize] = useState(500);
  const [overlap, setOverlap] = useState(100);
  const [modelSplit, setModelSplit] = useState(false);
  const refresh = useCallback(async () => {
    const [m, a] = await loadWorkspace(offset);
    setMaterials(m.materials); setHistory(a.attempts);
  }, [offset]);
  useEffect(() => {
    let cancelled = false;
    void loadWorkspace(offset).then(([m, a]) => {
      if (!cancelled) { setMaterials(m.materials); setHistory(a.attempts); }
    }).catch(e => { if (!cancelled) setError(messageOf(e)); });
    return () => { cancelled = true; };
  }, [offset]);
  const material = materials.find(m => m.id === selected);
  const points: Point[] = material ? JSON.parse(material.knowledge_points_json) : [];
  const run = async (work: () => Promise<void>) => {
    setBusy(true); setError(''); setNotice('');
    try { await work(); } catch (e) { setError(messageOf(e)); } finally { setBusy(false); }
  };
  const start = async (qid: number) => {
    const result = await call<{ attempt: Attempt }>(`/questions/${qid}/attempts`, 'POST', { is_page_hidden: Number(document.hidden) });
    setDetail(await call<Detail>(`/attempts/${result.attempt.id}`)); setGenerated(null); await refresh();
  };
  if (detail) return <AttemptView key={detail.attempt.id} initial={detail} onChanged={() => { void refresh().catch(e => setError(messageOf(e))); }} onClose={() => setDetail(null)} />;
  return <>
    <div className="study-grid"><section className="study-card"><h2>1 · 学习资料</h2>
      <label>上传资料<input type="file" accept=".txt,.pdf,.docx" disabled={busy} onChange={event => {
        const file = event.target.files?.[0]; event.target.value = ''; if (!file) return;
        void run(async () => {
          const form = new FormData(); form.append('file', file);
          const result = await call<{ material: Material; duplicate_material_id: number | null }>('/materials/upload', 'POST', form);
          setSelected(result.material.id); setPoint(0); await refresh();
          setNotice(result.duplicate_material_id ? `已上传，内容与资料 ${result.duplicate_material_id} 相同。` : '已上传，请处理资料。');
        });
      }} /></label>
      <label>选择资料<select value={selected} onChange={e => { setSelected(Number(e.target.value)); setPoint(0); }}><option value={0}>请选择</option>{materials.map(m => <option key={m.id} value={m.id}>{m.original_filename} · {labels[m.status]}</option>)}</select></label>
      {material && <p>{labels[material.status]} · {material.chunk_count} 个文本片段 {material.error_message}</p>}
      <div className="study-row"><label>分段大小<input type="number" min={100} max={10000} value={chunkSize} onChange={e => setChunkSize(Number(e.target.value))} /></label><label>重叠长度<input type="number" min={0} max={chunkSize - 1} value={overlap} onChange={e => setOverlap(Number(e.target.value))} /></label></div>
      <label className="study-option"><input type="checkbox" checked={modelSplit} onChange={e => setModelSplit(e.target.checked)} />按语义分段</label>
      <button disabled={busy || !material || ['ready', 'processing'].includes(material.status)} onClick={() => void run(async () => {
        await call(`/materials/${selected}/process`, 'POST', { chunk_size: chunkSize, chunk_overlap: overlap, use_model_splitter: modelSplit }); await refresh(); setNotice('资料已可使用。');
      })}>{busy ? '处理中…' : '处理资料'}</button>
      <button className="secondary" disabled={busy} onClick={() => void run(refresh)}>刷新资料与记录</button>
    </section>
    <section className="study-card"><h2>2 · 生成练习</h2>
      <label>知识点<select value={point} onChange={e => setPoint(Number(e.target.value))}><option value={0}>请选择知识点</option>{points.map(p => <option value={p.id} key={p.id}>{p.name}</option>)}</select></label>
      <div className="study-row"><label>题型<select value={type} onChange={e => setType(e.target.value)}>{['single_choice', 'multiple_choice', 'fill_blank', 'short_answer'].map(t => <option key={t} value={t}>{labels[t]}</option>)}</select></label><label>预计难度<select value={difficulty} onChange={e => setDifficulty(Number(e.target.value))}>{[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}</select></label></div>
      <label>出题目标<input value={goal} maxLength={500} onChange={e => setGoal(e.target.value)} placeholder="例如：练习边界条件的应用" /></label>
      <p className="study-meta">出题会参考你的历史摘要。难度为模型估计值。</p>
      <button disabled={busy || material?.status !== 'ready' || !point} onClick={() => void run(async () => {
        const result = await call<{ question: Question }>('/questions/generate', 'POST', { material_id: selected, primary_knowledge_point_id: point, question_type: type, difficulty_level: difficulty, goal: goal || undefined }); setGenerated(result.question);
      })}>{busy ? '请稍候…' : '生成题目'}</button>
    </section></div>
    {error && <p role="alert" className="study-error">{error}</p>}{notice && <p role="status" className="study-notice">{notice}</p>}
    {generated && <section className="study-card"><h2>准备好开始了吗</h2><p className="study-question">{generated.question}</p><button disabled={busy} onClick={() => void run(() => start(generated.id))}>开始作答</button><small>开始后计时；提交有效答案时才保存完整原题。</small></section>}
    <section className="study-card"><h2>3 · 作答历史</h2><p className="study-meta">时间按本地时区显示。重做会新增一次记录。</p>
      {!history.length && <p>还没有作答记录，先选择资料生成一道题。</p>}
      {history.map(a => <article className="study-history" key={a.id}><div><strong>{a.question_summary}</strong><p>{new Date(a.started_at).toLocaleString()} · 第 {a.attempt_no} 次 · {labels[a.status]}</p><small>{duration(a.active_duration_ms)} · 提示 {a.hint_count || 0} 次 · 查看答案 {a.answer_view_count || 0} 次{a.score !== null ? ` · 得分 ${Math.round(a.score * 100)}%` : ''}</small></div><div className="study-row"><button className="secondary" disabled={busy} onClick={() => void run(async () => { setDetail(await call<Detail>(`/attempts/${a.id}`)); })}>{a.status === 'in_progress' ? '继续作答' : '查看记录'}</button>{a.status !== 'in_progress' && a.status !== 'expired' && <button disabled={busy} className="secondary" onClick={() => void run(() => start(a.question_id))}>重做</button>}</div></article>)}
      <div className="study-row"><button className="secondary" disabled={busy || !offset} onClick={() => setOffset(Math.max(0, offset - 50))}>上一页</button><button className="secondary" disabled={busy || history.length < 50} onClick={() => setOffset(offset + 50)}>下一页</button></div>
    </section>
  </>;
}

export default function PracticeApp() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  useEffect(() => {
    void call<{ user: User }>('/auth/me').then(r => setUser(r.user)).catch(e => {
      if (!(e instanceof RequestError && e.status === 401)) setError(messageOf(e));
    }).finally(() => setLoading(false));
    const expired = () => setUser(null);
    window.addEventListener('ctrlpass-session-expired', expired);
    return () => window.removeEventListener('ctrlpass-session-expired', expired);
  }, []);
  if (loading) return <p className="study-loading">正在读取账号…</p>;
  return <div className="study-app">
    {error && <p role="alert" className="study-error">{error}</p>}
    {!user ? <Auth onLogin={u => { setUser(u); setError(''); }} /> : <>
      <header className="study-header"><div><p className="study-eyebrow">CtrlPass</p><h1>每一次练习，都有记录</h1></div><div className="study-row"><span>{user.display_name}</span><button className="secondary" onClick={() => void call('/auth/logout', 'POST', {}).then(() => setUser(null)).catch(e => setError(messageOf(e)))}>退出</button></div></header>
      <Workspace key={user.id} />
    </>}
  </div>;
}
