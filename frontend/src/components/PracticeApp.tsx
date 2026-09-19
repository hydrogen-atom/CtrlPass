import { useCallback, useEffect, useRef, useState } from 'react';
import { call, labels, messageOf, RequestError, requestKey } from '../api/practice';
import type { AgentOutput, AgentResult, Attempt, Detail, ErrorRecord, Hint, KnowledgeGraph, Material, Progress, User } from '../api/practice';
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

function KnowledgeGraphView({ graph }: { graph: KnowledgeGraph }) {
  const width = 760; const height = 430; const radius = Math.min(165, 28 + graph.nodes.length * 8);
  const positions = new Map(graph.nodes.map((node, index) => {
    const angle = (Math.PI * 2 * index / graph.nodes.length) - Math.PI / 2;
    return [node.id, { x: width / 2 + Math.cos(angle) * radius, y: height / 2 + Math.sin(angle) * radius }];
  }));
  return <div className="study-graph">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`知识图谱：${graph.nodes.length} 个节点，${graph.edges.length} 条关系`}>
      <defs><marker id="study-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>
      {graph.edges.map((edge, index) => { const from = positions.get(edge.source); const to = positions.get(edge.target); return from && to ? <g key={`${edge.source}-${edge.target}-${index}`}><line x1={from.x} y1={from.y} x2={to.x} y2={to.y} markerEnd="url(#study-arrow)" /><text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 5}>{edge.relation.slice(0, 12)}</text></g> : null; })}
      {graph.nodes.map(node => { const position = positions.get(node.id)!; return <g className="study-graph-node" key={node.id} transform={`translate(${position.x} ${position.y})`}><circle r="37" /><text textAnchor="middle" y="4">{node.label.length > 10 ? `${node.label.slice(0, 9)}…` : node.label}</text><title>{node.label}：{node.description}</title></g>; })}
    </svg>
    <div className="study-graph-relations">{graph.edges.map((edge, index) => <p key={`${edge.source}-${edge.target}-${index}`}><strong>{graph.nodes.find(node => node.id === edge.source)?.label}</strong> —{edge.relation}→ <strong>{graph.nodes.find(node => node.id === edge.target)?.label}</strong>{edge.evidence && <small>{edge.evidence}</small>}</p>)}</div>
  </div>;
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

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text?: string;
  output?: AgentOutput;
  agent?: AgentResult;
}

const taskOptions = [
  { value: 'auto', label: '自动', icon: '✦' },
  { value: 'qa', label: '问答', icon: '问' },
  { value: 'knowledge_graph', label: '图谱', icon: '图' },
  { value: 'practice', label: '练习', icon: '练' },
];

function Workspace({ user, onLogout }: { user: User; onLogout: () => void }) {
  const [materials, setMaterials] = useState<Material[]>([]);
  const [history, setHistory] = useState<Attempt[]>([]);
  const [selected, setSelected] = useState(0);
  const [prompt, setPrompt] = useState('');
  const [detail, setDetail] = useState<Detail | null>(null);
  const [agentTask, setAgentTask] = useState('auto');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const conversationEnd = useRef<HTMLDivElement>(null);
  const refresh = useCallback(async () => {
    const [m, a] = await loadWorkspace(0);
    setMaterials(m.materials); setHistory(a.attempts);
    setSelected(current => current || m.materials.find(item => item.status === 'ready')?.id || m.materials[0]?.id || 0);
  }, []);
  useEffect(() => {
    let cancelled = false;
    void loadWorkspace(0).then(([m, a]) => {
      if (!cancelled) {
        setMaterials(m.materials); setHistory(a.attempts);
        setSelected(m.materials.find(item => item.status === 'ready')?.id || m.materials[0]?.id || 0);
      }
    }).catch(e => { if (!cancelled) setError(messageOf(e)); });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => { conversationEnd.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, busy]);
  const material = materials.find(m => m.id === selected);
  const start = async (qid: number) => {
    setBusy(true); setError('');
    try {
    const result = await call<{ attempt: Attempt }>(`/questions/${qid}/attempts`, 'POST', { is_page_hidden: Number(document.hidden) });
      setDetail(await call<Detail>(`/attempts/${result.attempt.id}`)); await refresh();
    } catch (e) { setError(messageOf(e)); } finally { setBusy(false); }
  };

  const sendMessage = async (task = agentTask, text = prompt) => {
    if (!material || material.status !== 'ready' || busy) return;
    const fallbacks: Record<string, string> = {
      auto: '请根据这份资料安排最合适的学习任务', qa: '请总结这份资料的核心内容',
      knowledge_graph: '请生成这份资料的核心知识图谱', practice: '请根据我的学习记录安排一道练习题',
    };
    const content = text.trim() || fallbacks[task];
    const userMessage: ChatMessage = { id: `user-${Date.now()}`, role: 'user', text: content };
    setMessages(current => [...current, userMessage]); setPrompt(''); setBusy(true); setError(''); setNotice('');
    try {
      const response = await call<{ result: AgentOutput; agent: AgentResult }>('/agent/run', 'POST', {
        material_id: material.id, task, goal: content,
      });
      setMessages(current => [...current, { id: `assistant-${Date.now()}`, role: 'assistant', output: response.result, agent: response.agent }]);
      await refresh();
    } catch (e) { setError(messageOf(e)); } finally { setBusy(false); }
  };

  const upload = async (file: File) => {
    setBusy(true); setError(''); setNotice('正在上传并处理资料…');
    try {
      const form = new FormData(); form.append('file', file);
      const uploaded = await call<{ material: Material }>('/materials/upload', 'POST', form);
      setSelected(uploaded.material.id); setMessages([]);
      await call(`/materials/${uploaded.material.id}/process`, 'POST', { chunk_size: 500, chunk_overlap: 100, use_model_splitter: false });
      await refresh(); setNotice(`“${uploaded.material.original_filename}”已准备好，可以开始提问。`);
    } catch (e) { setError(messageOf(e)); } finally { setBusy(false); }
  };

  const processSelected = async () => {
    if (!material) return;
    setBusy(true); setError(''); setNotice('正在处理资料…');
    try {
      await call(`/materials/${material.id}/process`, 'POST', { chunk_size: 500, chunk_overlap: 100, use_model_splitter: false });
      await refresh(); setNotice('资料已准备好，可以开始对话。');
    } catch (e) { setError(messageOf(e)); } finally { setBusy(false); }
  };

  if (detail) return <AttemptView key={detail.attempt.id} initial={detail} onChanged={() => { void refresh().catch(e => setError(messageOf(e))); }} onClose={() => setDetail(null)} />;
  return <div className="chat-shell">
    <aside className="chat-sidebar">
      <div className="chat-brand"><span className="chat-brand-mark">C</span><div><strong>CtrlPass</strong><small>资料学习助手</small></div></div>
      <label className={`chat-upload ${busy ? 'disabled' : ''}`}>＋ 上传资料<input type="file" accept=".txt,.pdf,.docx" disabled={busy} onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void upload(file); }} /></label>
      <div className="chat-sidebar-section"><p className="chat-sidebar-title">学习资料</p><div className="chat-materials">
        {!materials.length && <small>上传 TXT、PDF 或 DOCX 开始学习</small>}
        {materials.map(item => <button className={item.id === selected ? 'active' : ''} key={item.id} onClick={() => { setSelected(item.id); setMessages([]); setError(''); setNotice(''); }}><span>{item.original_filename}</span><small><i className={`status-dot ${item.status}`} />{labels[item.status]}</small></button>)}
      </div></div>
      <div className="chat-sidebar-section chat-history-list"><p className="chat-sidebar-title">最近练习</p>
        {!history.length && <small>暂无练习记录</small>}
        {history.slice(0, 8).map(attempt => <button key={attempt.id} onClick={() => void call<Detail>(`/attempts/${attempt.id}`).then(setDetail).catch(e => setError(messageOf(e)))}><span>{attempt.question_summary}</span><small>{labels[attempt.status]}{attempt.score !== null ? ` · ${Math.round(attempt.score * 100)}分` : ''}</small></button>)}
      </div>
      <div className="chat-account"><div className="chat-avatar">{user.display_name.slice(0, 1).toUpperCase()}</div><span>{user.display_name}</span><button onClick={onLogout} aria-label="退出登录">退出</button></div>
    </aside>
    <main className="chat-main">
      <header className="chat-topbar"><div><strong>{material?.original_filename || '请选择学习资料'}</strong>{material && <small>{material.chunk_count} 个片段 · {labels[material.status]}</small>}</div><div className="chat-mobile-actions"><select aria-label="选择资料" value={selected} onChange={event => { setSelected(Number(event.target.value)); setMessages([]); }}><option value={0}>选择资料</option>{materials.map(item => <option key={item.id} value={item.id}>{item.original_filename}</option>)}</select><label aria-label="上传资料">＋<input type="file" accept=".txt,.pdf,.docx" disabled={busy} onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void upload(file); }} /></label><select aria-label="最近练习" value="" onChange={event => { const attemptId = Number(event.target.value); if (attemptId) void call<Detail>(`/attempts/${attemptId}`).then(setDetail).catch(e => setError(messageOf(e))); }}><option value="">历史</option>{history.slice(0, 8).map(attempt => <option key={attempt.id} value={attempt.id}>{attempt.question_summary}</option>)}</select><button onClick={onLogout}>退出</button></div></header>
      <div className="chat-conversation">
        {!material && <div className="chat-empty"><div className="chat-empty-mark">C</div><h1>从一份资料开始</h1><p>上传文档后，可以直接提问、生成知识图谱或安排练习。</p></div>}
        {material && !messages.length && <div className="chat-empty"><div className="chat-empty-mark">✦</div><h1>想从这份资料学什么？</h1><p>直接输入问题，或者选择下面的常用功能。</p><div className="chat-suggestions"><button onClick={() => void sendMessage('qa', '请总结这份资料的核心内容')}>总结核心内容<span>→</span></button><button onClick={() => void sendMessage('knowledge_graph', '生成整份资料的核心知识图谱')}>生成知识图谱<span>→</span></button><button onClick={() => void sendMessage('practice', '根据我的学习记录安排一道练习题')}>安排一次练习<span>→</span></button></div></div>}
        {messages.map(message => message.role === 'user' ? <article className="chat-message user" key={message.id}><div className="chat-bubble">{message.text}</div></article> : <article className="chat-message assistant" key={message.id}><div className="chat-assistant-mark">C</div><div className="chat-response">
          {message.agent && <p className="chat-result-title">{message.agent.message}</p>}
          {message.text && <p>{message.text}</p>}
          {message.output?.type === 'qa' && <><div className="study-answer">{message.output.answer}</div><div className="chat-sources">{message.output.source_refs.map(ref => <span key={`${ref.chunk_id}-${ref.page}`}>片段 {ref.chunk_id}{ref.page === null ? '' : ` · 第 ${ref.page} 页`}</span>)}</div></>}
          {message.output?.type === 'knowledge_graph' && <><KnowledgeGraphView graph={message.output.graph} /><div className="chat-sources">{message.output.graph.source_refs.map(ref => <span key={`${ref.chunk_id}-${ref.page}`}>片段 {ref.chunk_id}{ref.page === null ? '' : ` · 第 ${ref.page} 页`}</span>)}</div></>}
          {message.output?.type === 'practice' && <div className="chat-question-card"><span>{labels[message.output.question.question_type]} · 难度 {message.output.question.difficulty_level}/5</span><h3>{message.output.question.question}</h3><button disabled={busy} onClick={() => { if (message.output?.type === 'practice') void start(message.output.question.id); }}>开始作答</button></div>}
          {message.agent && <details className="chat-trace"><summary>查看 Agent 执行过程 · {message.agent.tool_steps} 次工具调用</summary><ol>{message.agent.trace.map((event, index) => <li key={`${event.phase}-${event.step}-${index}`}>{event.summary}</li>)}</ol></details>}
        </div></article>)}
        {busy && <article className="chat-message assistant"><div className="chat-assistant-mark">C</div><div className="chat-thinking"><span /><span /><span /></div></article>}
        <div ref={conversationEnd} />
      </div>
      <div className="chat-bottom">
        {notice && <p role="status" className="chat-inline-notice">{notice}</p>}{error && <p role="alert" className="chat-inline-error">{error}</p>}
        {material && material.status !== 'ready' && <button className="chat-process" disabled={busy || material.status === 'processing'} onClick={() => void processSelected()}>{material.status === 'processing' ? '资料处理中…' : '处理当前资料'}</button>}
        <div className="chat-mode-tabs">{taskOptions.map(option => <button key={option.value} className={agentTask === option.value ? 'active' : ''} onClick={() => setAgentTask(option.value)}><span>{option.icon}</span>{option.label}</button>)}</div>
        <div className="chat-composer"><textarea rows={1} value={prompt} maxLength={1000} disabled={busy || !material || material.status !== 'ready'} onChange={event => setPrompt(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void sendMessage(); } }} placeholder={!material ? '请先上传或选择资料' : material.status !== 'ready' ? '请先处理资料' : agentTask === 'qa' ? '针对资料提一个问题…' : agentTask === 'knowledge_graph' ? '描述希望梳理的知识范围…' : agentTask === 'practice' ? '描述想练习的内容、题型或难度…' : '向 CtrlPass 发送消息…'} /><button className="chat-send" aria-label="发送" disabled={busy || !material || material.status !== 'ready'} onClick={() => void sendMessage()}>↑</button></div>
        <small className="chat-disclaimer">回答基于已上传资料生成，请结合引用片段核对。</small>
      </div>
    </main>
  </div>;
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
    {!user ? <Auth onLogin={u => { setUser(u); setError(''); }} /> : <Workspace key={user.id} user={user} onLogout={() => void call('/auth/logout', 'POST', {}).then(() => setUser(null)).catch(e => setError(messageOf(e)))} />}
  </div>;
}
