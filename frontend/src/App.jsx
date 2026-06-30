import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

const LANGUAGES = ['Auto', 'English', '한국어', '日本語', 'Español']
const LANG_CODE = { Auto: 'en', English: 'en', '한국어': 'ko', '日本語': 'ja', 'Español': 'es' }
const STORE_KEY = 'apollo_chat_v1'
const SEEN_KEY = 'apollo_seen_v1'

// ── UI strings per language (fallback: en) ─────────────────────────────
const T = {
  en: {
    subtitle: 'Retrieval Terminal', language: 'Language', clear: 'Clear', help: 'Guide',
    you: 'YOU', sys: 'APOLLO',
    emptyTitle: 'Query the Federal Aviation Regulations',
    emptyEx: '“What aeronautical experience is required for a private pilot certificate?”  ·  “VFR fuel reserves — day vs night?”',
    placeholder: 'Type a query — Enter to transmit, Shift+Enter for newline',
    clearConfirm: 'Clear the whole conversation?',
    sStepPlan: 'Planning the search', sStepPlanDone: (n) => `Search plan · ${n} ${n === 1 ? 'query' : 'queries'}`,
    sStepSearch: 'Searching the record', sStepSearchDone: 'Retrieved candidates',
    sStepRerank: 'Re-ranking by relevance', sStepRerankDone: 'Top passages selected',
    sStepGen: 'Composing grounded answer',
    dPassages: 'passages', dUsed: 'used', dChars: 'chars', dTokens: 'tokens',
    dExplain: (k, r, c) => `Searched every indexed passage, kept the ${k} most relevant (${r}); the answer used ${c} of them. Tokens = text processed — fewer is cheaper and faster.`,
    gChunks: 'Passages (used / found)', gCtxSize: 'Context size', gCtxSent: 'Context sent',
    gAns: 'Answer written', gRewrite: 'Query rewrite', gTotal: 'Total tokens', gLang: 'Answer language',
    tRel: 'Relevance', tDoc: 'Document', tSize: 'Size', tUsed: 'Used', tPrev: 'Preview',
    open: 'open source', loading: 'Loading…', toolsUsed: 'Tools used',
    dBlocked: (r) => `🛡 Blocked (${r}) · 0 generation tokens`,
    dUnverified: (s) => `⚠ Cited sections not found in retrieved context: ${s}`,
    tut: [
      ['14 CFR Retrieval Terminal', 'Answers questions about the U.S. Federal Aviation Regulations (Title 14 CFR) from the indexed regulation text only — and shows its work.'],
      ['Transmit a query', 'Type a question and press Enter. Try “What are the VFR fuel-reserve requirements, day versus night?”'],
      ['Answer language', 'The Language control (top-right) sets the answer language and this interface — the sources stay the same.'],
      ['Watch the pipeline', 'While answering you see each step: planning the search, retrieving passages, then composing the grounded answer.'],
      ['Read the telemetry', 'Open “diagnostics” under any answer for tokens, passages found, relevance, and which were cited.'],
      ['Open the sources', 'Click any [n] reference or document to open the regulation at the exact passage. Your session is saved across refreshes.'],
    ],
    skip: 'Skip', back: 'Back', next: 'Next', start: 'Begin',
  },
  ko: {
    subtitle: '검색 터미널', language: '언어', clear: '지우기', help: '안내',
    you: '사용자', sys: '아폴로',
    emptyTitle: '미 연방 항공 규정(14 CFR)에 질문하기',
    emptyEx: '“자가용 조종사 자격에 필요한 비행 경력은?”  ·  “VFR 연료 예비량 — 주간 대 야간?”',
    placeholder: '질문 입력 — Enter 전송, Shift+Enter 줄바꿈',
    clearConfirm: '대화 전체를 지울까요?',
    sStepPlan: '검색 계획 수립', sStepPlanDone: (n) => `검색 계획 · 쿼리 ${n}개`,
    sStepSearch: '기록 검색 중', sStepSearchDone: '후보 구절 검색',
    sStepRerank: '관련도 재정렬 중', sStepRerankDone: '상위 구절 선정',
    sStepGen: '근거 기반 답변 작성',
    dPassages: '구절', dUsed: '사용', dChars: '문자', dTokens: '토큰',
    dExplain: (k, r, c) => `색인된 모든 구절을 검색해 가장 관련 높은 ${k}개를 골랐고(${r}), 답변엔 그중 ${c}개를 사용했습니다. 토큰 = 처리된 텍스트량 — 적을수록 싸고 빠름.`,
    gChunks: '구절 (사용 / 검색)', gCtxSize: '컨텍스트 크기', gCtxSent: '보낸 컨텍스트',
    gAns: '작성된 답변', gRewrite: '질의 재작성', gTotal: '총 토큰', gLang: '답변 언어',
    tRel: '관련도', tDoc: '문서', tSize: '크기', tUsed: '사용', tPrev: '미리보기',
    open: '원문 열기', loading: '불러오는 중…', toolsUsed: '사용한 도구',
    dBlocked: (r) => `🛡 차단됨 (${r}) · 생성 0 토큰`,
    dUnverified: (s) => `⚠ 검색 근거에 없는 인용 섹션: ${s}`,
    tut: [
      ['14 CFR 검색 터미널', '색인된 규정 본문만으로 미 연방 항공 규정(14 CFR)에 답하고, 그 과정을 보여줍니다.'],
      ['질문 전송', '질문을 입력하고 Enter. 예: “VFR 연료 예비량 요건은 주간과 야간에 어떻게 다른가?”'],
      ['답변 언어', '오른쪽 위 언어 설정이 답변 언어와 이 화면 언어를 바꿉니다 — 출처는 그대로.'],
      ['파이프라인 보기', '답변 중 각 단계가 보입니다: 검색 계획 → 구절 검색 → 근거 기반 답변 작성.'],
      ['텔레메트리 읽기', '답변 아래 “diagnostics”에서 토큰·검색된 구절·관련도·인용 여부를 확인하세요.'],
      ['출처 열기', '[n] 참조나 문서를 누르면 해당 구절 위치로 규정 원문이 열립니다. 세션은 새로고침해도 유지됩니다.'],
    ],
    skip: '건너뛰기', back: '이전', next: '다음', start: '시작',
  },
  ja: {
    subtitle: '検索ターミナル', language: '言語', clear: 'クリア', help: 'ガイド',
    you: 'あなた', sys: 'アポロ',
    emptyTitle: '米国連邦航空規則（14 CFR）に質問する',
    emptyEx: '「自家用操縦士資格に必要な飛行経験は？」 ・ 「VFRの燃料予備量、昼と夜の違いは？」',
    placeholder: '質問を入力 — Enterで送信、Shift+Enterで改行',
    clearConfirm: '会話をすべて消去しますか？',
    sStepPlan: '検索を計画中', sStepPlanDone: (n) => `検索計画 · クエリ${n}件`,
    sStepSearch: '記録を検索中', sStepSearchDone: '候補を取得',
    sStepRerank: '関連度で再順位付け', sStepRerankDone: '上位を選定',
    sStepGen: '根拠に基づく回答を作成中',
    dPassages: '箇所', dUsed: '使用', dChars: '文字', dTokens: 'トークン',
    dExplain: (k, r, c) => `全インデックスを検索し関連度の高い${k}件を採用（${r}）、回答にそのうち${c}件を使用。トークン=処理した文章量、少ないほど安く速い。`,
    gChunks: '箇所（使用 / 取得）', gCtxSize: 'コンテキスト量', gCtxSent: '送信コンテキスト',
    gAns: '回答の生成', gRewrite: 'クエリ再構成', gTotal: '合計トークン', gLang: '回答言語',
    tRel: '関連度', tDoc: '文書', tSize: 'サイズ', tUsed: '使用', tPrev: 'プレビュー',
    open: '出典を開く', loading: '読み込み中…', toolsUsed: '使用ツール',
    dBlocked: (r) => `🛡 ブロック (${r}) · 生成0トークン`,
    dUnverified: (s) => `⚠ 取得した文脈にない引用セクション: ${s}`,
    tut: [
      ['14 CFR 検索ターミナル', 'インデックス済みの規則本文のみで米国連邦航空規則（14 CFR）に回答し、その過程を表示します。'],
      ['質問を送信', '質問を入力してEnter。例「VFRの燃料予備量は昼と夜でどう違う？」'],
      ['回答言語', '右上の言語設定が回答とこの画面の言語を変えます — 出典はそのまま。'],
      ['処理を見る', '回答中に各ステップが見えます: 検索計画 → 箇所取得 → 根拠に基づく回答作成。'],
      ['テレメトリを読む', '回答下の「diagnostics」でトークン・取得箇所・関連度・引用を確認。'],
      ['出典を開く', '[n] や文書名を押すと該当箇所で原文が開きます。セッションは再読込でも保持。'],
    ],
    skip: 'スキップ', back: '戻る', next: '次へ', start: '開始',
  },
  es: {
    subtitle: 'Terminal de Búsqueda', language: 'Idioma', clear: 'Borrar', help: 'Guía',
    you: 'TÚ', sys: 'APOLO',
    emptyTitle: 'Consulta el Reglamento Federal de Aviación (14 CFR)',
    emptyEx: '«¿Qué experiencia aeronáutica exige una licencia de piloto privado?»  ·  «Reservas de combustible VFR: ¿día o noche?»',
    placeholder: 'Escribe una consulta — Enter para enviar, Shift+Enter para salto de línea',
    clearConfirm: '¿Borrar toda la conversación?',
    sStepPlan: 'Planificando la búsqueda', sStepPlanDone: (n) => `Plan · ${n} ${n === 1 ? 'consulta' : 'consultas'}`,
    sStepSearch: 'Buscando en el archivo', sStepSearchDone: 'Candidatos recuperados',
    sStepRerank: 'Reordenando por relevancia', sStepRerankDone: 'Mejores pasajes',
    sStepGen: 'Componiendo respuesta fundamentada',
    dPassages: 'pasajes', dUsed: 'usados', dChars: 'caract.', dTokens: 'tokens',
    dExplain: (k, r, c) => `Se buscó en todos los pasajes indexados y se conservaron los ${k} más relevantes (${r}); la respuesta usó ${c}. Tokens = texto procesado — menos es más barato y rápido.`,
    gChunks: 'Pasajes (usados / hallados)', gCtxSize: 'Tamaño de contexto', gCtxSent: 'Contexto enviado',
    gAns: 'Respuesta escrita', gRewrite: 'Reescritura', gTotal: 'Tokens totales', gLang: 'Idioma de respuesta',
    tRel: 'Relevancia', tDoc: 'Documento', tSize: 'Tamaño', tUsed: 'Usado', tPrev: 'Vista',
    open: 'abrir fuente', loading: 'Cargando…', toolsUsed: 'Herramientas usadas',
    dBlocked: (r) => `🛡 Bloqueado (${r}) · 0 tokens de generación`,
    dUnverified: (s) => `⚠ Secciones citadas no halladas en el contexto: ${s}`,
    tut: [
      ['Terminal de Búsqueda 14 CFR', 'Responde sobre el Reglamento Federal de Aviación de EE. UU. (Título 14 CFR) solo con el texto indexado, y muestra su trabajo.'],
      ['Envía una consulta', 'Escribe y pulsa Enter. Prueba «¿Reservas de combustible VFR de día frente a noche?»'],
      ['Idioma de respuesta', 'El control de Idioma (arriba a la derecha) fija el idioma de respuesta y de esta interfaz; las fuentes no cambian.'],
      ['Observa el proceso', 'Durante la respuesta ves cada paso: planificar, recuperar pasajes y componer la respuesta.'],
      ['Lee la telemetría', 'Abre «diagnostics» bajo cada respuesta: tokens, pasajes, relevancia y citas.'],
      ['Abre las fuentes', 'Pulsa una referencia [n] o un documento para abrir la fuente en el pasaje exacto. La sesión se guarda.'],
    ],
    skip: 'Omitir', back: 'Atrás', next: 'Siguiente', start: 'Empezar',
  },
}

function loadMessages() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || [] } catch { return [] }
}

export default function App() {
  const [messages, setMessages] = useState(loadMessages)
  const [input, setInput] = useState('')
  const [language, setLanguage] = useState('Auto')
  const [live, setLive] = useState(null)
  const [doc, setDoc] = useState(null)
  const [tutorial, setTutorial] = useState(!localStorage.getItem(SEEN_KEY))
  const [theme, setTheme] = useState(() => localStorage.getItem('apollo_theme') || 'dark')
  const endRef = useRef(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('apollo_theme', theme)
  }, [theme])
  const firstScroll = useRef(true)
  const taRef = useRef(null)
  const docCache = useRef(new Map())   // source -> full text, prefetched for instant open
  const t = T[LANG_CODE[language]] || T.en

  function autoGrow() {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
  }

  useEffect(() => { localStorage.setItem(STORE_KEY, JSON.stringify(messages)) }, [messages])
  useEffect(() => {
    // First paint (e.g. after refresh) lands at the bottom instantly — no animated scroll.
    endRef.current?.scrollIntoView({ behavior: firstScroll.current ? 'auto' : 'smooth' })
    firstScroll.current = false
  }, [messages, live])

  async function send(e) {
    e?.preventDefault()
    const question = input.trim()
    if (!question || live) return
    setMessages((m) => [...m, { role: 'user', text: question, ts: Date.now() }])
    setInput('')
    if (taRef.current) taRef.current.style.height = 'auto'
    setLive({ steps: [], text: '' })

    const t0 = performance.now()
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: question, language,
          history: messages.slice(-4).map((m) => ({ role: m.role, text: m.text })),
        }),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        const msg = detail.error || `request failed (${res.status})`
        setMessages((m) => [...m, { role: 'assistant', text: `⚠️ ${msg}`, ts: Date.now() }])
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '', citations = [], diagnostics = null, answer = '', steps = []
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n'); buffer = parts.pop()
        for (const part of parts) {
          const line = part.replace(/^data: /, '').trim()
          if (!line) continue
          const evt = JSON.parse(line)
          if (evt.type === 'tool') { steps = mergeStep(steps, evt); setLive((s) => ({ ...s, steps })) }
          else if (evt.type === 'delta') { answer += evt.text; setLive((s) => ({ ...s, text: answer })) }
          else if (evt.type === 'done') { citations = evt.citations || []; diagnostics = evt.diagnostics || null }
        }
      }
      steps = steps.map((s) => ({ ...s, status: 'done' }))
      setMessages((m) => [...m, {
        role: 'assistant', text: answer, citations, diagnostics, steps,
        ts: Date.now(), latencyMs: Math.round(performance.now() - t0),
      }])
      // Warm the browser cache so opening a source is instant.
      prefetchDocs([...citations.map((c) => c.source),
                    ...(diagnostics?.chunks || []).map((c) => c.source)])
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', text: `// error: ${err.message}`, ts: Date.now() }])
    } finally { setLive(null) }
  }

  function onKeyDown(e) {
    // Ignore Enter while an IME is mid-composition (Korean/Japanese/Chinese),
    // otherwise the still-composing last character leaks into the next message.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  async function fetchDoc(source) {
    if (docCache.current.has(source)) return docCache.current.get(source)
    const res = await fetch(`/api/doc/${encodeURIComponent(source)}`)
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}))
      throw new Error(detail.error || `couldn't open source (${res.status})`)
    }
    const data = await res.json()
    const text = data.text || data.error || 'not found'
    docCache.current.set(source, text)
    return text
  }

  function prefetchDocs(sources) {
    for (const s of new Set(sources)) {
      if (s && !docCache.current.has(s)) fetchDoc(s).catch(() => {})
    }
  }

  async function openDoc(source, preview) {
    if (docCache.current.has(source)) { setDoc({ source, text: docCache.current.get(source), preview }); return }
    setDoc({ source, text: null, preview })   // shows Loading… only on a cold open
    try { setDoc({ source, text: await fetchDoc(source), preview }) }
    catch (err) { setDoc({ source, text: `error: ${err.message}`, preview }) }
  }

  function clearChat() { if (confirm(t.clearConfirm)) setMessages([]) }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <svg className="logo" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="3.4" fill="currentColor" stroke="none" />
            <ellipse cx="12" cy="12" rx="10" ry="4.3" transform="rotate(-28 12 12)" />
            <circle cx="3.6" cy="8.4" r="1.5" fill="currentColor" stroke="none" />
          </svg>
14&nbsp;CFR<span className="sub">{t.subtitle}</span>
        </div>
        <div className="tools">
          <label className="lang">{t.language}
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              {LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </label>
          <button className="ghost icon" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title="Toggle theme" aria-label="Toggle theme">
            <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true">
              <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.3" />
              <path d="M8 1.5 A6.5 6.5 0 0 0 8 14.5 Z" fill="currentColor" />
            </svg>
          </button>
          <button className="ghost" onClick={clearChat}>{t.clear}</button>
          <button className="ghost" onClick={() => setTutorial(true)}>{t.help}</button>
        </div>
      </header>

      <main className="chat">
        {messages.length === 0 && !live && (
          <div className="empty">
            <div className="empty-mark" />
            <h2>{t.emptyTitle}</h2>
            <p>{t.emptyEx}</p>
          </div>
        )}
        {messages.map((m, i) => <Message key={i} m={m} t={t} onOpenDoc={openDoc} />)}
        {live && <LiveMessage live={live} t={t} />}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={send}>
        <textarea ref={taRef} rows={1} value={input}
          onChange={(e) => { setInput(e.target.value); autoGrow() }}
          onKeyDown={onKeyDown} placeholder={t.placeholder} autoFocus />
        <button type="submit" disabled={!!live || !input.trim()}>↵</button>
      </form>

      {doc && <DocModal doc={doc} t={t} onClose={() => setDoc(null)} />}
      {tutorial && <Tutorial t={t} onClose={() => { localStorage.setItem(SEEN_KEY, '1'); setTutorial(false) }} />}
    </div>
  )
}

// The § glyph is kept in retrieval/citations but not shown in the answer prose.
const stripSection = (s) => (s || '').replace(/§\s*/g, 'section ')

function mergeStep(steps, evt) {
  const idx = steps.findIndex((s) => s.name === evt.name)
  const entry = { name: evt.name, status: evt.status, sources: evt.sources }
  if (idx >= 0) { const next = [...steps]; next[idx] = entry; return next }
  return steps.map((s) => ({ ...s, status: s.status || 'done' })).concat(entry)
}

function stepLabel(s, t) {
  if (s.name === 'rewrite') return s.status === 'done' ? t.sStepPlanDone(s.sources?.length || 1) : t.sStepPlan
  if (s.name === 'search') return s.status === 'done' ? t.sStepSearchDone : t.sStepSearch
  if (s.name === 'rerank') return s.status === 'done' ? t.sStepRerankDone : t.sStepRerank
  if (s.name === 'generate') return t.sStepGen
  return s.name
}

function Message({ m, t, onOpenDoc }) {
  return (
    <div className={`row row-${m.role}`}>
      <div className="bubble">
        {m.role === 'assistant'
          ? <div className="text md"><ReactMarkdown>{stripSection(m.text)}</ReactMarkdown></div>
          : <div className="text">{m.text}</div>}

        {m.diagnostics?.blocked && <div className="alert blocked">{t.dBlocked(m.diagnostics.blocked)}</div>}
        {m.diagnostics?.unverified_sections?.length > 0 && (
          <div className="alert warn">{t.dUnverified(m.diagnostics.unverified_sections.map((s) => `§${s}`).join(', '))}</div>
        )}

        {m.citations?.length > 0 && (
          <div className="sources">
            {m.citations.map((c) => {
              const ch = m.diagnostics?.chunks?.find((x) => x.n === c.n)
              return (
                <button key={c.n} className="ref" onClick={() => onOpenDoc(c.source, ch?.preview)}>
                  <span className="ref-n">{c.n}</span>{c.source}
                  {c.section && <span className="ref-sec"> §{c.section}</span>}
                </button>
              )
            })}
          </div>
        )}

        {m.steps?.length > 0 && (
          <details className="toolsrun">
            <summary>{t.toolsUsed} · {m.steps.length}</summary>
            {m.steps.map((s, i) => (
              <div key={i} className="trow">
                <span className="led" /> {stepLabel(s, t)}
                {s.sources && <span className="step-src"> {s.sources.join(' · ')}</span>}
              </div>
            ))}
          </details>
        )}

        {m.diagnostics && <Diagnostics d={m.diagnostics} latencyMs={m.latencyMs} t={t} onOpenDoc={onOpenDoc} />}
        {m.role === 'assistant' && m.ts && <div className="stamp">{new Date(m.ts).toLocaleString()}</div>}
      </div>
    </div>
  )
}

function LiveMessage({ live, t }) {
  return (
    <div className="row row-assistant">
      <div className="bubble">
        <div className="steps">
          {live.steps.map((s, i) => (
            <div key={i} className={`step ${s.status === 'done' ? 'done' : 'active'}`}>
              <span className="led" /> {stepLabel(s, t)}
              {s.name === 'rewrite' && s.sources && <span className="step-src"> {s.sources.join(' · ')}</span>}
              {s.name === 'search' && s.sources && <span className="step-src"> {s.sources.join(', ')}</span>}
            </div>
          ))}
        </div>
        {live.text
          ? <div className="text md"><ReactMarkdown>{stripSection(live.text)}</ReactMarkdown><span className="caret">▍</span></div>
          : <span className="dots"><i/><i/><i/></span>}
      </div>
    </div>
  )
}

function Bar({ value }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)))
  return <span className="bar"><span className="bar-fill" style={{ width: `${pct}%` }} /><span className="bar-num">{pct}%</span></span>
}

function Diagnostics({ d, latencyMs, t, onOpenDoc }) {
  const tk = d.tokens || {}
  return (
    <details className="diag">
      <summary>
        <span className="pill"><b>{d.k}</b> {t.dPassages}</span>
        <span className="pill"><b>{d.cited_count}</b> {t.dUsed}</span>
        <span className="pill"><b>{d.context_chars?.toLocaleString()}</b> {t.dChars}</span>
        <span className="pill tok-in" title="input tokens"><b>{tk.input?.toLocaleString()}</b> in</span>
        <span className="pill tok-out" title="output tokens"><b>{tk.output?.toLocaleString()}</b> out</span>
        {latencyMs != null && <span className="pill"><b>{(latencyMs / 1000).toFixed(1)}</b>s</span>}
      </summary>

      {d.retrieval && <p className="diag-explain">{t.dExplain(d.k, d.retrieval, d.cited_count)}</p>}

      <div className="diag-tokens">
        <div><span className="k">{t.gChunks}</span><span className="v">{d.cited_count} / {d.k}</span></div>
        <div><span className="k">{t.gCtxSize}</span><span className="v">{d.context_chars?.toLocaleString()}</span></div>
        <div><span className="k">{t.gCtxSent}</span><span className="v">{tk.input?.toLocaleString()}</span></div>
        <div><span className="k">{t.gAns}</span><span className="v">{tk.output?.toLocaleString()}</span></div>
        {tk.rewrite > 0 && <div><span className="k">{t.gRewrite}</span><span className="v">{tk.rewrite?.toLocaleString()}</span></div>}
        <div><span className="k">{t.gTotal}</span><span className="v">{tk.total?.toLocaleString()}</span></div>
        <div><span className="k">{t.gLang}</span><span className="v">{d.language}</span></div>
      </div>

      <table className="diag-table">
        <thead><tr><th>#</th><th>{t.tRel}</th><th>{t.tDoc}</th><th>{t.tSize}</th><th>{t.tUsed}</th><th>{t.tPrev}</th></tr></thead>
        <tbody>
          {d.chunks.map((c) => (
            <tr key={c.n} className={c.cited ? 'used' : ''}>
              <td>{c.n}</td>
              <td><Bar value={c.score} /></td>
              <td><button className="doc-link" onClick={() => onOpenDoc(c.source, c.preview)}>{c.source}<span className="ci">#{c.chunk_index}</span></button></td>
              <td>{c.chars}</td>
              <td>{c.cited ? <span className="yes">✓</span> : <span className="no">·</span>}</td>
              <td className="preview">{c.preview}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  )
}

// Locate the cited chunk inside the document. The chunk prefixes "§ Title\n" and
// the cleaned doc joins newlines to spaces, so match whitespace-flexibly and fall
// back to shorter fragments.
function findHighlight(text, preview) {
  const raw = (preview || '').replace(/…$/, '').trim()
  if (!raw) return null
  for (const n of [140, 90, 50]) {
    const frag = raw.slice(0, n).trim()
    if (frag.length < 12) break
    const pat = frag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+')
    const m = text.match(new RegExp(pat))
    if (m) return [m.index, m.index + m[0].length]
  }
  return null
}

function DocModal({ doc, t, onClose }) {
  const markRef = useRef(null)
  useEffect(() => {
    if (doc.text == null) return
    // Wait for the (large) <pre> to lay out, then jump to the highlighted chunk.
    const id = requestAnimationFrame(() =>
      requestAnimationFrame(() =>
        markRef.current?.scrollIntoView({ block: 'center', behavior: 'auto' })
      )
    )
    return () => cancelAnimationFrame(id)
  }, [doc.text])
  let body
  if (doc.text == null) body = <p className="loading-doc">{t.loading}</p>
  else {
    const span = findHighlight(doc.text, doc.preview)
    body = span
      ? <pre>{doc.text.slice(0, span[0])}<mark ref={markRef}>{doc.text.slice(span[0], span[1])}</mark>{doc.text.slice(span[1])}</pre>
      : <pre>{doc.text}</pre>
  }
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head"><b>{doc.source}</b><button className="ghost" onClick={onClose}>✕</button></div>
        <div className="modal-body">{body}</div>
      </div>
    </div>
  )
}

function Tutorial({ t, onClose }) {
  const [i, setI] = useState(0)
  const steps = t.tut
  const last = i === steps.length - 1
  return (
    <div className="modal-bg">
      <div className="tut">
        <div className="tut-progress">{String(i + 1).padStart(2, '0')} / {String(steps.length).padStart(2, '0')}</div>
        <h3>{steps[i][0]}</h3>
        <p>{steps[i][1]}</p>
        <div className="tut-actions">
          <button className="ghost" onClick={onClose}>{t.skip}</button>
          <div className="grow" />
          {i > 0 && <button className="ghost" onClick={() => setI(i - 1)}>{t.back}</button>}
          <button className="primary" onClick={() => last ? onClose() : setI(i + 1)}>{last ? t.start : t.next}</button>
        </div>
      </div>
    </div>
  )
}
