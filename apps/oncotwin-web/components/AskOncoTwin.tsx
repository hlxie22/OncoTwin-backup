'use client';

import { FormEvent, useEffect, useMemo, useState } from 'react';
import { api } from '@/lib/api';
import type { AssistantAnswer, AssistantBootstrap, AssistantGrounding, AssistantTurn } from '@/lib/types';
import { SourceViewer } from '@/components/SourceViewer';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  grounding?: AssistantGrounding[];
  followups?: string[];
  matched?: boolean;
};

function Grounding({ rows }: { rows?: AssistantGrounding[] }) {
  if (!rows?.length) return null;
  return <div className="ask-grounding">
    <span>Sources</span>
    <div className="ask-source-row">
      {rows.map((s, i) => {
        if (s.kind === 'record' && s.document_id && s.page_number) {
          return <SourceViewer key={s.key || i} source={{
            document_id: s.document_id,
            document_name: s.document_name || 'Source record',
            page_number: s.page_number,
            source_snippet: s.source_snippet || '',
          }} label={s.document_name ? `${s.document_name} · p.${s.page_number}` : 'Check source'} />;
        }
        if (s.url) return <a key={s.key || i} href={s.url} target="_blank" rel="noreferrer" className="source-link">{s.label || 'Open source'} ↗</a>;
        return null;
      })}
    </div>
  </div>;
}

export function AskOncoTwin({ patientId }: { patientId: string }) {
  const [boot, setBoot] = useState<AssistantBootstrap | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const value = await api<AssistantBootstrap>(`/patients/${patientId}/assistant`);
        if (live) setBoot(value);
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { live = false; };
  }, [patientId]);

  const activeSuggestions = useMemo(() => {
    const last = [...messages].reverse().find((x) => x.role === 'assistant');
    if (last?.followups?.length) return last.followups;
    return boot?.suggested_questions || [];
  }, [messages, boot]);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    const prior = messages.slice(-8);
    const nextUser: ChatMessage = { role: 'user', content: q };
    setMessages((x) => [...x, nextUser]);
    setInput('');
    setBusy(true);
    setError('');
    try {
      const history: AssistantTurn[] = prior.map((x) => ({ role: x.role, content: x.content }));
      const value = await api<AssistantAnswer>(`/patients/${patientId}/assistant/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, history }),
      });
      setBoot((b) => b ? { ...b, stage: value.stage, mode: value.mode, suggested_questions: value.suggested_questions } : b);
      setMessages((x) => [...x, {
        role: 'assistant',
        content: value.answer,
        grounding: value.grounding,
        followups: value.suggested_followups,
        matched: value.matched_question,
      }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    void ask(input);
  }

  return <section className="ask-shell" aria-labelledby="ask-oncotwin-title">
    <div className="ask-head">
      <div>
        <div className="eyebrow">ASK ONCOTWIN</div>
        <h2 id="ask-oncotwin-title">Make sense of your cancer journey</h2>
        <p>{boot?.greeting || 'Ask about your history, trajectory, what changed, or what to discuss with your care team.'}</p>
      </div>
      {boot ? <span className="ask-mode-pill">Source-grounded</span> : null}
    </div>

    {!messages.length ? <div className="ask-starter">
      <strong>Try asking</strong>
      <div className="ask-chip-row">
        {(boot?.suggested_questions || []).map((q) => <button key={q} type="button" className="ask-chip" onClick={() => void ask(q)} disabled={busy}>{q}</button>)}
      </div>
    </div> : null}

    {messages.length ? <div className="ask-thread" aria-live="polite">
      {messages.map((m, i) => <div key={`${m.role}-${i}`} className={`ask-message ${m.role}`}>
        <div className="ask-message-label">{m.role === 'user' ? 'You' : 'OncoTwin'}</div>
        <div className="ask-bubble"><p>{m.content}</p><Grounding rows={m.grounding} /></div>
      </div>)}
      {busy ? <div className="ask-message assistant"><div className="ask-message-label">OncoTwin</div><div className="ask-bubble"><p>Thinking about your verified record…</p></div></div> : null}
    </div> : null}

    {error ? <div className="error-card ask-error" role="alert"><strong>OncoTwin could not answer that question.</strong><p>{error}</p></div> : null}

    {messages.length && activeSuggestions.length ? <div className="ask-followups">
      <span>Continue the conversation</span>
      <div className="ask-chip-row">
        {activeSuggestions.map((q) => <button key={q} type="button" className="ask-chip" onClick={() => void ask(q)} disabled={busy}>{q}</button>)}
      </div>
    </div> : null}

    <form className="ask-compose" onSubmit={submit}>
      <label htmlFor={`ask-${patientId}`}>Ask a question</label>
      <div className="ask-input-row">
        <textarea id={`ask-${patientId}`} value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about your history, newest scan, trajectory, or what a result means…" rows={2} disabled={busy} onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void ask(input); }
        }} />
        <button className="primary-button" disabled={busy || !input.trim()}>{busy ? 'Answering…' : 'Send'}</button>
      </div>
    </form>

    <p className="ask-boundary">OncoTwin explains your verified record and research trajectory. It does not choose treatment or replace your oncology team.</p>
  </section>;
}
