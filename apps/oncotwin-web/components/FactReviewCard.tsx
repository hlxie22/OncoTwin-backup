'use client';

import { useState } from 'react';
import type { CandidateFact } from '@/lib/types';
import { SourceViewer } from './SourceViewer';

export type ReviewDecision = (id: string, decision: 'accept' | 'reject' | 'edit', value?: string) => Promise<void>;

export function FactReviewCard({ fact, onDecision, busy = false }: { fact: CandidateFact; onDecision: ReviewDecision; busy?: boolean }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(fact.value);

  async function saveEdit() {
    const next = value.trim();
    if (!next) return;
    await onDecision(fact.id, 'edit', next);
    setEditing(false);
  }

  return (
    <article className="verify-card">
      <div className="verify-top">
        <div>
          <span className="card-kicker">{fact.fact_type.replaceAll('_', ' ')}</span>
          {editing ? (
            <label className="edit-field">
              <span>Correct value</span>
              <input value={value} onChange={(e) => setValue(e.target.value)} autoFocus />
            </label>
          ) : <h2>{fact.value}</h2>}
        </div>
        <span className="confidence">{Math.round(fact.confidence * 100)}% extraction confidence</span>
      </div>
      <blockquote className="snippet">“{fact.source_snippet}”</blockquote>
      <p className="warning-text">{fact.review_reason ?? 'This important field needs your confirmation before it enters the patient record.'}</p>
      <div className="verify-actions">
        <SourceViewer source={{ document_id: fact.document_id, document_name: fact.document_name, page_number: fact.source_page, source_snippet: fact.source_snippet }} label="Check report" />
        {editing ? (
          <>
            <button className="secondary-button" disabled={busy} onClick={() => { setValue(fact.value); setEditing(false); }}>Cancel</button>
            <button className="primary-button" disabled={busy || !value.trim()} onClick={saveEdit}>Save correction</button>
          </>
        ) : (
          <>
            <button className="secondary-button" disabled={busy} onClick={() => setEditing(true)}>Edit</button>
            <button className="secondary-button danger" disabled={busy} onClick={() => onDecision(fact.id, 'reject')}>Not correct</button>
            <button className="primary-button" disabled={busy} onClick={() => onDecision(fact.id, 'accept')}>Confirm</button>
          </>
        )}
      </div>
    </article>
  );
}
