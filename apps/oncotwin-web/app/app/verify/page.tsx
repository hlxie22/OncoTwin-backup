'use client';

import { useEffect, useState } from 'react';
import { api, ensurePatient } from '@/lib/api';
import type { CandidateFact, Patient } from '@/lib/types';
import { FactReviewCard } from '@/components/FactReviewCard';

export default function VerifyPage() {
  const [patient, setPatient] = useState<Patient | null>(null);
  const [facts, setFacts] = useState<CandidateFact[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function refresh(p?: Patient) {
    const pat = p ?? patient;
    if (pat) setFacts(await api<CandidateFact[]>(`/patients/${pat.id}/review`));
  }

  useEffect(() => {
    (async () => {
      try {
        const p = await ensurePatient(); setPatient(p); await refresh(p);
      } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    })();
  }, []);

  async function decide(id: string, decision: 'accept' | 'reject' | 'edit', value?: string) {
    setBusy(true); setError('');
    try {
      await api(`/candidates/${id}/decision`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision, edited_value: value ?? null }) });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return <div className="page">
    <header className="page-head"><div><div className="eyebrow">REVIEW</div><h1>Check anything that needs a second look</h1><p>Some important details need your confirmation before OncoTwin adds them to your cancer history.</p></div>{facts ? <span className="counter">{facts.length} to review</span> : null}</header>
    {error ? <div className="error-card" role="alert"><strong>Review action failed.</strong><p>{error}</p></div> : null}
    {facts === null ? <div className="skeleton big" aria-label="Loading review items" /> : facts.length ? <div className="verify-list">{facts.map((f) => <FactReviewCard key={f.id} fact={f} onDecision={decide} busy={busy} />)}</div> : <div className="success-card large"><strong>Nothing needs your review right now.</strong><p>If OncoTwin finds something uncertain or conflicting, it will appear here for you to check.</p></div>}
  </div>;
}
