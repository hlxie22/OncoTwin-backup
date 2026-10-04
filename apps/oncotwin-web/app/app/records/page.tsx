'use client';

import Link from 'next/link';
import { FormEvent, useEffect, useState } from 'react';
import { api, ensurePatient } from '@/lib/api';
import type { DocumentRecord, IntelligenceArtifact, Patient } from '@/lib/types';
import { SourceViewer } from '@/components/SourceViewer';

function pretty(value: string | null | undefined) {
  return value ? value.replaceAll('_', ' ').replace(/\b\w/g, (x) => x.toUpperCase()) : 'Classifying';
}

export default function RecordsPage() {
  const [patient, setPatient] = useState<Patient | null>(null);
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explanations, setExplanations] = useState<Record<string, IntelligenceArtifact>>({});
  const [explainBusy, setExplainBusy] = useState<string | null>(null);

  async function refresh(p?: Patient) {
    const pat = p ?? patient;
    if (!pat) return;
    setDocs(await api<DocumentRecord[]>(`/patients/${pat.id}/documents`));
  }

  useEffect(() => {
    (async () => {
      try {
        const p = await ensurePatient();
        setPatient(p);
        await refresh(p);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  async function upload(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!patient) return;
    const input = e.currentTarget.elements.namedItem('file') as HTMLInputElement;
    if (!input.files?.[0]) return;
    setBusy(true); setError(null); setMessage('Reading the report and linking extracted facts to source pages…');
    const form = new FormData(); form.append('file', input.files[0]);
    try {
      const result = await api<DocumentRecord>(`/patients/${patient.id}/documents?process=true`, { method: 'POST', body: form });
      setMessage(result.status === 'needs_review' ? 'Record added. A few important facts need review before they enter your cancer history.' : 'Record added to your cancer history.');
      input.value = '';
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    if (!confirm('Remove this record and facts derived only from it?')) return;
    try {
      await api(`/documents/${id}`, { method: 'DELETE' });
      setExplanations((old) => { const next = { ...old }; delete next[id]; return next; });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function explain(doc: DocumentRecord) {
    setExplainBusy(doc.id); setError(null);
    try {
      let artifact = await api<IntelligenceArtifact | { status: string }>(`/documents/${doc.id}/explanation`);
      if (!('artifact_type' in artifact)) {
        artifact = await api<IntelligenceArtifact>(`/documents/${doc.id}/explanation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh: false }),
        });
      }
      setExplanations((old) => ({ ...old, [doc.id]: artifact as IntelligenceArtifact }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExplainBusy(null);
    }
  }

  return <div className="page">
    <header className="page-head">
      <div><div className="eyebrow">YOUR RECORDS</div><h1>Your records</h1><p>Keep your scans, pathology, treatment records, labs, and genomic reports together in one place.</p></div>
      <Link href="/app/update" className="primary-button">Add newest scan</Link>
    </header>

    {error ? <div className="error-card" role="alert"><strong>Record action failed.</strong><p>{error}</p></div> : null}

    <section className="upload-panel modern-upload">
      <form onSubmit={upload}>
        <div><div className="card-kicker">ADD ANOTHER RECORD</div><strong>Upload a cancer record</strong><p>PDF reports only in this release. New scan reports are best added from the Trajectory page.</p></div>
        <label className="file-button"><input name="file" type="file" accept="application/pdf" disabled={busy} />Choose PDF</label>
        <button className="primary-button" disabled={busy}>{busy ? 'Processing…' : 'Upload & process'}</button>
      </form>
      {message && <div className="inline-message" aria-live="polite">{message}</div>}
    </section>

    <section className="section-block">
      <div className="section-title"><div><div className="eyebrow">SOURCE LIBRARY</div><h2>{docs.length ? `${docs.length} uploaded record${docs.length === 1 ? '' : 's'}` : 'No records yet'}</h2></div></div>
      <div className="record-list">
        {docs.length ? docs.map((d) => <div key={d.id} className="record-with-explanation">
          <article className="record-row">
            <div className="record-icon">PDF</div>
            <div className="record-main"><strong>{d.filename}</strong><span>{pretty(d.document_type)} · added {new Date(d.created_at).toLocaleString()}</span>{d.processing_error && <em>{d.processing_error}</em>}</div>
            <span className={`status ${d.status}`}>{pretty(d.status)}</span>
            <button className="text-button" disabled={explainBusy === d.id || d.status === 'failed'} onClick={() => explain(d)}>{explainBusy === d.id ? 'Explaining…' : 'Explain this report'}</button>
            <button className="text-button danger" onClick={() => remove(d.id)}>Remove</button>
          </article>
          {explanations[d.id] ? <article className="cp4-explanation-card" aria-live="polite">
            <div className="card-kicker">PLAIN-LANGUAGE REPORT EXPLANATION</div>
            <h3>{String(explanations[d.id].content?.summary || 'Report explanation')}</h3>
            {Array.isArray(explanations[d.id].content?.key_points) && explanations[d.id].content.key_points.length > 1 ? <ul>{explanations[d.id].content.key_points.map((x: string) => <li key={x}>{x}</li>)}</ul> : null}
            <div className="cp4-source-row">{(explanations[d.id].grounding || []).map((s) => s.document_id && s.page_number ? <SourceViewer key={s.key} source={{ document_id: s.document_id, document_name: s.document_name || d.filename, page_number: s.page_number, source_snippet: s.source_snippet || '' }} label="Check source" /> : null)}</div>
          </article> : null}
        </div>) : <div className="empty-state"><strong>Your record library is empty.</strong><p>Upload your first PDF to start building your cancer history.</p></div>}
      </div>
    </section>
  </div>;
}
