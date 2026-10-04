'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api, ensurePatient, ensureSession } from '@/lib/api';
import type { EvidenceBundle, IntelligenceArtifact, IntelligenceBundle, Patient } from '@/lib/types';
import { SourceViewer } from '@/components/SourceViewer';

function sourceNode(s: any, i: number) {
  if (s?.kind === 'record' && s.document_id && s.page_number) {
    return <SourceViewer key={s.key || i} source={{ document_id: s.document_id, document_name: s.document_name || 'Source record', page_number: s.page_number, source_snippet: s.source_snippet || '' }} label="Check source" />;
  }
  if (s?.url) return <a key={s.key || i} className="source-link" href={s.url} target="_blank" rel="noreferrer">Open source ↗</a>;
  return null;
}

export default function PreparePage() {
  const [patient, setPatient] = useState<Patient | null>(null);
  const [bundle, setBundle] = useState<IntelligenceBundle | null>(null);
  const [questions, setQuestions] = useState<IntelligenceArtifact | null>(null);
  const [brief, setBrief] = useState<IntelligenceArtifact | null>(null);
  const [evidence, setEvidence] = useState<EvidenceBundle | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  async function refresh(p?: Patient) {
    const pat = p ?? patient;
    if (!pat) return;
    const b = await api<IntelligenceBundle>(`/patients/${pat.id}/intelligence`);
    setBundle(b);
    setQuestions(b.visit_questions || null);
    setBrief(b.visit_brief || null);
    const ev = b.evidence?.content as any;
    if (ev) setEvidence({ artifact_id: b.evidence?.id || undefined, ...ev } as EvidenceBundle);
  }

  useEffect(() => {
    (async () => {
      try {
        const p = await ensurePatient(); setPatient(p); await refresh(p);
      } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    })();
  }, []);

  async function generateQuestions() {
    if (!patient) return; setBusy('questions'); setError('');
    try {
      const value = await api<IntelligenceArtifact>(`/patients/${patient.id}/intelligence/questions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ refresh: false }) });
      setQuestions(value); await refresh(patient);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(''); }
  }

  async function generateBrief() {
    if (!patient) return; setBusy('brief'); setError('');
    try {
      const value = await api<IntelligenceArtifact>(`/patients/${patient.id}/intelligence/visit-brief`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ refresh: false }) });
      setBrief(value); await refresh(patient);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(''); }
  }

  async function loadEvidence(refreshLive = false) {
    if (!patient) return; setBusy('evidence'); setError('');
    try {
      const value = await api<EvidenceBundle>(`/patients/${patient.id}/evidence`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ refresh: refreshLive, max_literature: 4, max_trials: 4 }) });
      setEvidence(value); await refresh(patient);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(''); }
  }

  async function createExport() {
    if (!patient) return; setBusy('export'); setError('');
    try {
      await ensureSession();
      const value = await api<{ artifact_id: string }>(`/patients/${patient.id}/export`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ include_evidence: true, include_forecast: true }) });
      window.open(`/api/oncotwin/exports/${value.artifact_id}/download`, '_blank', 'noopener,noreferrer');
      await refresh(patient);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(''); }
  }

  const questionRows = (questions?.content?.questions || []) as any[];
  const briefContent = brief?.content || {};

  return <div className="page cp4-prepare-page">
    <header className="page-head">
      <div><div className="eyebrow">PREPARE FOR YOUR VISIT</div><h1>Turn your cancer history into a useful conversation</h1><p>Build questions, a concise visit brief, and a small set of source-linked research and trial leads from the information already verified in OncoTwin.</p></div>
      <Link href="/app/forecast" className="secondary-button">Back to trajectory</Link>
    </header>
    {error ? <div className="error-card" role="alert"><strong>This step could not be completed.</strong><p>{error}</p></div> : null}

    <section className="cp4-action-grid">
      <article className="feature-panel cp4-action-card"><div><div className="card-kicker">QUESTIONS</div><h2>Questions for your next visit</h2><p>Prioritized from your verified history, recent changes, uncertainty, and available evidence.</p></div><button className="primary-button" disabled={Boolean(busy)} onClick={generateQuestions}>{busy === 'questions' ? 'Preparing…' : questions ? 'Refresh questions' : 'Prepare questions'}</button></article>
      <article className="feature-panel cp4-action-card"><div><div className="card-kicker">ONE-PAGE BRIEF</div><h2>A concise summary to review or share</h2><p>Current cancer picture, recent changes, unresolved questions, and the research-trajectory context kept separate from the language summary.</p></div><button className="primary-button" disabled={Boolean(busy)} onClick={generateBrief}>{busy === 'brief' ? 'Preparing…' : brief ? 'Refresh brief' : 'Prepare brief'}</button></article>
    </section>

    {questions ? <section className="section-block"><div className="section-title"><div><div className="eyebrow">NEXT VISIT</div><h2>{String(questions.content?.intro || 'Questions to bring with you')}</h2></div></div><div className="cp4-question-list">{questionRows.map((q, i) => <article key={i} className="cp4-question"><span>{i + 1}</span><div><h3>{q.question}</h3><p>{q.why_this_may_be_useful}</p></div></article>)}</div><div className="cp4-source-row">{(questions.grounding || []).map(sourceNode)}</div></section> : null}

    {brief ? <section className="section-block cp4-brief"><div className="section-title"><div><div className="eyebrow">VISIT BRIEF</div><h2>{String(briefContent.headline || 'Your visit brief')}</h2></div><button className="secondary-button" disabled={Boolean(busy)} onClick={createExport}>{busy === 'export' ? 'Creating PDF…' : 'Create shareable PDF'}</button></div>
      <div className="cp4-brief-grid"><div><h3>Your current picture</h3><ul>{(briefContent.current_picture || []).map((x: string) => <li key={x}>{x}</li>)}</ul></div><div><h3>Recent changes</h3><ul>{(briefContent.recent_changes || []).map((x: string) => <li key={x}>{x}</li>)}</ul></div><div><h3>Uncertainty to clarify</h3><ul>{(briefContent.uncertainties || []).map((x: string) => <li key={x}>{x}</li>)}</ul></div></div>
      <p className="muted">{String(briefContent.research_forecast_note || '')}</p><div className="cp4-source-row">{(brief.grounding || []).map(sourceNode)}</div></section> : null}

    <section className="section-block">
      <div className="section-title"><div><div className="eyebrow">EVIDENCE & TRIALS</div><h2>Source-linked research to discuss with your care team</h2><p>OncoTwin searches authoritative sources from verified patient context. It does not invent current literature from model memory.</p></div><button className="secondary-button" disabled={Boolean(busy)} onClick={() => loadEvidence(true)}>{busy === 'evidence' ? 'Searching…' : evidence ? 'Refresh sources' : 'Find evidence & trials'}</button></div>
      {evidence?.note ? <div className="notice-card"><strong>Evidence snapshot</strong><p>{evidence.note}</p></div> : null}
      {evidence ? <div className="cp4-evidence-grid">
        <div><h3>Research</h3>{evidence.literature?.length ? evidence.literature.map((x: any) => <article className="cp4-source-card" key={x.source_id}><span>{x.journal || 'PubMed'} {x.date ? `· ${x.date}` : ''}</span><h4>{x.title}</h4>{x.patient_summary?.plain_language_summary ? <p>{x.patient_summary.plain_language_summary}</p> : <p className="muted">Open the source for details; a grounded summary was not generated.</p>}<a href={x.url} target="_blank" rel="noreferrer">Open PubMed source ↗</a></article>) : <div className="empty-state compact"><strong>No literature sources in this result.</strong></div>}</div>
        <div><h3>Clinical trials</h3>{evidence.trials?.length ? evidence.trials.map((x: any) => <article className="cp4-source-card" key={x.source_id}><span>{x.source_id} · {String(x.status || '').replaceAll('_',' ')}</span><h4>{x.title}</h4>{x.patient_relevance?.relevance_summary ? <p>{x.patient_relevance.relevance_summary}</p> : <p className="muted">Potential relevance has not been summarized. Open the registry record and ask the trial site about eligibility.</p>}<a href={x.url} target="_blank" rel="noreferrer">Open ClinicalTrials.gov ↗</a></article>) : <div className="empty-state compact"><strong>No trial sources in this result.</strong></div>}</div>
      </div> : <div className="empty-state"><strong>No evidence search has been run for this state.</strong><p>Run a search when you want a small, current set of sources tied to the verified information in your record.</p></div>}
      {evidence?.trial_disclaimer ? <p className="muted cp4-disclaimer">{evidence.trial_disclaimer}</p> : null}
    </section>

    {bundle?.what_changed ? <section className="notice-card"><strong>Recent scan update is connected.</strong><p>The visit tools can use the verified changes from your newest scan without changing the quantitative forecast.</p></section> : null}
  </div>;
}
