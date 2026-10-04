'use client';

import { FormEvent, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { api, ensurePatient } from '@/lib/api';
import type { CandidateFact, DocumentRecord, Forecast, IntelligenceArtifact, IntelligenceBundle, Patient, StateResponse } from '@/lib/types';
import { FactReviewCard } from '@/components/FactReviewCard';
import { ForecastChart } from '@/components/ForecastChart';
import { ModelSupport } from '@/components/ModelSupport';
import { SourceViewer } from '@/components/SourceViewer';
import { AskOncoTwin } from '@/components/AskOncoTwin';

const FLOW_KEY = 'oncotwin_cp3_scan_flow_v1';
type Stage = 'baseline' | 'processing' | 'verify' | 'ready' | 'complete';
type SavedFlow = {
  patientId: string;
  stage: Stage;
  beforeState: StateResponse | null;
  document: DocumentRecord | null;
  afterState: StateResponse | null;
  forecast: Forecast | null;
};

function pct(v: number | undefined | null) {
  return typeof v === 'number' && Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—';
}

function delta(before: number | undefined, after: number | undefined) {
  if (typeof before !== 'number' || typeof after !== 'number') return '—';
  const d = Math.round((after - before) * 100);
  return `${d > 0 ? '+' : ''}${d} percentage points`;
}

function pretty(value: string | null | undefined) {
  return value ? value.replaceAll('_', ' ').replace(/\b\w/g, (x) => x.toUpperCase()) : '—';
}

function changes(before: StateResponse | null, after: StateResponse | null) {
  if (!before || !after) return [] as string[];
  const a = before.state;
  const b = after.state;
  const out: string[] = [];
  if (a.latest_scan?.fact_id !== b.latest_scan?.fact_id) out.push(`Latest scan updated to ${pretty(b.latest_scan?.assessment)}${b.latest_scan?.date ? ` (${b.latest_scan.date})` : ''}.`);
  if (a.current_treatment?.fact_id !== b.current_treatment?.fact_id && b.current_treatment) out.push(`Current treatment record updated to ${b.current_treatment.name}.`);
  const oldSites = new Set(a.disease_sites);
  const addedSites = b.disease_sites.filter((x) => !oldSites.has(x));
  if (addedSites.length) out.push(`New disease-site information: ${addedSites.map(pretty).join(', ')}.`);
  const oldGenomics = new Set(a.genomics.map((x) => x.alteration));
  const addedGenomics = b.genomics.map((x) => x.alteration).filter((x) => !oldGenomics.has(x));
  if (addedGenomics.length) out.push(`New genomic information in your record: ${addedGenomics.join(', ')}.`);
  const oldMissing = new Set(a.missing_information);
  const resolved = [...oldMissing].filter((x) => !b.missing_information.includes(x));
  if (resolved.length) out.push(`Previously missing information now resolved: ${resolved.map(pretty).join(', ')}.`);
  if (!out.length && before.state_hash !== after.state_hash) out.push('Your cancer history was updated, even though the main summary did not change.');
  if (!out.length) out.push('No other headline information changed with this scan.');
  return out;
}

export default function UpdatePage() {
  const [patient, setPatient] = useState<Patient | null>(null);
  const [stage, setStage] = useState<Stage>('baseline');
  const [beforeState, setBeforeState] = useState<StateResponse | null>(null);
  const [priorForecast, setPriorForecast] = useState<Forecast | null>(null);
  const [document, setDocument] = useState<DocumentRecord | null>(null);
  const [reviewFacts, setReviewFacts] = useState<CandidateFact[]>([]);
  const [afterState, setAfterState] = useState<StateResponse | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [whatChanged, setWhatChanged] = useState<IntelligenceArtifact | null>(null);
  const [intelligenceError, setIntelligenceError] = useState('');

  function persist(next: Partial<SavedFlow>) {
    if (!patient) return;
    const payload: SavedFlow = {
      patientId: patient.id,
      stage: next.stage ?? stage,
      beforeState: next.beforeState === undefined ? beforeState : next.beforeState,
      document: next.document === undefined ? document : next.document,
      afterState: next.afterState === undefined ? afterState : next.afterState,
      forecast: next.forecast === undefined ? forecast : next.forecast,
    };
    sessionStorage.setItem(FLOW_KEY, JSON.stringify(payload));
  }

  async function loadReview(p: Patient, docId: string) {
    const facts = await api<CandidateFact[]>(`/patients/${p.id}/review`);
    const filtered = facts.filter((f) => f.document_id === docId);
    setReviewFacts(filtered);
    return filtered;
  }

  async function bootstrap() {
    setError('');
    const p = await ensurePatient();
    setPatient(p);
    const [state, f] = await Promise.all([
      api<StateResponse>(`/patients/${p.id}/state`),
      api<Forecast>(`/patients/${p.id}/forecast`),
    ]);
    setPriorForecast(f?.id ? f : null);

    let saved: SavedFlow | null = null;
    try {
      const raw = sessionStorage.getItem(FLOW_KEY);
      saved = raw ? JSON.parse(raw) as SavedFlow : null;
    } catch {
      saved = null;
    }

    if (saved?.patientId === p.id && saved.document) {
      setBeforeState(saved.beforeState || state);
      setDocument(saved.document);
      setAfterState(saved.afterState);
      setForecast(saved.forecast);
      if (saved.stage === 'complete' && saved.forecast) {
        setStage('complete');
        try {
          const intel = await api<IntelligenceBundle>(`/patients/${p.id}/intelligence`);
          setWhatChanged(intel.what_changed || null);
        } catch { /* CP4 intelligence is additive; the scan flow remains usable. */ }
        return;
      }
      const pending = await loadReview(p, saved.document.id);
      setStage(pending.length ? 'verify' : 'ready');
      return;
    }

    setBeforeState(state);
    setStage('baseline');
  }

  useEffect(() => {
    bootstrap().catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  async function upload(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!patient || !beforeState) return;
    const input = e.currentTarget.elements.namedItem('file') as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    setBusy(true); setError(''); setMessage('Reading the report and connecting important details to the right source pages…'); setStage('processing');
    const form = new FormData(); form.append('file', file);
    try {
      const doc = await api<DocumentRecord>(`/patients/${patient.id}/documents?process=true`, { method: 'POST', body: form });
      setDocument(doc);
      persist({ stage: 'processing', document: doc, beforeState });
      input.value = '';
      if (doc.document_type && !['radiology', 'mixed'].includes(doc.document_type)) {
        setStage('ready');
        setMessage(`This record was classified as ${doc.document_type}. It remains in Records, but the newest-scan flow expects a radiology or mixed report.`);
        persist({ stage: 'ready', document: doc });
        return;
      }
      const pending = await loadReview(patient, doc.id);
      if (pending.length) {
        setStage('verify');
        setMessage('A few important details need your confirmation before they are added to your cancer history.');
        persist({ stage: 'verify', document: doc });
      } else {
        setStage('ready');
        persist({ stage: 'ready', document: doc });
        await finishUpdate(doc);
      }
    } catch (e) {
      setStage('baseline');
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function decide(id: string, decision: 'accept' | 'reject' | 'edit', value?: string) {
    if (!patient || !document) return;
    setBusy(true); setError('');
    try {
      await api(`/candidates/${id}/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, edited_value: value ?? null }),
      });
      const pending = await loadReview(patient, document.id);
      if (!pending.length) {
        setStage('ready');
        persist({ stage: 'ready' });
        await finishUpdate(document);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function finishUpdate(doc = document) {
    if (!patient || !doc) return;
    setBusy(true); setError(''); setMessage('Updating your cancer history and trajectory…');
    try {
      const state = await api<StateResponse>(`/patients/${patient.id}/state`);
      const f = await api<Forecast>(`/patients/${patient.id}/forecast`, { method: 'POST' });
      setAfterState(state);
      setForecast(f);
      setStage('complete');
      setMessage('Update complete. Your newest scan is now part of your cancer history and trajectory.');
      const saved: SavedFlow = { patientId: patient.id, stage: 'complete', beforeState, document: doc, afterState: state, forecast: f };
      sessionStorage.setItem(FLOW_KEY, JSON.stringify(saved));
      if (beforeState?.state_hash) {
        try {
          const intel = await api<IntelligenceArtifact>(`/patients/${patient.id}/intelligence/what-changed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ before_state_hash: beforeState.state_hash, document_id: doc.id }),
          });
          setWhatChanged(intel);
          setIntelligenceError('');
        } catch (intelError) {
          setIntelligenceError(intelError instanceof Error ? intelError.message : String(intelError));
        }
      }
    } catch (e) {
      setStage('ready');
      setError(e instanceof Error ? e.message : String(e));
      persist({ stage: 'ready' });
    } finally {
      setBusy(false);
    }
  }

  async function resetFlow() {
    sessionStorage.removeItem(FLOW_KEY);
    setDocument(null); setReviewFacts([]); setAfterState(null); setForecast(null); setWhatChanged(null); setIntelligenceError(''); setMessage(''); setError('');
    await bootstrap();
  }

  const step = stage === 'baseline' ? 1 : stage === 'processing' ? 2 : stage === 'verify' ? 3 : stage === 'ready' ? 3 : 4;
  const stateChanges = useMemo(() => changes(beforeState, afterState), [beforeState, afterState]);
  const scanSource = afterState?.sources.scan_assessment;
  const isWrongType = Boolean(document?.document_type && !['radiology', 'mixed'].includes(document.document_type));

  return (
    <div className="page update-page">
      <header className="page-head">
        <div><div className="eyebrow">NEW SCAN</div><h1>Add your newest scan</h1><p>OncoTwin will add the scan to your history and show how your trajectory changes.</p></div>
        <Link href="/app/records" className="secondary-button">All records</Link>
      </header>

      <ol className="journey-steps" aria-label="New scan update progress">
        {['Before scan', 'Add report', 'Verify', 'See update'].map((label, i) => <li key={label} className={step > i + 1 ? 'done' : step === i + 1 ? 'active' : ''}><span>{step > i + 1 ? '✓' : i + 1}</span><b>{label}</b></li>)}
      </ol>

      {error ? <div className="error-card" role="alert"><strong>We could not finish this step.</strong><p>{error}</p></div> : null}
      {message ? <div className="inline-status" aria-live="polite">{message}</div> : null}

      {stage !== 'complete' ? (
        <section className="before-card">
          <div>
            <div className="card-kicker">BEFORE THIS SCAN</div>
            <h2>{pretty(beforeState?.state.latest_scan?.assessment) === '—' ? 'No previous scan assessment yet' : `Latest committed scan: ${pretty(beforeState?.state.latest_scan?.assessment)}`}</h2>
            <p>{beforeState?.state.latest_scan?.date ? `Previous scan dated ${beforeState.state.latest_scan.date}.` : 'This is the latest scan already in your history.'}</p>
          </div>
          <div className="before-metrics">
            <span>Previous 6-month outlook</span>
            <strong>{priorForecast?.horizons?.['6'] ? pct(priorForecast.horizons['6'].selected_pfs) : 'Not calculated'}</strong>
          </div>
        </section>
      ) : null}

      {stage === 'baseline' || stage === 'processing' ? (
        <section className="scan-upload-card">
          <div className="scan-upload-copy"><div className="card-kicker">ADD RADIOLOGY REPORT</div><h2>Upload the newest scan report</h2><p>Use the final radiology report PDF. Important scan facts will either commit with source evidence or pause for your review.</p></div>
          <form onSubmit={upload}>
            <label className="file-drop"><input name="file" type="file" accept="application/pdf" disabled={busy} /><span className="file-drop-icon">PDF</span><span><b>Choose newest scan report</b><small>PDF only in this release</small></span></label>
            <button className="primary-button" disabled={busy}>{busy ? 'Processing report…' : 'Upload & add evidence'}</button>
          </form>
        </section>
      ) : null}

      {stage === 'verify' ? (
        <section className="section-block">
          <div className="section-title"><div><div className="eyebrow">VERIFY IMPORTANT FACTS</div><h2>Check what OncoTwin extracted before the update</h2></div><span className="counter">{reviewFacts.length} left</span></div>
          <div className="verify-list">{reviewFacts.map((f) => <FactReviewCard key={f.id} fact={f} onDecision={decide} busy={busy} />)}</div>
        </section>
      ) : null}

      {stage === 'ready' ? (
        <section className="feature-panel split-feature">
          <div><div className="card-kicker">READY TO UPDATE</div><h2>{isWrongType ? 'This file does not look like a radiology report' : 'Verification is complete'}</h2><p>{isWrongType ? 'The record was saved, but this flow will not treat a non-radiology document as the newest scan.' : 'Your scan is ready to be added to your history and trajectory.'}</p></div>
          {isWrongType ? <button onClick={resetFlow} className="secondary-button">Choose another scan</button> : <button onClick={() => finishUpdate()} disabled={busy} className="primary-button">{busy ? 'Updating…' : 'Update my trajectory'}</button>}
        </section>
      ) : null}

      {stage === 'complete' && afterState && forecast ? (
        <div className="update-results">
          <section className="after-banner"><div><div className="card-kicker">AFTER THIS SCAN</div><h2>Your cancer history has been updated</h2><p>Your newest scan is now part of your history and trajectory.</p></div><ModelSupport support={forecast.support} compact /></section>

          <section className="change-panels">
            <article className="change-card scan-report-card">
              <div className="change-icon">1</div>
              <div><div className="card-kicker">WHAT THE SCAN REPORTED</div><h3>{pretty(afterState.state.latest_scan?.assessment)}</h3><p>{afterState.state.latest_scan?.date ? `Report date ${afterState.state.latest_scan.date}.` : 'Scan date not established.'}</p><SourceViewer source={scanSource} label="Check scan source" /></div>
            </article>
            <article className="change-card forecast-change-card">
              <div className="change-icon">2</div>
              <div><div className="card-kicker">HOW YOUR OUTLOOK CHANGED</div>
                {forecast.horizons?.['6'] ? <><h3>{delta(forecast.horizons['6'].pre_pfs, forecast.horizons['6'].selected_pfs)} at 6 months</h3><p>Before this scan: {pct(forecast.horizons['6'].pre_pfs)} · After this scan: {pct(forecast.horizons['6'].selected_pfs)}.</p></> : <><h3>Quantitative update unavailable</h3><p>The support panel below explains why OncoTwin abstained.</p></>}
              </div>
            </article>
            <article className="change-card information-card">
              <div className="change-icon">3</div>
              <div><div className="card-kicker">WHAT NEW INFORMATION WAS ADDED</div><ul>{stateChanges.map((x) => <li key={x}>{x}</li>)}</ul></div>
            </article>
          </section>

          {whatChanged ? <section className="cp4-change-narrative">
            <div className="card-kicker">PUTTING THE CHANGE IN PLAIN LANGUAGE</div>
            <h3>{String(whatChanged.content?.narrative?.headline || 'What changed')}</h3>
            <p>{String(whatChanged.content?.narrative?.scan_summary || '')}</p>
            <p>{String(whatChanged.content?.narrative?.state_change_summary || '')}</p>
            <p className="muted">{String(whatChanged.content?.narrative?.outlook_context || '')}</p>
            <div className="cp4-source-row">{(whatChanged.grounding || []).map((s) => s.document_id && s.page_number ? <SourceViewer key={s.key} source={{ document_id: s.document_id, document_name: s.document_name || 'Source record', page_number: s.page_number, source_snippet: s.source_snippet || '' }} label="Check source" /> : null)}</div>
          </section> : intelligenceError ? <div className="notice-card"><strong>The scan and forecast updated normally.</strong><p>The optional plain-language explanation is temporarily unavailable: {intelligenceError}</p></div> : null}

          {forecast.curves ? <section className="trajectory-card update-chart"><div className="trajectory-card-head"><div><div className="card-kicker">BEFORE → AFTER</div><h2>How this scan changed your trajectory</h2><p>Compare your outlook before this scan with your updated outlook after the scan was added.</p></div></div><ForecastChart forecast={forecast} /></section> : null}

          {forecast.horizons ? <section className="horizon-grid update-horizons">{['3','6','12','18'].map((m) => { const h = forecast.horizons?.[m]; return <article key={m} className="horizon-card"><span>{m} months</span><strong>{pct(h?.selected_pfs)}</strong><p>After this scan</p><div className="horizon-delta"><span>Before {pct(h?.pre_pfs)}</span><b>{delta(h?.pre_pfs, h?.selected_pfs).replace(' percentage points',' pp')}</b></div></article>; })}</section> : null}

          <section className="section-block support-section"><div className="section-title"><div><div className="eyebrow">ABOUT THIS ESTIMATE</div><h2>How much information is available</h2></div></div><ModelSupport support={forecast.support} /></section>

                    {patient ? <AskOncoTwin patientId={patient.id} /> : null}

          <div className="bottom-actions"><Link href="/app/forecast" className="primary-button">Back to trajectory</Link><Link href="/app/prepare" className="secondary-button">Prepare for your visit</Link><button onClick={resetFlow} className="secondary-button">Add another scan</button></div>
        </div>
      ) : null}
    </div>
  );
}
