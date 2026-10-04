'use client';

import { FormEvent, useEffect, useState } from 'react';
import Link from 'next/link';
import { api, ensurePatient } from '@/lib/api';
import type { DocumentRecord, Forecast, Patient, StateResponse } from '@/lib/types';
import { ForecastChart } from '@/components/ForecastChart';
import { ModelSupport } from '@/components/ModelSupport';

function pct(v: number | undefined | null): string {
  return typeof v === 'number' && Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—';
}

function pp(before: number | undefined, after: number | undefined) {
  if (typeof before !== 'number' || typeof after !== 'number') return '—';
  const d = Math.round((after - before) * 100);
  return `${d > 0 ? '+' : ''}${d} pp`;
}

function pretty(value: string | null | undefined) {
  return value ? value.replaceAll('_', ' ').replace(/\b\w/g, (x) => x.toUpperCase()) : 'Processing';
}

export default function ForecastPage() {
  const [patient, setPatient] = useState<Patient | null>(null);
  const [state, setState] = useState<StateResponse | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const [uploadProgress, setUploadProgress] = useState('');
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [showPost, setShowPost] = useState(false);
  const [error, setError] = useState('');

  async function refresh(p?: Patient) {
    const pat = p ?? patient;
    if (!pat) return;
    const [s, f, documents] = await Promise.all([
      api<StateResponse>(`/patients/${pat.id}/state`),
      api<Forecast>(`/patients/${pat.id}/forecast`),
      api<DocumentRecord[]>(`/patients/${pat.id}/documents`),
    ]);
    setState(s);
    setForecast(f?.id ? f : null);
    setDocs(documents);
  }

  useEffect(() => {
    (async () => {
      try {
        const p = await ensurePatient();
        setPatient(p);
        await refresh(p);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!patient) return;
    const onFocus = () => { refresh(patient).catch(() => undefined); };
    window.addEventListener('focus', onFocus);
    const onVisibility = () => { if (document.visibilityState === 'visible') onFocus(); };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('focus', onFocus);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [patient]);

  async function uploadRecords(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!patient) return;
    const input = e.currentTarget.elements.namedItem('files') as HTMLInputElement;
    const selected = Array.from(input.files ?? []);
    if (!selected.length) return;

    setUploadBusy(true);
    setUploadMessage('');
    setUploadProgress(`Uploading 0 of ${selected.length}...`);
    setError('');

    let completed = 0;
    let needsReview = 0;
    const failures: string[] = [];

    for (const file of selected) {
      try {
        const form = new FormData();
        form.append('file', file);
        const result = await api<DocumentRecord>(`/patients/${patient.id}/documents?process=true`, {
          method: 'POST',
          body: form,
        });
        completed += 1;
        if (result.status === 'needs_review') needsReview += 1;
      } catch (err) {
        failures.push(`${file.name}: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setUploadProgress(`Processed ${completed + failures.length} of ${selected.length}...`);
      }
    }

    input.value = '';
    await refresh(patient);

    if (failures.length) {
      setError(`Some records could not be added: ${failures.join(' | ')}`);
    }
    if (completed) {
      setUploadMessage(
        needsReview
          ? `${completed} record${completed === 1 ? '' : 's'} added. ${needsReview} need${needsReview === 1 ? 's' : ''} review.`
          : `${completed} record${completed === 1 ? '' : 's'} added to your cancer history.`
      );
    }
    setUploadProgress('');
    setUploadBusy(false);
  }

  async function generate() {
    if (!patient) return;
    setRunning(true);
    setError('');
    try {
      const f = await api<Forecast>(`/patients/${patient.id}/forecast`, { method: 'POST' });
      setForecast(f);
      setState(await api<StateResponse>(`/patients/${patient.id}/state`));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  if (loading) return <div className="page"><div className="skeleton big" aria-label="Loading your trajectory" /></div>;

  const stale = Boolean(forecast?.id && state?.state_hash && forecast.state_hash !== state.state_hash);
  const supported = forecast?.run_status === 'COMPLETED' && forecast.curves && forecast.horizons;

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">TRAJECTORY</div>
          <h1>How your outlook is changing</h1>
          <p>See how your outlook changes as new scans are added to your history.</p>
        </div>
        {forecast ? <ModelSupport support={forecast.support} compact /> : null}
      </header>

      {error ? <div className="error-card" role="alert"><strong>Forecast could not be updated.</strong><p>{error}</p></div> : null}
      {stale ? <div className="notice-card"><strong>Your records changed after this forecast.</strong><p>Recalculate to use your latest records.</p><button onClick={generate} disabled={running} className="secondary-button">{running ? 'Recalculating…' : 'Recalculate now'}</button></div> : null}

      <section className="upload-panel modern-upload">
        <form onSubmit={uploadRecords}>
          <div>
            <div className="card-kicker">ADD RECORDS</div>
            <strong>Add records to build your history</strong>
            <p>Select one PDF or several at once. OncoTwin will process each record and flag anything that needs your review.</p>
          </div>
          <label className="file-button">
            <input name="files" type="file" accept="application/pdf,.pdf" multiple disabled={uploadBusy} />
            Choose PDFs
          </label>
          <button className="primary-button" disabled={uploadBusy}>
            {uploadBusy ? (uploadProgress || 'Processing...') : 'Upload & process'}
          </button>
        </form>
        {uploadMessage ? <div className="inline-message" aria-live="polite">{uploadMessage}</div> : null}
      </section>

      {docs.length ? (
        <section className="section-block">
          <div className="section-title">
            <div><div className="eyebrow">YOUR RECORDS</div><h2>{docs.length} uploaded record{docs.length === 1 ? '' : 's'}</h2></div>
            {docs.some((d) => d.status === 'needs_review') ? <Link href="/app/verify" className="secondary-button">Review flagged details</Link> : null}
          </div>
          <div className="record-list">
            {docs.map((d) => (
              <article className="record-row" key={d.id}>
                <div className="record-icon">PDF</div>
                <div className="record-main">
                  <strong>{d.filename}</strong>
                  <span>{pretty(d.document_type)} · added {new Date(d.created_at).toLocaleDateString()}</span>
                  {d.processing_error ? <em>{d.processing_error}</em> : null}
                </div>
                <span className={`status ${d.status}`}>{pretty(d.status)}</span>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {!forecast ? (
        <section className="feature-panel forecast-empty">
          <div>
            <div className="card-kicker">NO FORECAST YET</div>
            <h2>Build your trajectory from your current records</h2>
            <p>OncoTwin needs a dated metastatic diagnosis and at least one scan before it can show a trajectory.</p>
          </div>
          <button className="primary-button" disabled={running} onClick={generate}>{running ? 'Checking…' : 'Show my trajectory'}</button>
        </section>
      ) : (
        <>
          <section className="trajectory-card">
            <div className="trajectory-card-head">
              <div>
                <div className="card-kicker">24-MONTH VIEW</div>
                <h2>Your trajectory</h2>
                <p>Compare your outlook before the newest scan with your outlook after that scan was added.</p>
              </div>
              {forecast.current_scan?.date ? <span className="date-pill">Newest scan {forecast.current_scan.date}</span> : null}
            </div>

            {supported ? <ForecastChart forecast={forecast} showPost={showPost} /> : <div className="unavailable-panel"><strong>OncoTwin cannot show a trajectory yet.</strong><p>See below for the information that is still needed.</p></div>}

            {forecast.horizons ? (
              <div className="horizon-grid">
                {['3', '6', '12', '18'].map((m) => {
                  const h = forecast.horizons?.[m];
                  return (
                    <article key={m} className="horizon-card">
                      <span>{m} months</span>
                      <strong>{pct(h?.selected_pfs)}</strong>
                      <p>After this scan</p>
                      <div className="horizon-delta"><span>Before this scan {pct(h?.pre_pfs)}</span><b>{pp(h?.pre_pfs, h?.selected_pfs)}</b></div>
                    </article>
                  );
                })}
              </div>
            ) : null}
          </section>
          <section className="explain-grid">
            <article className="explain-card">
              <div className="explain-number">1</div>
              <div><div className="card-kicker">BEFORE THIS SCAN</div><h3>Your earlier outlook</h3><p>Based on the history in your record before the newest scan.</p></div>
            </article>
            <article className="explain-card">
              <div className="explain-number">2</div>
              <div><div className="card-kicker">AFTER THIS SCAN</div><h3>Your updated outlook</h3><p>Updated after adding the newest scan to that same history.</p></div>
            </article>
            <article className="explain-card">
              <div className="explain-number">3</div>
              <div><div className="card-kicker">WHAT CHANGED</div><h3>The effect of the new scan</h3><p>The difference shows how much the newest scan changed the estimate.</p></div>
            </article>
          </section>

          <section className="section-block support-section">
            <div className="section-title"><div><div className="eyebrow">ABOUT THIS ESTIMATE</div><h2>How much information is available</h2></div></div>
            <ModelSupport support={forecast.support} />
          </section>

          <section className="model-details-card">
            <details>
              <summary>Technical model details</summary>
              <div className="details-body">
                <p>These details are included for technical review and reproducibility.</p>
                {supported ? <button className="secondary-button" onClick={() => setShowPost((x) => !x)}>{showPost ? 'Hide full technical curve' : 'Show full technical curve'}</button> : null}
                <dl className="model-metadata">
                  <div><dt>Release</dt><dd>{forecast.model?.checkpoint || 'V2-07'}</dd></div>
                  <div><dt>Selected rule</dt><dd>PRE + 0.52 × (POST − PRE)</dd></div>
                  <div><dt>Run</dt><dd>{forecast.id?.slice(0, 8) || '—'}</dd></div>
                </dl>
              </div>
            </details>
          </section>

          <div className="bottom-actions">
            <Link href="/app/update" className="primary-button">Add newest scan</Link>
            <Link href="/app/prepare" className="secondary-button">Prepare for a visit</Link>
            <button onClick={generate} disabled={running} className="secondary-button">{running ? 'Recalculating…' : 'Recalculate for current state'}</button>
          </div>
        </>
      )}
    </div>
  );
}
