'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { api, ensurePatient } from '@/lib/api';
import type { CandidateFact, Forecast, Patient, StateResponse, TimelineEvent } from '@/lib/types';
import { SourceViewer } from '@/components/SourceViewer';
import { ModelSupport } from '@/components/ModelSupport';
import { AskOncoTwin } from '@/components/AskOncoTwin';

function pct(v: number | undefined | null) {
  return typeof v === 'number' && Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—';
}

function pretty(value: string | null | undefined) {
  if (!value) return null;
  return value.replaceAll('_', ' ').replace(/\b\w/g, (x) => x.toUpperCase());
}

export default function OverviewPage() {
  const [patient, setPatient] = useState<Patient | null>(null);
  const [data, setData] = useState<StateResponse | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [reviewCount, setReviewCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const p = await ensurePatient();
        setPatient(p);
        const [state, f, timeline, review] = await Promise.all([
          api<StateResponse>(`/patients/${p.id}/state`),
          api<Forecast>(`/patients/${p.id}/forecast`),
          api<TimelineEvent[]>(`/patients/${p.id}/timeline`),
          api<CandidateFact[]>(`/patients/${p.id}/review`),
        ]);
        setData(state);
        setForecast(f?.id ? f : null);
        setEvents(timeline);
        setReviewCount(review.length);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const recent = useMemo(() => [...events].reverse().slice(0, 3), [events]);

  if (error) return <div className="page"><div className="error-card"><strong>We could not load your workspace.</strong><p>{error}</p></div></div>;
  if (!data) return <div className="page"><div className="skeleton big" aria-label="Loading your cancer history" /></div>;

  const s = data.state;
  const forecastCurrent = Boolean(forecast?.id && forecast.state_hash && forecast.state_hash === data.state_hash);
  const h6 = forecast?.horizons?.['6'];
  const h12 = forecast?.horizons?.['12'];

  return (
    <div className="page">
      <header className="page-head overview-head">
        <div>
          <div className="eyebrow">YOUR ONCOTWIN</div>
          <h1>Your cancer right now</h1>
          <p>A summary of what your records show today.</p>
        </div>
        <div className="head-actions">
          {reviewCount > 0 ? <Link className="attention-pill" href="/app/verify">{reviewCount} item{reviewCount === 1 ? '' : 's'} to review</Link> : null}
        </div>
      </header>

      <section className="overview-hero">
        <div className="overview-hero-copy">
          <div className="card-kicker">CURRENT CANCER PICTURE</div>
          <h2>{s.diagnosis ?? 'Diagnosis not yet established from uploaded records'}</h2>
          <p className="hero-subtype">{s.subtype ?? 'Subtype information is still incomplete'}</p>
          <div className="receptor-row">
            {Object.entries(s.receptors).map(([k, v]) => <span key={k} className="receptor"><b>{k.toUpperCase()}</b> {v ?? 'unknown'}</span>)}
          </div>
          <SourceViewer source={data.sources.her2_status ?? data.sources.er_status ?? data.sources.diagnosis} label="Check diagnosis source" />
        </div>
        <div className="next-action-card">
          <div className="card-kicker">NEXT UPDATE</div>
          <h3>Add your newest scan when it arrives</h3>
          <p>Add a new scan when you get one. OncoTwin will show what changed and update your trajectory.</p>
          <Link href="/app/update" className="primary-button">Add newest scan</Link>
        </div>
      </section>

      <section className="summary-grid overview-grid">
        <article className="summary-card">
          <div className="card-kicker">CURRENT TREATMENT</div>
          <h3>{s.current_treatment?.name ?? 'No current treatment found'}</h3>
          <p>{s.current_treatment?.date ? `Recorded ${s.current_treatment.date}` : 'Treatment dates remain visible when they are available in your records.'}</p>
          <SourceViewer source={data.sources.treatment} />
        </article>
        <article className="summary-card">
          <div className="card-kicker">LATEST SCAN</div>
          <h3>{pretty(s.latest_scan?.assessment) ?? 'No scan assessment found'}</h3>
          <p>{s.latest_scan?.date ?? 'Add a radiology report to update your scan history.'}</p>
          <SourceViewer source={data.sources.scan_assessment} />
        </article>
        <article className="summary-card">
          <div className="card-kicker">KNOWN DISEASE SITES</div>
          <div className="chip-row">{s.disease_sites.length ? s.disease_sites.map((x) => <span className="chip" key={x}>{pretty(x)}</span>) : <span className="muted">No committed disease sites yet</span>}</div>
        </article>
        <article className="summary-card">
          <div className="card-kicker">GENOMICS IN YOUR RECORD</div>
          <div className="chip-row">{s.genomics.length ? s.genomics.map((x) => <span className="chip" key={x.fact_id}>{x.alteration}</span>) : <span className="muted">No committed alterations yet</span>}</div>
          <SourceViewer source={data.sources.genomic_alteration} />
        </article>
      </section>

            {patient ? <AskOncoTwin patientId={patient.id} /> : null}

<section className="section-block">
        <div className="section-title">
          <div><div className="eyebrow">TRAJECTORY</div><h2>Your latest outlook</h2></div>
          <Link href="/app/forecast" className="text-link">Open trajectory →</Link>
        </div>
        {!forecast ? (
          <div className="feature-panel split-feature">
            <div><h3>No forecast has been calculated yet</h3><p>Once your records include when metastatic cancer was diagnosed and at least one scan, OncoTwin can show your trajectory.</p></div>
            <Link href="/app/forecast" className="primary-button">Check model readiness</Link>
          </div>
        ) : (
          <div className="forecast-snapshot">
            <div>
              <div className="forecast-snapshot-top">
                <ModelSupport support={forecast.support} compact />
                {!forecastCurrent ? <span className="stale-pill">Records changed since this forecast</span> : <span className="current-pill">Current state</span>}
              </div>
              <h3>{forecast.run_status === 'COMPLETED' ? 'Your latest progression-free outlook' : 'The quantitative forecast is unavailable for this state'}</h3>
              <p>{forecast.run_status === 'COMPLETED' ? 'See your latest trajectory and how it changed after your newest scan.' : 'Open the trajectory page to see the exact support limitation.'}</p>
            </div>
            {forecast.run_status === 'COMPLETED' ? (
              <div className="snapshot-numbers">
                <div><span>6 months</span><strong>{pct(h6?.selected_pfs)}</strong></div>
                <div><span>12 months</span><strong>{pct(h12?.selected_pfs)}</strong></div>
              </div>
            ) : null}
          </div>
        )}
      </section>

      <section className="section-block two-column-section">
        <div>
          <div className="section-title"><div><div className="eyebrow">RECENT HISTORY</div><h2>Latest changes in your record</h2></div><Link href="/app/timeline" className="text-link">Full timeline →</Link></div>
          {recent.length ? <div className="recent-list">{recent.map((e) => (
            <article key={e.id} className="recent-item">
              <div><span className="timeline-meta">{e.event_date ?? 'Date not established'} · {pretty(e.event_type)}</span><h3>{e.title}</h3><p>{e.summary}</p></div>
              <SourceViewer source={e.source} />
            </article>
          ))}</div> : <div className="empty-state compact"><strong>No timeline events yet</strong><p>Uploaded records will build this history.</p></div>}
        </div>
        <div>
          <div className="section-title"><div><div className="eyebrow">YOUR RECORDS</div><h2>What is still missing</h2></div></div>
          {s.missing_information.length ? <div className="missing-stack">{s.missing_information.map((x) => <div className="missing-item" key={x}><span>!</span><div><strong>{pretty(x)}</strong><p>This information has not been confirmed from your records yet.</p></div></div>)}</div> : <div className="success-card large"><strong>Your core cancer information is available.</strong><p>You can keep adding records as your care changes.</p></div>}
        </div>
      </section>
    </div>
  );
}
