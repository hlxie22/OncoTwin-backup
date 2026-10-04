'use client';

import { useEffect, useMemo, useState } from 'react';
import { api, ensurePatient } from '@/lib/api';
import type { TimelineEvent } from '@/lib/types';
import { SourceViewer } from '@/components/SourceViewer';

function pretty(value: string) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (x) => x.toUpperCase());
}

export default function TimelinePage() {
  const [events, setEvents] = useState<TimelineEvent[] | null>(null);
  const [filter, setFilter] = useState('all');
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const p = await ensurePatient();
        setEvents(await api<TimelineEvent[]>(`/patients/${p.id}/timeline`));
      } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    })();
  }, []);

  const categories = useMemo(() => events ? ['all', ...Array.from(new Set(events.map((e) => e.event_type)))] : ['all'], [events]);
  const shown = useMemo(() => events?.filter((e) => filter === 'all' || e.event_type === filter) ?? [], [events, filter]);

  return <div className="page">
    <header className="page-head"><div><div className="eyebrow">YOUR CANCER HISTORY</div><h1>Your cancer timeline</h1></div></header>
    {error ? <div className="error-card" role="alert"><strong>Timeline could not be loaded.</strong><p>{error}</p></div> : null}
    {events && events.length ? <div className="filter-row" aria-label="Filter timeline">{categories.map((x) => <button key={x} className={filter === x ? 'filter-chip active' : 'filter-chip'} aria-pressed={filter === x} onClick={() => setFilter(x)}>{x === 'all' ? 'All events' : pretty(x)}</button>)}</div> : null}
    {!events ? <div className="skeleton big" aria-label="Loading timeline" /> : shown.length ? <div className="timeline">{shown.map((e, i) => <div className="timeline-item" key={e.id}>
      <div className="timeline-rail"><span>{i + 1}</span></div>
      <article><div className="timeline-meta">{e.event_date ?? 'Date not established'} · {pretty(e.event_type)}</div><h2>{e.title}</h2><p>{e.summary}</p><SourceViewer source={e.source} /></article>
    </div>)}</div> : <div className="empty-state"><strong>{events.length ? 'No events match this filter.' : 'Your timeline is empty.'}</strong><p>{events.length ? 'Choose another event type.' : 'Upload a record and committed facts will appear here.'}</p></div>}
  </div>;
}
