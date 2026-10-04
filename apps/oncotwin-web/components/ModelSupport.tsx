import type { ForecastSupport } from '@/lib/types';

export function ModelSupport({ support, compact = false }: { support?: ForecastSupport; compact?: boolean }) {
  const status = support?.status || 'UNKNOWN';
  const normalized = status.toLowerCase();
  return (
    <div className={compact ? 'support-compact' : 'support-panel'}>
      <div className={`support-badge ${normalized}`}>{status === 'STANDARD' ? 'Enough information' : status === 'LIMITED' ? 'Limited information' : status === 'UNAVAILABLE' ? 'Not enough information' : status}</div>
      {!compact && (
        support?.explanations?.length ? (
          <ul className="support-list">
            {support.explanations.map((x, i) => <li key={`${i}-${x}`}>{x}</li>)}
          </ul>
        ) : (
          <p className="muted support-copy">OncoTwin has enough information to show this trajectory.</p>
        )
      )}
    </div>
  );
}
