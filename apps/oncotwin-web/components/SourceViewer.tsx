'use client';

import { useEffect, useId, useRef, useState } from 'react';
import type { Source } from '@/lib/types';

export function SourceViewer({ source, label = 'View source' }: { source?: Source | null; label?: string }) {
  const [open, setOpen] = useState(false);
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false);
        requestAnimationFrame(() => triggerRef.current?.focus());
      }
    };
    window.addEventListener('keydown', onKey);
    requestAnimationFrame(() => closeRef.current?.focus());
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  if (!source) return null;

  const close = () => {
    setOpen(false);
    requestAnimationFrame(() => triggerRef.current?.focus());
  };

  return (
    <>
      <button ref={triggerRef} className="source-link" onClick={() => setOpen(true)}>{label}</button>
      {open && (
        <div className="modal-backdrop" onMouseDown={close}>
          <div
            className="source-modal"
            onMouseDown={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
          >
            <div className="modal-head">
              <div>
                <span className="eyebrow">WHERE THIS CAME FROM</span>
                <h3 id={titleId}>{source.document_name} · page {source.page_number}</h3>
              </div>
              <button ref={closeRef} className="icon-button" onClick={close} aria-label="Close source viewer">×</button>
            </div>
            <div className="source-grid">
              <img
                src={`/api/oncotwin/documents/${source.document_id}/pages/${source.page_number}/image`}
                alt={`Rendered page ${source.page_number} of ${source.document_name}`}
              />
              <div className="source-copy">
                <div className="card-kicker">SUPPORTING TEXT</div>
                <blockquote className="snippet">“{source.source_snippet}”</blockquote>
                <p className="muted">OncoTwin keeps important facts connected to the original report so you can check the source directly.</p>
                <p className="modal-tip">Press Esc or click outside this window to close it.</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
