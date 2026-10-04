'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { ReactNode } from 'react';

const links = [
  ['/app', 'Overview'],
  ['/app/forecast', 'Trajectory'],
  ['/app/records', 'Records'],
  ['/app/verify', 'Review'],
  ['/app/timeline', 'Timeline'],
] as const;

function isActive(pathname: string, href: string) {
  return href === '/app' ? pathname === href : pathname === href || pathname.startsWith(`${href}/`);
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Patient workspace navigation">
        <Link href="/app" className="brand" aria-label="OncoTwin overview">
          <span className="brand-mark" aria-hidden="true">O</span>
          <span>OncoTwin</span>
        </Link>
        <div className="nav-label">PATIENT WORKSPACE</div>
        <nav>
          {links.map(([href, label]) => (
            <Link
              key={href}
              href={href}
              className={isActive(pathname, href) ? 'nav-link active' : 'nav-link'}
              aria-current={isActive(pathname, href) ? 'page' : undefined}
            >
              {label}
            </Link>
          ))}
        </nav>
        <div className="sidebar-note">
          <strong>Research release</strong>
        </div>
      </aside>
      <main className="app-main" id="main-content">{children}</main>
    </div>
  );
}
