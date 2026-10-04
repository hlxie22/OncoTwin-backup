'use client';

import type { Patient } from './types';

export async function ensureSession(): Promise<void> {
  const res = await fetch('/api/session', { method: 'POST', cache: 'no-store' });
  if (!res.ok) throw new Error(await res.text());
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  await ensureSession();
  const res = await fetch(`/api/oncotwin${path}`, { ...init, cache: 'no-store' });
  if (!res.ok) {
    const text = await res.text();
    let message = text || `${res.status} ${res.statusText}`;
    try {
      const parsed = JSON.parse(text);
      if (parsed?.detail) message = String(parsed.detail);
    } catch {
      // Keep the plain response text.
    }
    throw new Error(message);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export async function ensurePatient(): Promise<Patient> {
  const patients = await api<Patient[]>('/patients');
  if (patients.length) return patients[0];
  return api<Patient>('/patients', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display_name: 'My cancer journey', disease_pack: 'mbc_v2' }),
  });
}
