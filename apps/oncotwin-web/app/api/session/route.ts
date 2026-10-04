import { NextResponse } from 'next/server';

const API = process.env.ONCOTWIN_API_INTERNAL ?? 'http://127.0.0.1:8000';

export async function POST() {
  const existing = NextResponse.json({ ok: true });
  // Cookie access is handled by checking request cookies in the proxy; session creation
  // remains cheap in demo mode and returns the same demo user.
  const response = await fetch(`${API}/v1/auth/demo`, { method: 'POST', cache: 'no-store' });
  if (!response.ok) return new NextResponse(await response.text(), { status: response.status });
  const body = await response.json();
  existing.cookies.set('oncotwin_demo_token', body.access_token, {
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    path: '/',
    maxAge: 60 * 60 * 24,
  });
  return existing;
}
