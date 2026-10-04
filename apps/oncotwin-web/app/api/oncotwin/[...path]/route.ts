import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';

const API = process.env.ONCOTWIN_API_INTERNAL ?? 'http://127.0.0.1:8000';

async function proxy(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const cookieStore = await cookies();
  const token = cookieStore.get('oncotwin_demo_token')?.value;
  if (!token) return NextResponse.json({ detail: 'No session' }, { status: 401 });

  const target = new URL(`${API}/v1/${path.join('/')}`);
  req.nextUrl.searchParams.forEach((v, k) => target.searchParams.set(k, v));

  const headers = new Headers();
  headers.set('Authorization', `Bearer ${token}`);
  const contentType = req.headers.get('content-type');
  if (contentType) headers.set('content-type', contentType);

  const hasBody = !['GET', 'HEAD'].includes(req.method);
  const response = await fetch(target, {
    method: req.method,
    headers,
    body: hasBody ? await req.arrayBuffer() : undefined,
    cache: 'no-store',
  });
  const outHeaders = new Headers();
  const responseType = response.headers.get('content-type');
  if (responseType) outHeaders.set('content-type', responseType);
  return new NextResponse(await response.arrayBuffer(), { status: response.status, headers: outHeaders });
}

export const GET = proxy;
export const POST = proxy;
export const DELETE = proxy;
export const PUT = proxy;
export const PATCH = proxy;
