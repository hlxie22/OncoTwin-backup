import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'OncoTwin',
  description: 'Patient cancer intelligence platform',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
