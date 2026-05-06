import { loadTheoremDashboardData } from '@/lib/server/theorem-data';

import { TheoremsClient } from './TheoremsClient';

export const dynamic = 'force-dynamic';

export default async function TheoremsPage() {
  const data = await loadTheoremDashboardData();
  return <TheoremsClient data={data} />;
}
