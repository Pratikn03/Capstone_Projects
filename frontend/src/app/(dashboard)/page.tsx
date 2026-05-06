import { loadTheoremDashboardData } from '@/lib/server/theorem-data';
import { ResearchHubClient } from './ResearchHubClient';

export const dynamic = 'force-dynamic';

export default async function ResearchHubPage() {
  const theoremData = await loadTheoremDashboardData();

  return <ResearchHubClient theoremData={theoremData} />;
}
