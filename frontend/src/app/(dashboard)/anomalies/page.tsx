'use client';

import { AnomalyTimeline } from '@/components/charts/AnomalyTimeline';
import { AnomalyList } from '@/components/charts/AnomalyList';
import { Panel } from '@/components/ui/Panel';
import { mockAnomalies, mockAnomalyZScores } from '@/lib/api/mock-data';

export default function AnomaliesPage() {
  const anomalies = mockAnomalies();
  const zScores = mockAnomalyZScores(72);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-bold text-white">Anomaly Detection</h1>

      <AnomalyTimeline data={zScores} />

      <Panel title="Event Log" badge={`${anomalies.length} events`} badgeColor="warn">
        <AnomalyList anomalies={anomalies} />
      </Panel>
    </div>
  );
}
