import { Suspense } from 'react';
import { Skeleton } from '@/components/ui/Skeleton';
import { ZoneDetail } from '@/components/dashboard/ZoneDetail';

interface PageProps {
  params: Promise<{ zoneId: string }>;
}

export default async function ZonePage({ params }: PageProps) {
  const { zoneId } = await params;

  return (
    <div className="flex flex-col gap-6 p-6">
      <h1 className="text-2xl font-bold text-slate-100">
        Zone Control: <span className="text-gradient">{zoneId.toUpperCase()}</span>
      </h1>
      <Suspense fallback={<Skeleton className="w-full h-96" />}>
        <ZoneDetail zoneId={zoneId} />
      </Suspense>
    </div>
  );
}
