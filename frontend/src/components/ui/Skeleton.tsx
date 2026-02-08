export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse bg-white/5 rounded-lg ${className}`} />;
}

export function PanelSkeleton({ height = 'h-64' }: { height?: string }) {
  return (
    <div className={`glass-panel rounded-xl p-4 ${height}`}>
      <div className="flex items-center justify-between mb-4">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-4 w-20" />
      </div>
      <Skeleton className="h-full w-full rounded-lg" />
    </div>
  );
}
