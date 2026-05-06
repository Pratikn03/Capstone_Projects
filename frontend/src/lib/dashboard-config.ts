export const dashboardConfig = {
  operatorName: process.env.NEXT_PUBLIC_DASHBOARD_OPERATOR_NAME || 'Operator',
  operatorRole: process.env.NEXT_PUBLIC_DASHBOARD_OPERATOR_ROLE || 'ORIUS Researcher',
  appLabel: process.env.NEXT_PUBLIC_DASHBOARD_APP_LABEL || 'ORIUS Research',
};

export function operatorInitial(name = dashboardConfig.operatorName): string {
  const trimmed = name.trim();
  return trimmed ? trimmed.charAt(0).toUpperCase() : 'O';
}
