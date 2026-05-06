export type TheoremTier = 'flagship' | 'supporting' | 'structural' | 'definition' | 'draft';

export type PromotionGateStatus = 'pass' | 'pending';

export interface TheoremPromotionGate {
  key: string;
  label: string;
  status: PromotionGateStatus;
  detail: string;
}

export interface TheoremDashboardRow {
  id: string;
  displayId: string;
  name: string;
  tier: TheoremTier;
  roleLabel: string;
  chapter: string;
  assumptions: string[];
  statement: string;
  formal: string;
  significance: string;
  batteryInstantiation: string;
  proofMethod: string;
  sourceLocation: string;
  domainApplicability: string;
  promotionGates: TheoremPromotionGate[];
  sortOrder: number;
}

export interface TheoremDashboardSummary {
  flagshipDefended: number;
  supportingDefended: number;
  draftNonDefended: number;
  definitionIds: string[];
  sourcePath: string;
  sourceUpdatedAt: string | null;
}

export interface TheoremDashboardData {
  theorems: TheoremDashboardRow[];
  summary: TheoremDashboardSummary;
  promotionPolicy: string;
}
