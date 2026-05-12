import 'server-only';

import { existsSync } from 'fs';
import fs from 'fs/promises';
import path from 'path';

import type {
  TheoremDashboardData,
  TheoremDashboardRow,
  TheoremDashboardSummary,
  TheoremPromotionGate,
  TheoremTier,
} from '@/lib/theorem-dashboard-types';

type AuditAnchor = {
  path?: string;
  symbol?: string;
  location?: string;
  note?: string;
};

type AuditRow = {
  theorem_id: string;
  title: string;
  surface_kind: string;
  defense_tier: string;
  proof_tier: string;
  program_role: string;
  scope_note: string;
  statement_location: string;
  proof_location: string;
  assumptions_used?: string[];
  typed_obligations?: string[];
  unresolved_assumptions?: string[];
  weakest_step: string;
  rigor_rating: string;
  code_correspondence: string;
  code_correspondence_detail?: string;
  remediation_detail?: string;
  generator_targets?: string[];
  code_anchors?: AuditAnchor[];
  test_anchors?: AuditAnchor[];
};

type AuditPayload = {
  theorems: AuditRow[];
  summary?: {
    defense_tier_counts?: Record<string, number>;
  };
};

type PromotionGateCsvRow = {
  theorem_id: string;
  gate: string;
  gate_pass: string;
  evidence: string;
  blocker: string;
};

type PromotionMatrixCsvRow = {
  theorem_id: string;
  statement_complete?: string;
  assumptions_complete?: string;
  proof_file?: string;
  code_anchor?: string;
  tests_count?: string;
  artifacts_count?: string;
  claim_boundary?: string;
  promotion_ready?: string;
  artifact_hashes_complete?: string;
};

type PromotionScorecardCandidate = {
  title?: string;
  current_tier?: string;
  candidate_status?: string;
  promotion_ready?: boolean;
  blocking_gates?: string[];
  weakest_step?: string;
  remediation_detail?: string;
};

type PromotionScorecardPayload = {
  promotion_ready?: boolean;
  candidates?: Record<string, PromotionScorecardCandidate>;
};

type PromotionGateLoadResult = {
  rowsByTheorem: Map<string, PromotionGateCsvRow[]>;
  sourceAvailable: boolean;
  scorecardAvailable: boolean;
  scorecard: PromotionScorecardPayload;
};

const DASHBOARD_THEOREM_IDS = ['T1', 'T2', 'T3a', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11'];
const SORT_ORDER = new Map(DASHBOARD_THEOREM_IDS.map((id, index) => [id, index]));
const PROMOTION_PACKAGE_REQUIRED_IDS = new Set<string>();
const PROMOTION_SCORECARD_SOURCE = 'reports/publication/theorem_promotion_matrix.csv';
const THEOREM_PROMOTION_AUTHORITY_LABEL =
  'promotion authority: reports/publication/theorem_promotion_matrix.csv + theorem_result_cards/*.json';

const PROMOTION_POLICY =
  'A theorem can be flagship only if it has statement, assumptions, proof, code anchor, tests, artifact evidence, artifact hashes, manuscript anchor, and a claim-boundary note. The dashboard follows the active audit and the generated theorem promotion matrix; it does not hand-demote scoped flagship rows.';

function resolveRepoRoot(): string {
  let current = process.cwd();
  for (let depth = 0; depth < 8; depth += 1) {
    if (existsSync(path.join(current, 'reports', 'publication')) && existsSync(path.join(current, 'frontend'))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }

  const cwd = process.cwd();
  return path.basename(cwd) === 'frontend' ? path.resolve(cwd, '..') : cwd;
}

function cleanTitle(row: AuditRow): string {
  if (row.theorem_id === 'T7') return 'T7_Battery: Feasible Fallback Existence';
  if (row.theorem_id === 'T11') return 'T11: ORIUS Runtime-Assurance Contract';
  return row.title.replace(/,\s*T\d+$/u, '');
}

function displayIdFor(row: AuditRow): string {
  if (row.theorem_id === 'T7') return 'T7_Battery';
  return row.theorem_id;
}

function tierFromDefenseTier(surfaceKind: string, defenseTier = ''): TheoremTier {
  if (surfaceKind === 'definition') return 'definition';
  if (defenseTier === 'flagship_defended') return 'flagship';
  if (defenseTier === 'supporting_defended') return 'supporting';
  if (defenseTier === 'draft_non_defended') return 'draft';
  return 'structural';
}

function promotionCandidatePromoted(promotionCandidate?: PromotionScorecardCandidate): boolean {
  return promotionCandidate?.promotion_ready === true;
}

function tierFor(row: AuditRow, promotionCandidate?: PromotionScorecardCandidate): TheoremTier {
  return tierFromDefenseTier(row.surface_kind, row.defense_tier);
}

function roleLabelFromTier(row: AuditRow, tier: TheoremTier): string {
  if (row.surface_kind === 'definition') return 'Definition';
  if (row.theorem_id === 'T7') return 'Battery Instantiation';
  if (row.theorem_id === 'T11') return 'Universal Runtime Contract';
  if (tier === 'draft') return 'Draft / Non-Defended';
  if (tier === 'supporting') return 'Supporting Defended';
  if (tier === 'flagship') return 'Flagship Defended';
  return 'Structural';
}

function domainApplicabilityFor(row: AuditRow, promotionCandidate?: PromotionScorecardCandidate): string {
  if (row.theorem_id === 'T7') {
    return 'Battery instantiation only; AV and Healthcare use bounded runtime fallback lemmas under T11.';
  }
  if (row.theorem_id === 'T11') {
    return 'Universal: any compliant domain inherits the ORIUS runtime-assurance contract.';
  }
  if (row.theorem_id === 'T9' || row.theorem_id === 'T10') {
    return `Scoped flagship lower-bound row; ${THEOREM_PROMOTION_AUTHORITY_LABEL}.`;
  }
  if (row.theorem_id === 'T5') {
    return 'Flagship finite-horizon runtime certificate theorem; validity ends at expiry or invalidating evidence.';
  }
  return 'Current dashboard row follows the active publication audit and its bounded scope note.';
}

function hasAssumptionGate(row: AuditRow): boolean {
  return Boolean(
    row.assumptions_used?.length ||
      row.typed_obligations?.length ||
      (row.surface_kind === 'definition' && row.unresolved_assumptions !== undefined)
  );
}

function gate(key: string, label: string, passed: boolean, detail: string): TheoremPromotionGate {
  return {
    key,
    label,
    status: passed ? 'pass' : 'pending',
    detail,
  };
}

function titleCaseGate(value: string): string {
  return value
    .replace(/^artifact_evidence:/u, 'Artifact Evidence: ')
    .replace(/[_:]+/gu, ' ')
    .split(' ')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function csvGatePasses(value: string): boolean {
  return ['1', 'true', 'yes'].includes(value.trim().toLowerCase());
}

function positiveInteger(value: string | undefined): boolean {
  return Number(value ?? '0') > 0;
}

function gatePassValue(passed: boolean): string {
  return passed ? '1' : '0';
}

function matrixRowToGates(row: PromotionMatrixCsvRow): PromotionGateCsvRow[] {
  const theoremId = row.theorem_id;
  return [
    {
      theorem_id: theoremId,
      gate: 'statement',
      gate_pass: gatePassValue(csvGatePasses(row.statement_complete ?? '')),
      evidence: 'statement_complete=yes',
      blocker: 'statement is incomplete',
    },
    {
      theorem_id: theoremId,
      gate: 'assumptions',
      gate_pass: gatePassValue(csvGatePasses(row.assumptions_complete ?? '')),
      evidence: 'assumptions_complete=yes',
      blocker: 'assumptions are incomplete',
    },
    {
      theorem_id: theoremId,
      gate: 'proof',
      gate_pass: gatePassValue(Boolean(row.proof_file)),
      evidence: row.proof_file ?? '',
      blocker: 'missing proof file',
    },
    {
      theorem_id: theoremId,
      gate: 'code',
      gate_pass: gatePassValue(Boolean(row.code_anchor)),
      evidence: row.code_anchor ?? '',
      blocker: 'missing code anchor',
    },
    {
      theorem_id: theoremId,
      gate: 'tests',
      gate_pass: gatePassValue(positiveInteger(row.tests_count)),
      evidence: `${row.tests_count ?? '0'} registered test(s)`,
      blocker: 'missing registered tests',
    },
    {
      theorem_id: theoremId,
      gate: 'artifact',
      gate_pass: gatePassValue(positiveInteger(row.artifacts_count)),
      evidence: `${row.artifacts_count ?? '0'} registered artifact(s)`,
      blocker: 'missing registered artifacts',
    },
    {
      theorem_id: theoremId,
      gate: 'artifact_hash',
      gate_pass: gatePassValue(csvGatePasses(row.artifact_hashes_complete ?? '')),
      evidence: 'artifact_hashes_complete=yes',
      blocker: 'artifact hashes are incomplete',
    },
    {
      theorem_id: theoremId,
      gate: 'claim_boundary',
      gate_pass: gatePassValue(Boolean(row.claim_boundary)),
      evidence: row.claim_boundary ?? '',
      blocker: 'missing claim-boundary note',
    },
    {
      theorem_id: theoremId,
      gate: 'promotion_ready',
      gate_pass: gatePassValue(csvGatePasses(row.promotion_ready ?? '')),
      evidence: PROMOTION_SCORECARD_SOURCE,
      blocker: 'promotion matrix does not mark this row ready',
    },
  ];
}

function promotionGatesFromCsv(rows: PromotionGateCsvRow[]): TheoremPromotionGate[] {
  return rows.map((row) => {
    const passed = csvGatePasses(row.gate_pass);
    return gate(
      row.gate,
      titleCaseGate(row.gate),
      passed,
      passed ? row.evidence || 'Registered as passing in theorem_promotion_gates.csv' : row.blocker
    );
  });
}

function promotionGatesFor(row: AuditRow): TheoremPromotionGate[] {
  return [
    gate('statement', 'Statement', Boolean(row.statement_location), row.statement_location || 'Missing statement anchor'),
    gate('assumptions', 'Assumptions', hasAssumptionGate(row), 'Assumptions, typed obligations, or explicit definition scope are registered'),
    gate('proof', 'Proof', Boolean(row.proof_location), row.proof_location || 'Missing proof anchor'),
    gate('code', 'Code Anchor', Boolean(row.code_anchors?.length), `${row.code_anchors?.length ?? 0} code anchor(s)`),
    gate('tests', 'Tests', Boolean(row.test_anchors?.length), `${row.test_anchors?.length ?? 0} test anchor(s)`),
    gate('artifacts', 'Artifact Evidence', Boolean(row.generator_targets?.length), `${row.generator_targets?.length ?? 0} generated target(s)`),
    gate('domains', 'Domain Applicability Matrix', true, domainApplicabilityFor(row)),
  ];
}

function missingPromotionPackageGates(row: AuditRow): TheoremPromotionGate[] {
  return [
    gate(
      'promotion_package_missing',
      'Promotion Package Missing',
      false,
      `Missing generated promotion package rows or scorecard candidate for ${row.theorem_id}; promotion status must not fall back to audit-only data.`
    ),
  ];
}

function chapterFor(row: AuditRow): string {
  const location = row.statement_location || row.proof_location;
  const chapterMatch = location.match(/ch(\d+)/u);
  if (chapterMatch) return `Ch. ${Number(chapterMatch[1])}`;
  if (location.includes('chapters_merged/ch04')) return 'Ch. 16-20';
  return 'Publication audit';
}

function statementFor(row: AuditRow, promotionCandidate?: PromotionScorecardCandidate): string {
  if (row.theorem_id === 'T11') {
    return 'Any compliant domain inherits the ORIUS runtime-assurance contract through the typed adapter obligations.';
  }
  if (row.theorem_id === 'T9' || row.theorem_id === 'T10') {
    return `${row.scope_note} ${domainApplicabilityFor(row, promotionCandidate)}`;
  }
  return row.scope_note;
}

function rowToDashboard(
  row: AuditRow,
  promotionRows: PromotionGateCsvRow[] = [],
  promotionPackageAvailable = true,
  promotionCandidate?: PromotionScorecardCandidate
): TheoremDashboardRow {
  const anchors = row.code_anchors ?? [];
  const tests = row.test_anchors ?? [];
  const promotionPackageMissing =
    PROMOTION_PACKAGE_REQUIRED_IDS.has(row.theorem_id) &&
    (!promotionPackageAvailable || promotionRows.length === 0 || !promotionCandidate);
  const promotionGates = promotionPackageMissing
    ? missingPromotionPackageGates(row)
    : promotionRows.length
      ? promotionGatesFromCsv(promotionRows)
      : promotionGatesFor(row);
  const theoremTier = tierFor(row, promotionCandidate);
  const domainApplicability = domainApplicabilityFor(row, promotionCandidate);
  return {
    id: row.theorem_id,
    displayId: displayIdFor(row),
    name: cleanTitle(row),
    tier: theoremTier,
    roleLabel: roleLabelFromTier(row, theoremTier),
    chapter: chapterFor(row),
    assumptions: [...(row.assumptions_used ?? []), ...(row.typed_obligations ?? []), ...(row.unresolved_assumptions ?? [])],
    statement: statementFor(row, promotionCandidate),
    formal: `${row.statement_location} | proof: ${row.proof_location}`,
    significance: promotionCandidate?.remediation_detail || row.remediation_detail || promotionCandidate?.weakest_step || row.weakest_step,
    batteryInstantiation:
      row.theorem_id === 'T7'
        ? 'T7_Battery / C7_BatteryFallback is the battery piecewise hold-or-safe-landing instantiation, not a cross-domain theorem.'
        : row.code_correspondence_detail || domainApplicability,
    proofMethod: `${row.proof_tier}; ${row.rigor_rating}; code=${row.code_correspondence}; anchors=${anchors.length}; tests=${tests.length}`,
    sourceLocation: row.statement_location,
    domainApplicability,
    promotionGates,
    sortOrder: SORT_ORDER.get(row.theorem_id) ?? 999,
  };
}

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = '';
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === ',' && !quoted) {
      values.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

async function loadCsvFallback(csvPath: string): Promise<AuditPayload> {
  const raw = await fs.readFile(csvPath, 'utf-8');
  const [headerLine, ...lines] = raw.trim().split(/\r?\n/u);
  const headers = parseCsvLine(headerLine);
  const rows: AuditRow[] = lines.map((line) => {
    const values = parseCsvLine(line);
    const record = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
    return {
      theorem_id: record.theorem_id || record.id,
      title: record.title,
      surface_kind: record.surface_kind || record.kind || 'theorem',
      defense_tier: 'draft_non_defended',
      proof_tier: 'inventory',
      program_role: record.program || 'generated_inventory',
      scope_note: 'Loaded from reports/publication/theorem_surface_register.csv inventory fallback.',
      statement_location: [record.source_file, record.line].filter(Boolean).join(':'),
      proof_location: [record.source_file, record.line].filter(Boolean).join(':'),
      assumptions_used: [],
      typed_obligations: [],
      unresolved_assumptions: [],
      weakest_step: 'CSV fallback lacks full active-audit proof metadata.',
      rigor_rating: 'inventory',
      code_correspondence: 'partial',
      generator_targets: ['theorem_surface_register'],
      code_anchors: [],
      test_anchors: [],
    };
  });
  return { theorems: rows };
}

function parseCsvRecords<T extends Record<string, string>>(raw: string): T[] {
  const [headerLine, ...lines] = raw.trim().split(/\r?\n/u);
  if (!headerLine) return [];
  const headers = parseCsvLine(headerLine);
  return lines
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      const values = parseCsvLine(line);
      return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ''])) as T;
    });
}

async function loadPromotionGateRows(repoRoot: string): Promise<PromotionGateLoadResult> {
  const gatePath = path.join(repoRoot, PROMOTION_SCORECARD_SOURCE);
  const legacyScorecardPath = path.join(repoRoot, 'reports', 'publication', 'theorem_promotion_scorecard.json');
  const rowsByTheorem = new Map<string, PromotionGateCsvRow[]>();
  const scorecard: PromotionScorecardPayload = existsSync(legacyScorecardPath)
    ? JSON.parse(await fs.readFile(legacyScorecardPath, 'utf-8'))
    : {};
  if (existsSync(gatePath)) {
    const rows = parseCsvRecords<PromotionMatrixCsvRow>(await fs.readFile(gatePath, 'utf-8'));
    for (const row of rows) {
      rowsByTheorem.set(row.theorem_id, matrixRowToGates(row));
    }
  }
  const scorecardAvailable = existsSync(gatePath);
  return {
    rowsByTheorem,
    sourceAvailable: existsSync(gatePath) && scorecardAvailable,
    scorecardAvailable,
    scorecard,
  };
}

export async function loadTheoremDashboardData(): Promise<TheoremDashboardData> {
  const repoRoot = resolveRepoRoot();
  const auditPath = path.join(repoRoot, 'reports', 'publication', 'active_theorem_audit.json');
  const registerPath = path.join(repoRoot, 'reports', 'publication', 'theorem_surface_register.csv');
  const promotionGateRows = await loadPromotionGateRows(repoRoot);
  const payload: AuditPayload = existsSync(auditPath)
    ? JSON.parse(await fs.readFile(auditPath, 'utf-8'))
    : await loadCsvFallback(registerPath);

  const theorems = payload.theorems
    .filter((row) => SORT_ORDER.has(row.theorem_id))
    .map((row) =>
      rowToDashboard(
        row,
        promotionGateRows.rowsByTheorem.get(row.theorem_id),
        promotionGateRows.sourceAvailable,
        promotionGateRows.scorecard.candidates?.[row.theorem_id]
      )
    )
    .sort((left, right) => left.sortOrder - right.sortOrder);

  const stat = existsSync(auditPath) ? await fs.stat(auditPath) : null;
  const summary: TheoremDashboardSummary = {
    flagshipDefended: theorems.filter((row) => row.tier === 'flagship').length,
    supportingDefended: theorems.filter((row) => row.tier === 'supporting').length,
    draftNonDefended: theorems.filter((row) => row.tier === 'draft').length,
    definitionIds: theorems.filter((row) => row.tier === 'definition').map((row) => row.displayId),
    sourcePath: existsSync(auditPath)
      ? `reports/publication/active_theorem_audit.json; ${THEOREM_PROMOTION_AUTHORITY_LABEL}`
      : 'reports/publication/theorem_surface_register.csv',
    sourceUpdatedAt: stat ? stat.mtime.toISOString() : null,
  };

  return {
    theorems,
    summary,
    promotionPolicy: PROMOTION_POLICY,
  };
}
