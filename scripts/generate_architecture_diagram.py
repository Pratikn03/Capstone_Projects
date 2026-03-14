"""
Generate a publication-grade, full-page GridPulse / DC³S architecture diagram.

Layout: three grouped swim-lane bands (Offline, Online, Operations) stacked
vertically with clear cross-lane data-flow arrows.  Designed to fill one
printed page at A4 / letter size with 10 pt two-column article margins.

Output: PNG (300 DPI) + SVG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Colour palette ──────────────────────────────────────────────────────────
BG        = "#FAFCFE"
TEXT      = "#1A2A3A"
MUTED     = "#5A6A7A"
LINE      = "#2B4A6B"
LANE_EDGE = "#C5D3E2"

# Band/pill fills
OFFLINE_FILL  = "#E8F0FA"
ONLINE_FILL   = "#EDF8EF"
OPS_FILL      = "#F5EFF8"

# Node fills per family
DATA_FC,  DATA_EC  = "#DCE8F4", "#5A80A5"
MODEL_FC, MODEL_EC = "#D7E9FA", "#4478AB"
UQ_FC,    UQ_EC    = "#D0E3F7", "#3B6EA3"
SHIELD_FC, SHIELD_EC = "#FFE5A8", "#B87600"
OPT_FC,   OPT_EC   = "#E6D9F2", "#6A4A8E"
AUDIT_FC, AUDIT_EC  = "#D6F0DE", "#358755"
IOT_FC,   IOT_EC   = "#FFF0CC", "#BD8B1A"
MON_FC,   MON_EC   = "#E2D8EF", "#7050A0"
SERVE_FC, SERVE_EC  = "#D3ECF0", "#2F7F90"
PLAIN_FC, PLAIN_EC  = "#FFFFFF", "#8296AA"
SHADOW_FC = "#D0D8E2"
SHADOW_A  = 0.25


# ── Drawing helpers ─────────────────────────────────────────────────────────
def _band(ax, x, y, w, h, title, fc, *, fs=11):
    """A thin coloured band header spanning the full width."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                       lw=0.8, ec=LANE_EDGE, fc=fc, zorder=0)
    ax.add_patch(p)
    ax.text(x + 0.012, y + h / 2, title, ha="left", va="center",
            fontsize=fs, fontweight="bold", color=TEXT, zorder=5)


def _box(ax, x, y, w, h, lines, *, fc=PLAIN_FC, ec=PLAIN_EC, fs=7, bold_first=False):
    """Draw a rounded box with multi-line text.  Returns (x, y, w, h)."""
    # shadow
    ax.add_patch(FancyBboxPatch((x+0.003, y-0.003), w, h,
                 boxstyle="round,pad=0.008,rounding_size=0.008",
                 lw=0, ec="none", fc=SHADOW_FC, alpha=SHADOW_A, zorder=1))
    # box
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.008,rounding_size=0.008",
                 lw=1.0, ec=ec, fc=fc, zorder=2))
    if isinstance(lines, str):
        lines = lines.split("\n")
    n = len(lines)
    for i, ln in enumerate(lines):
        ty = y + h / 2 + (n - 1) / 2 * 0.013 - i * 0.013
        fw = "bold" if (bold_first and i == 0) else "normal"
        ax.text(x + w / 2, ty, ln, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=TEXT, zorder=3,
                clip_on=True)
    return (x, y, w, h)


def _cx(b): return b[0] + b[2] / 2
def _cy(b): return b[1] + b[3] / 2
def _top(b): return (_cx(b), b[1] + b[3])
def _bot(b): return (_cx(b), b[1])
def _right(b): return (b[0] + b[2], _cy(b))
def _left(b): return (b[0], _cy(b))


def _arr(ax, src, dst, *, label=None, color=LINE, lw=1.2, ls="-",
         rad=0.0, ms=14, lbl_off=(0, 0), lbl_fs=6.5, zorder=2):
    """Draw an arrow with a large, visible arrowhead."""
    ax.add_patch(FancyArrowPatch(
        src, dst,
        arrowstyle="-|>",
        mutation_scale=ms,
        lw=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
        shrinkA=2,
        shrinkB=2,
    ))
    if label:
        mx = (src[0]+dst[0])/2 + lbl_off[0]
        my = (src[1]+dst[1])/2 + lbl_off[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=lbl_fs, color=MUTED, zorder=4)


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/figures/architecture.png")
    ap.add_argument("--svg", default="reports/figures/architecture.svg")
    ap.add_argument("--paper", default="paper/assets/figures/fig01_architecture")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(18, 13.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Title ───────────────────────────────────────────────────────────
    ax.text(0.50, 0.985,
            "GridPulse / DC³S  —  System Architecture",
            ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
    ax.text(0.50, 0.970,
            "Offline training produces promoted forecasting artifacts  ·  "
            "Online DC³S turns degraded telemetry into safe dispatch  ·  "
            "Operations feed monitoring, governance, and serving",
            ha="center", va="center", fontsize=8.5, color=MUTED)

    # ====================================================================
    #  BAND A — OFFLINE  (top third)
    # ====================================================================
    A_top = 0.950
    _band(ax, 0.02, A_top, 0.96, 0.022,
          "OFFLINE PLANE  —  data → features → train → calibrate → promote", OFFLINE_FILL)

    # Row A1: Data Sources
    bw = 0.115;  bh = 0.045;  y1 = A_top - 0.065
    a1 = _box(ax, 0.035, y1, bw, bh,
              ["OPSD (Germany)", "load · wind · solar", "price (€/MWh)"],
              fc=DATA_FC, ec=DATA_EC, bold_first=True)
    a2 = _box(ax, 0.165, y1, bw, bh,
              ["EIA-930 (USA)", "MISO · ERCOT · PJM", "demand · gen mix"],
              fc=DATA_FC, ec=DATA_EC, bold_first=True)
    a3 = _box(ax, 0.295, y1, bw, bh,
              ["Open-Meteo", "temp · wind · cloud", "solar radiation"],
              fc=DATA_FC, ec=DATA_EC, bold_first=True)
    a4 = _box(ax, 0.425, y1, 0.10, bh,
              ["Carbon factors", "DE · MISO", "ERCOT · PJM"],
              fc=DATA_FC, ec=DATA_EC, bold_first=True)
    a5 = _box(ax, 0.545, y1, 0.13, bh,
              ["Validation & ingestion", "schema · range · cadence", "impute → hourly parquet"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True)
    a6 = _box(ax, 0.695, y1, 0.14, bh,
              ["Feature engineering", "temporal · lags · rolling", "cyclical · weather (98/118)"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True)
    a7 = _box(ax, 0.855, y1, 0.115, bh,
              ["Chrono split", "70 / 15 / 15", "24 h gap"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True)

    # Data flow: sources → validation → features → split
    _arr(ax, _right(a1), _left(a2), ms=12)
    _arr(ax, _right(a2), _left(a3), ms=12)
    _arr(ax, _right(a3), _left(a4), ms=12)
    _arr(ax, _right(a4), _left(a5), ms=12)
    _arr(ax, _right(a5), _left(a6), ms=12)
    _arr(ax, _right(a6), _left(a7), ms=12)

    # Row A2: Forecasting models
    y2 = y1 - 0.063
    mbw = 0.090;  mbh = 0.045
    m1 = _box(ax, 0.035, y2, mbw, mbh,
              ["LightGBM [PROD]", "1 000 trees · d=12", "lr 0.03 · 256 lv"],
              fc=MODEL_FC, ec=MODEL_EC, bold_first=True, fs=6.5)
    m2 = _box(ax, 0.135, y2, mbw, mbh,
              ["LSTM", "3 layers · 256 hid", "dropout 0.3"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    m3 = _box(ax, 0.235, y2, mbw, mbh,
              ["TCN", "4 chan · kernel 5", "dropout 0.3"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    m4 = _box(ax, 0.335, y2, mbw, mbh,
              ["N-BEATS", "stacks · blocks", "lookback 168"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    m5 = _box(ax, 0.435, y2, mbw, mbh,
              ["TFT", "attention · gating", "variable selection"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    m6 = _box(ax, 0.535, y2, mbw, mbh,
              ["PatchTST", "patch len · heads", "channel-indep"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    m_eval = _box(ax, 0.645, y2, 0.10, mbh,
                  ["Walk-forward eval", "sMAPE · RMSE", "MAE · R²"],
                  fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    m_art  = _box(ax, 0.760, y2, 0.10, mbh,
                  ["Model artifacts", ".pkl (GBM)", ".pt (DL)"],
                  fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    m_reg  = _box(ax, 0.875, y2, 0.095, mbh,
                  ["Registry", "promote()", "atomic copy"],
                  fc=AUDIT_FC, ec=AUDIT_EC, bold_first=True, fs=6.5)

    for a, b in [(m1, m2), (m2, m3), (m3, m4), (m4, m5), (m5, m6)]:
        _arr(ax, _right(a), _left(b), ms=10)
    _arr(ax, _right(m6), _left(m_eval), ms=12)
    _arr(ax, _right(m_eval), _left(m_art), ms=12)
    _arr(ax, _right(m_art), _left(m_reg), ms=12, label="lock",
         lbl_off=(0, 0.012))

    # Vertical: split output → model artifacts
    _arr(ax, (_cx(a7), a7[1]), (_cx(m_art), m_art[1]+m_art[3]),
         lw=0.9, color=MUTED, label="train / val / test",
         lbl_off=(0.048, 0))

    # Row A3: UQ pipeline
    y3 = y2 - 0.058
    uw = 0.125; uh = 0.042
    u1 = _box(ax, 0.035, y3, uw, uh,
              ["CQR calibration", "nonconformity scores", "α=0.10 → 90% cov"],
              fc=UQ_FC, ec=UQ_EC, bold_first=True, fs=6.5)
    u2 = _box(ax, 0.175, y3, 0.10, uh,
              ["FACI adaptation", "online coverage", "tracking"],
              fc=UQ_FC, ec=UQ_EC, bold_first=True, fs=6.5)
    u3 = _box(ax, 0.290, y3, 0.11, uh,
              ["Per-horizon PICP", "MPIW metrics", "h₁ … h₂₄"],
              fc=UQ_FC, ec=UQ_EC, bold_first=True, fs=6.5)
    u4 = _box(ax, 0.415, y3, uw, uh,
              ["RAC-Cert inflation", "reliability → width", "conditional conserv."],
              fc=SHIELD_FC, ec=SHIELD_EC, bold_first=True, fs=6.5)
    u5 = _box(ax, 0.555, y3, uw, uh,
              ["Mondrian binning", "group-conditional", "coverage audit"],
              fc=UQ_FC, ec=UQ_EC, bold_first=True, fs=6.5)
    u6 = _box(ax, 0.700, y3, 0.13, uh,
              ["Interval artifacts", "*_conformal.json", "release-scoped"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)

    _arr(ax, _right(u1), _left(u2), ms=12)
    _arr(ax, _right(u2), _left(u3), ms=12)
    _arr(ax, _right(u3), _left(u4), ms=12)
    _arr(ax, _right(u4), _left(u5), ms=12)
    _arr(ax, _right(u5), _left(u6), ms=12)

    # Vertical: model forecasts → CQR calibration
    _arr(ax, (_cx(m1), m1[1]), (_cx(u1), u1[1]+u1[3]),
         lw=0.9, color=MODEL_EC, ms=12, label="forecasts",
         lbl_off=(0.035, 0))

    # Vertical: registry version → interval artifacts
    _arr(ax, (_cx(m_reg), m_reg[1]), (_cx(u6), u6[1]+u6[3]),
         lw=0.9, color=AUDIT_EC, ms=12, label="version",
         lbl_off=(0.035, 0))

    # ====================================================================
    #  BAND B — ONLINE  (middle third)
    # ====================================================================
    B_top = y3 - 0.030
    _band(ax, 0.02, B_top, 0.96, 0.022,
          "ONLINE PLANE  —  optimise → shield → dispatch → certify", ONLINE_FILL)

    # Row B1: Optimisation
    yo1 = B_top - 0.060
    ow = 0.13;  oh = 0.042
    o1 = _box(ax, 0.035, yo1, ow, oh,
              ["Scenario generation", "lower / upper bounds", "from CQR intervals"],
              fc=OPT_FC, ec=OPT_EC, bold_first=True, fs=6.5)
    o2 = _box(ax, 0.185, yo1, ow, oh,
              ["DRO formulation", "min z + λ_deg Σ(P)", "energy bal · SoC dyn"],
              fc=OPT_FC, ec=OPT_EC, bold_first=True, fs=6.5)
    o3 = _box(ax, 0.335, yo1, 0.10, oh,
              ["HiGHS solver", "LP → optimal", "schedule"],
              fc=OPT_FC, ec=OPT_EC, bold_first=True, fs=6.5)
    o4 = _box(ax, 0.455, yo1, 0.10, oh,
              ["Baselines", "grid-only", "naive battery"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    o5 = _box(ax, 0.575, yo1, 0.12, oh,
              ["Impact evaluation", "cost↓ · carbon↓", "peak shaving↓"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    o6 = _box(ax, 0.715, yo1, 0.12, oh,
              ["Dispatch plans", "EVPI · VSS", "*.json"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)

    _arr(ax, _right(o1), _left(o2), ms=12)
    _arr(ax, _right(o2), _left(o3), ms=12)
    _arr(ax, _right(o3), _left(o4), ms=12)
    _arr(ax, _right(o4), _left(o5), ms=12)
    _arr(ax, _right(o5), _left(o6), ms=12)

    # Vertical: CQR intervals → scenario generation
    _arr(ax, _bot(u1), _top(o1), lw=1.1, color=UQ_EC, ms=14,
         label="intervals", lbl_off=(0.04, 0))

    # Vertical: RAC-Cert inflation → DRO formulation
    _arr(ax, _bot(u4), _top(o2), lw=1.1, color=SHIELD_EC, ms=14,
         label="inflation\nschedule", lbl_off=(0.04, 0))

    # Row B2: DC³S shield (highlighted row)
    ys1 = yo1 - 0.070
    sw = 0.095;  sh = 0.052
    s1 = _box(ax, 0.035, ys1, sw, sh,
              ["① Observe", "telemetry", "load · ren · SoC"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=7)
    s2 = _box(ax, 0.145, ys1, sw, sh,
              ["② Quality", "missing · stale", "spike · reorder"],
              fc=SHIELD_FC, ec=SHIELD_EC, bold_first=True, fs=7)
    s3 = _box(ax, 0.255, ys1, sw, sh,
              ["③ Drift", "KS stat · EWM", "online update"],
              fc=SHIELD_FC, ec=SHIELD_EC, bold_first=True, fs=7)
    s4 = _box(ax, 0.365, ys1, sw, sh,
              ["④ Inflate", "RAC-Cert", "widen intervals"],
              fc=SHIELD_FC, ec=SHIELD_EC, bold_first=True, fs=7)
    s5 = _box(ax, 0.475, ys1, sw, sh,
              ["⑤ Solve", "DRO dispatch", "widened set"],
              fc=OPT_FC, ec=OPT_EC, bold_first=True, fs=7)
    s6 = _box(ax, 0.585, ys1, sw, sh,
              ["⑥ Repair", "projection onto", "feasible set"],
              fc=SHIELD_FC, ec=SHIELD_EC, bold_first=True, fs=7)
    s7 = _box(ax, 0.695, ys1, sw, sh,
              ["⑦ Post-check", "SoC · rate · FTIT", "guarantee_checks"],
              fc=SHIELD_FC, ec=SHIELD_EC, bold_first=True, fs=7)
    s8 = _box(ax, 0.805, ys1, 0.08, sh,
              ["⑧ Certify", "hash chain", "audit trail"],
              fc=AUDIT_FC, ec=AUDIT_EC, bold_first=True, fs=7)
    s9 = _box(ax, 0.900, ys1, 0.075, sh,
              ["Safe dispatch", "charge / hold", "ACK / NACK"],
              fc=AUDIT_FC, ec=AUDIT_EC, bold_first=True, fs=7)

    shield_steps = [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    for a, b in zip(shield_steps[:-1], shield_steps[1:]):
        _arr(ax, _right(a), _left(b), ms=12)

    # Shield label
    ax.text(0.035, ys1 + sh + 0.008,
            "DC³S safety shield  (< 0.04 ms P95 per step)",
            fontsize=8, fontweight="bold", color=SHIELD_EC, va="bottom")

    # Vertical: proposed action from optimizer → shield solve
    _arr(ax, _bot(o3), _top(s5), lw=1.2, color=OPT_EC, ms=14,
         label="proposed\naction", lbl_off=(0.04, 0))

    # Vertical: dispatch plans → certify
    _arr(ax, _bot(o6), _top(s8), lw=0.9, color=MUTED, ms=12,
         label="plan ref", lbl_off=(0.032, 0))

    # ====================================================================
    #  BAND C — OPERATIONS  (bottom third)
    # ====================================================================
    C_top = ys1 - 0.035
    _band(ax, 0.02, C_top, 0.96, 0.022,
          "OPERATIONS PLANE  —  IoT · streaming · monitoring · governance · serving", OPS_FILL)

    # Row C1: IoT + Streaming
    yi = C_top - 0.058
    iw = 0.12; ih = 0.042
    i1 = _box(ax, 0.035, yi, iw, ih,
              ["Edge agent", "shadow_mode=true", "applied=false"],
              fc=IOT_FC, ec=IOT_EC, bold_first=True, fs=6.5)
    i2 = _box(ax, 0.170, yi, iw, ih,
              ["Device contract", "cadence 1/h ±120 s", "TTL 30 s · ACK/NACK"],
              fc=IOT_FC, ec=IOT_EC, bold_first=True, fs=6.5)
    i3 = _box(ax, 0.305, yi, iw, ih,
              ["Kafka consumer", "OPSDTelemetryEvent", "Pydantic validation"],
              fc=IOT_FC, ec=IOT_EC, bold_first=True, fs=6.5)
    i4 = _box(ax, 0.440, yi, iw, ih,
              ["Temporal + range", "cadence enforce", "delta outlier checks"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    i5 = _box(ax, 0.575, yi, 0.10, ih,
              ["Checkpoint", "exactly-once", "every 200 msgs"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    i6 = _box(ax, 0.690, yi, 0.11, ih,
              ["DuckDB/Parquet", "validated events", "time-ordered"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)

    for a, b in [(i1, i2), (i2, i3), (i3, i4), (i4, i5), (i5, i6)]:
        _arr(ax, _right(a), _left(b), ms=12)

    # Vertical: safe dispatch → edge agent
    _arr(ax, _bot(s9), (0.938, yi + ih), lw=1.2, color=AUDIT_EC, ms=14,
         label="safe cmd", lbl_off=(0.013, 0.008))

    # Vertical: edge agent → shield observe (telemetry uplink)
    _arr(ax, _top(i1), _bot(s1), lw=0.9, color=IOT_EC, ls="--", ms=12,
         label="telemetry", lbl_off=(-0.035, 0))

    # Row C2: Monitoring + Anomaly + Governance
    ym = yi - 0.058
    mw = 0.105; mh = 0.042
    g1 = _box(ax, 0.035, ym, mw, mh,
              ["Data drift", "KS test · PSI", "per feature"],
              fc=MON_FC, ec=MON_EC, bold_first=True, fs=6.5)
    g2 = _box(ax, 0.155, ym, mw, mh,
              ["Model drift", "rolling RMSE", ">10% → alert"],
              fc=MON_FC, ec=MON_EC, bold_first=True, fs=6.5)
    g3 = _box(ax, 0.275, ym, mw, mh,
              ["DC³S health", "intervention rate", "inflation P95"],
              fc=SHIELD_FC, ec=SHIELD_EC, bold_first=True, fs=6.5)
    g4 = _box(ax, 0.395, ym, mw, mh,
              ["Anomaly detect", "residual z-score", "isolation forest"],
              fc=MON_FC, ec=MON_EC, bold_first=True, fs=6.5)
    g5 = _box(ax, 0.515, ym, mw, mh,
              ["Retrain triggers", "weekly · drift", "DC³S-based"],
              fc=MON_FC, ec=MON_EC, bold_first=True, fs=6.5)
    g6 = _box(ax, 0.635, ym, 0.10, mh,
              ["CPSBench", "5 scenarios", "5-seed eval"],
              fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    g7 = _box(ax, 0.750, ym, 0.10, mh,
              ["Governance", "release manifest", "evidence map"],
              fc=AUDIT_FC, ec=AUDIT_EC, bold_first=True, fs=6.5)

    for a, b in [(g1, g2), (g2, g3), (g3, g4), (g4, g5), (g5, g6), (g6, g7)]:
        _arr(ax, _right(a), _left(b), ms=12)

    # Vertical: certificates → DC³S health
    _arr(ax, _bot(s8), (_cx(g3), g3[1]+g3[3]), lw=1.0, color=SHIELD_EC, ms=14,
         label="health", lbl_off=(0.035, 0))

    # Vertical: DuckDB events → anomaly detection
    _arr(ax, _bot(i6), (_cx(g4), g4[1]+g4[3]), lw=0.9, color=MUTED, ms=12,
         label="events", lbl_off=(0.035, 0))

    # Row C3: Serving
    ya = ym - 0.058
    aw = 0.11; ah = 0.042
    sv1 = _box(ax, 0.035, ya, aw, ah,
               ["FastAPI", "/forecast · /optimize", "/dc3s/step · /monitor"],
               fc=SERVE_FC, ec=SERVE_EC, bold_first=True, fs=6.5)
    sv2 = _box(ax, 0.160, ya, 0.10, ah,
               ["PostgreSQL", "+ Redis cache", "model versions"],
               fc=SERVE_FC, ec=SERVE_EC, bold_first=True, fs=6.5)
    sv3 = _box(ax, 0.275, ya, aw, ah,
               ["Next.js 15", "8 pages · overview", "dispatch · carbon"],
               fc=SERVE_FC, ec=SERVE_EC, bold_first=True, fs=6.5)
    sv4 = _box(ax, 0.400, ya, 0.10, ah,
               ["Prometheus", "+ Grafana", "alerts · panels"],
               fc=SERVE_FC, ec=SERVE_EC, bold_first=True, fs=6.5)
    sv5 = _box(ax, 0.515, ya, aw, ah,
               ["Docker Compose", "API · streaming", "full-stack"],
               fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    sv6 = _box(ax, 0.640, ya, 0.10, ah,
               ["K8s / ECS", "health checks", "ConfigMap"],
               fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)
    sv7 = _box(ax, 0.755, ya, 0.10, ah,
               ["Reports", "figures · tables", "publication/"],
               fc=PLAIN_FC, ec=PLAIN_EC, bold_first=True, fs=6.5)

    for a, b in [(sv1, sv2), (sv2, sv3), (sv3, sv4), (sv4, sv5), (sv5, sv6), (sv6, sv7)]:
        _arr(ax, _right(a), _left(b), ms=12)

    # Vertical: metrics → Prometheus
    _arr(ax, _bot(g4), (_cx(sv4), sv4[1]+sv4[3]), lw=1.0, color=MON_EC, ms=14,
         label="metrics", lbl_off=(0.035, 0))

    # Vertical: governance → reports
    _arr(ax, _bot(g7), (_cx(sv7), sv7[1]+sv7[3]), lw=0.9, color=AUDIT_EC, ms=12,
         label="evidence", lbl_off=(0.035, 0))

    # ====================================================================
    #  Cross-band arrows (dashed for feedback / online)
    # ====================================================================
    # Retrain trigger → models (feedback loop)
    _arr(ax, (0.035, _cy(g5)), (0.035, m1[1]), lw=1.0, ls="--",
         color=MON_EC, ms=14, label="retrain\ntrigger", lbl_off=(0.035, 0))

    # API /dc3s/step → shield observe (online call)
    _arr(ax, (_cx(sv1), sv1[1]+sv1[3]), (_cx(s1), s1[1]), lw=1.0, ls="--",
         color=SERVE_EC, ms=14, label="/dc3s/step", lbl_off=(-0.033, 0))

    # Validated events → governance
    _arr(ax, _bot(i6), (_cx(g7), g7[1]+g7[3]), lw=0.9, ls="--",
         color=MUTED, ms=12, label="runtime\nevidence", lbl_off=(0.045, 0))

    # Registry → governance (model versions feed governance)
    _arr(ax, (_cx(m_reg), m_reg[1]), (_cx(g7)+0.02, g7[1]+g7[3]),
         lw=0.9, ls="--", color=AUDIT_EC, ms=12,
         label="version\ntrail", lbl_off=(0.035, 0))

    # CPSBench → registry (verification gates release)
    _arr(ax, (_cx(g6), g6[1]+g6[3]), (_cx(m_reg)-0.01, m_reg[1]),
         lw=0.9, ls="--", color=MUTED, ms=12,
         label="verification\ngate", lbl_off=(-0.048, 0))

    # ====================================================================
    #  Legend bar
    # ====================================================================
    leg_y = ya - 0.035
    ax.plot([0.03, 0.97], [leg_y + 0.018, leg_y + 0.018],
            color=LANE_EDGE, lw=0.5, zorder=0)
    items = [
        (0.04, DATA_FC,   DATA_EC,   "Data sources"),
        (0.14, MODEL_FC,  MODEL_EC,  "Forecasting"),
        (0.24, UQ_FC,     UQ_EC,     "Uncertainty / CQR"),
        (0.37, OPT_FC,    OPT_EC,    "Optimisation"),
        (0.49, SHIELD_FC, SHIELD_EC, "DC³S / RAC-Cert"),
        (0.63, AUDIT_FC,  AUDIT_EC,  "Audit / Governance"),
        (0.77, IOT_FC,    IOT_EC,    "IoT / Streaming"),
        (0.87, SERVE_FC,  SERVE_EC,  "Serving"),
    ]
    for lx, fc, ec, lbl in items:
        _box(ax, lx, leg_y, 0.014, 0.014, "", fc=fc, ec=ec, fs=1)
        ax.text(lx + 0.018, leg_y + 0.007, lbl,
                fontsize=7, color=TEXT, va="center")

    ax.text(0.50, leg_y - 0.016,
            "Solid arrows = data flow  ·  Dashed arrows = feedback / online request  ·  "
            "Colour coding matches component family",
            ha="center", va="center", fontsize=7, color=MUTED)

    # ====================================================================
    #  Save
    # ====================================================================
    fig.tight_layout(pad=0.3)
    out_paths = []
    for p in [args.out, args.svg]:
        out = Path(p)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        out_paths.append(out)

    paper_base = Path(args.paper)
    paper_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".svg"):
        pf = paper_base.with_suffix(ext)
        fig.savefig(pf, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        out_paths.append(pf)

    plt.close(fig)
    for p in out_paths:
        print(f"  ✓ {p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
