"""
Generate a crystal-clear one-page ORIUS / DC³S architecture diagram.

Layout: three swim-lane bands (Offline → Online DC³S → Operations) with:
  - Offline:    data → features → models → CQR calibration → locked artifacts
  - Online:     DC³S 5-stage horseshoe pipeline + 6 domain adapters + safety guarantee
  - Operations: IoT → monitoring → CPSBench → FastAPI → governance

Redesigned for readability: 16×11 in canvas, 9pt+ fonts, ~25 nodes, DC³S centrepiece.

Output: PNG (300 DPI) + SVG at paper/assets/figures/fig01_architecture.*
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Colour palette ────────────────────────────────────────────────────────────
BG             = "#F8FAFB"
TEXT           = "#1A2A3A"
MUTED          = "#4B5563"
LINE           = "#334155"
LINE_STAGE     = "#92400E"
LANE_EDGE      = "#BFD0E8"

# Lane fills
OFFLINE_FILL   = "#EBF4FF"
ONLINE_FILL    = "#FFFBF0"
OPS_FILL       = "#F0F7F1"

# Node families
DATA_FC, DATA_EC       = "#DBEAFE", "#1D4ED8"
MODEL_FC, MODEL_EC     = "#EDE9FE", "#6D28D9"
UQ_FC, UQ_EC           = "#D1FAE5", "#047857"
ARTIFACT_FC, ART_EC    = "#FEF3C7", "#B45309"

STAGE_FC, STAGE_EC     = "#FFF3CD", "#B45309"     # DC³S pipeline stages
CERTIFY_FC, CERT_EC    = "#FCE7F3", "#BE185D"     # Certify / CERTos
INPUT_FC, INPUT_EC     = "#EFF6FF", "#1E40AF"
OUTPUT_FC, OUTPUT_EC   = "#F0FDF4", "#15803D"
DOMAIN_FC, DOMAIN_EC   = "#F1F5F9", "#475569"
GUARANTEE_FC, GUAR_EC  = "#FEF2F2", "#B91C1C"

IOT_FC, IOT_EC         = "#FEF9C3", "#92400E"
MON_FC, MON_EC         = "#E0E7FF", "#3730A3"
BENCH_FC, BENCH_EC     = "#ECFDF5", "#065F46"
SERVE_FC, SERVE_EC     = "#E0F2FE", "#075985"
GOV_FC, GOV_EC         = "#FDF4FF", "#6B21A8"

SHADOW_FC = "#CBD5E1"
SHADOW_A  = 0.18


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cx(b): return b[0] + b[2] / 2
def _cy(b): return b[1] + b[3] / 2
def _top(b): return (_cx(b), b[1] + b[3])
def _bot(b): return (_cx(b), b[1])
def _right(b): return (b[0] + b[2], _cy(b))
def _left(b):  return (b[0], _cy(b))


def _band(ax, x, y, w, h, title, fc, ec, *, fs=10):
    """Coloured lane-header band."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        lw=1.0, ec=ec, fc=fc, zorder=0,
    ))
    ax.text(x + 0.014, y + h / 2, title,
            ha="left", va="center",
            fontsize=fs, fontweight="bold", color=TEXT, zorder=5)


def _box(ax, x, y, w, h, lines, *, fc="#FFF", ec="#94A3B8", fs: float = 8.5,
         bold_first=True, lw=1.3):
    """Rounded box with drop shadow. Returns (x, y, w, h)."""
    ax.add_patch(FancyBboxPatch(           # shadow
        (x + 0.004, y - 0.004), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        lw=0, fc=SHADOW_FC, alpha=SHADOW_A, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(           # box
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        lw=lw, ec=ec, fc=fc, zorder=2,
    ))
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    step = min(0.019, h / max(n, 1.5))
    for i, ln in enumerate(lines):
        ty = y + h / 2 + (n - 1) * step / 2 - i * step
        fw = "bold" if (bold_first and i == 0) else "normal"
        ax.text(x + w / 2, ty, ln,
                ha="center", va="center",
                fontsize=fs, fontweight=fw, color=TEXT,
                zorder=3, clip_on=True)
    return (x, y, w, h)


def _pill(ax, cx, cy, label, *, fc=DOMAIN_FC, ec=DOMAIN_EC, fs: float = 7.8,
          pw=0.100, ph=0.028):
    """Pill-shaped domain badge."""
    ax.add_patch(FancyBboxPatch(
        (cx - pw / 2, cy - ph / 2), pw, ph,
        boxstyle="round,pad=0.004,rounding_size=0.014",
        lw=1.0, ec=ec, fc=fc, zorder=3,
    ))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=TEXT, zorder=4)


def _arr(ax, src, dst, *, label=None, color=LINE, lw=1.5, ls="-",
         rad=0.0, ms=14, lbl_off=(0, 0), lbl_fs=7.5, zorder=4):
    """Arrow with visible arrowhead."""
    ax.add_patch(FancyArrowPatch(
        src, dst,
        arrowstyle="-|>",
        mutation_scale=ms,
        lw=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder, shrinkA=3, shrinkB=3,
    ))
    if label:
        mx = (src[0] + dst[0]) / 2 + lbl_off[0]
        my = (src[1] + dst[1]) / 2 + lbl_off[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=lbl_fs, color=MUTED, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))


def _latency(ax, cx, top_y, txt):
    """Amber latency badge placed just above a box top."""
    ax.text(cx, top_y + 0.006, txt,
            ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#92400E",
            bbox=dict(fc="#FFFBEB", ec="#FCD34D", lw=0.8, pad=1.5,
                      boxstyle="round,pad=0.002,rounding_size=0.003"),
            zorder=6)


# ── Layout constants ──────────────────────────────────────────────────────────
# All y-coordinates are bottom-of-object (matplotlib convention).
# Work top-down:

#  Title block            : handled inline
#  OFFLINE lane header    : y = 0.878, h = 0.020
#  OFFLINE boxes          : y = 0.793, h = 0.074
#  ── gap ──
#  ONLINE  lane header    : y = 0.756, h = 0.020
#  DC³S top row (①②③)    : y = 0.645, h = 0.085
#  DC³S bot row (④⑤)     : y = 0.519, h = 0.085
#  Output box             : y = 0.457, h = 0.043
#  Domain adapter pills   : cy= 0.422
#  ── gap ──
#  OPERATIONS lane header : y = 0.355, h = 0.020
#  OPERATIONS boxes       : y = 0.274, h = 0.068
#  Legend bar             : y = 0.060

OFFLINE_BAND_Y  = 0.878
OFFLINE_BOX_Y   = 0.793
OFFLINE_BOX_H   = 0.074

ONLINE_BAND_Y   = 0.756

TOP_ROW_Y  = 0.645
TOP_ROW_H  = 0.085

BOT_ROW_Y  = 0.519
BOT_ROW_H  = 0.085

OUT_BOX_Y  = 0.457
OUT_BOX_H  = 0.043

DOMAIN_CY  = 0.420

OPS_BAND_Y = 0.355
OPS_BOX_Y  = 0.274
OPS_BOX_H  = 0.068

# DC³S stage dimensions
SW  = 0.183   # stage box width
SH  = TOP_ROW_H
GAP = 0.017   # gap between adjacent stage boxes

# Top row starts at sx0 (leave ~0.135 for input labels on left)
sx0 = 0.148   # ① DETECT
sx1 = sx0 + SW + GAP   # ② CALIBRATE
sx2 = sx1 + SW + GAP   # ③ CONSTRAIN

# Bottom row: ④ aligned under ③, ⑤ aligned under ②
sx3 = sx2   # ④ SHIELD  (same x as ③)
sx4 = sx1   # ⑤ CERTIFY (same x as ②)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",   default="reports/figures/architecture.png")
    ap.add_argument("--svg",   default="reports/figures/architecture.svg")
    ap.add_argument("--paper", default="paper/assets/figures/fig01_architecture")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(0.50, 0.972,
            "ORIUS / DC\u00b3S  \u2014  System Architecture",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=TEXT)
    ax.text(0.50, 0.954,
            "Offline training promotes forecasting artifacts  \u00b7  "
            "Online DC\u00b3S closes the Observation-Action Safety Gap in \u2264 41 \u03bcs  \u00b7  "
            "Operations govern, monitor, and serve",
            ha="center", va="center", fontsize=8.5, color=MUTED)

    # =========================================================================
    # LANE A  \u2014  OFFLINE
    # =========================================================================
    _band(ax, 0.02, OFFLINE_BAND_Y, 0.96, 0.020,
          "OFFLINE PLANE  \u2014  data  \u2192  features  \u2192  train  \u2192  calibrate  \u2192  promote",
          OFFLINE_FILL, DATA_EC)

    BW = 0.174   # offline box width
    BG_off = 0.012  # gap
    xs = [0.030 + i * (BW + BG_off) for i in range(5)]

    a1 = _box(ax, xs[0], OFFLINE_BOX_Y, BW, OFFLINE_BOX_H,
              ["Grid Data", "OPSD (Germany)", "EIA-930 (USA)"],
              fc=DATA_FC, ec=DATA_EC)
    a2 = _box(ax, xs[1], OFFLINE_BOX_Y, BW, OFFLINE_BOX_H,
              ["Feature Engineering", "98 / 118 signals", "lags \u00b7 cyclical \u00b7 weather"],
              fc=DATA_FC, ec=DATA_EC)
    a3 = _box(ax, xs[2], OFFLINE_BOX_Y, BW, OFFLINE_BOX_H,
              ["Model Zoo", "LightGBM \u2605  LSTM  TCN", "TFT  \u00b7  PatchTST  N-BEATS"],
              fc=MODEL_FC, ec=MODEL_EC)
    a4 = _box(ax, xs[3], OFFLINE_BOX_Y, BW, OFFLINE_BOX_H,
              ["CQR Calibration", "\u03b1 = 0.10  \u2192  PICP \u2265 90 %", "RAC-Cert inflation law"],
              fc=UQ_FC, ec=UQ_EC)
    a5 = _box(ax, xs[4], OFFLINE_BOX_Y, BW, OFFLINE_BOX_H,
              ["Locked Artifacts", ".pkl + conformal.json", "atomic promote()"],
              fc=ARTIFACT_FC, ec=ART_EC)

    for a, b in [(a1, a2), (a2, a3), (a3, a4), (a4, a5)]:
        _arr(ax, _right(a), _left(b), lw=1.5, ms=13)

    # =========================================================================
    # LANE B  \u2014  ONLINE  /  DC\u00b3S
    # =========================================================================
    _band(ax, 0.02, ONLINE_BAND_Y, 0.96, 0.020,
          "ONLINE PLANE  \u2014  DC\u00b3S  :  detect \u2192 calibrate \u2192 constrain \u2192 shield \u2192 certify",
          ONLINE_FILL, STAGE_EC)

    # ── Top row: ①②③ ──────────────────────────────────────────────────────
    s1 = _box(ax, sx0, TOP_ROW_Y, SW, SH,
              ["\u2460  DETECT  (OQE)",
               "5 fault types \u2192 w\u209c \u2208 [0.05, 1.0]",
               "dropout \u00b7 stale \u00b7 spike",
               "delay-jitter \u00b7 out-of-order"],
              fc=STAGE_FC, ec=STAGE_EC, fs=8.5, lw=1.7)

    s2 = _box(ax, sx1, TOP_ROW_Y, SW, SH,
              ["\u2461  CALIBRATE  (RUI)",
               "inflate C\u209c: q\u209c \u00d7 (1 + k(1\u2212w\u209c))",
               "drift penalty \u00b7 RAC-Cert",
               "FTIT-RO inflation law"],
              fc=STAGE_FC, ec=STAGE_EC, fs=8.5, lw=1.7)

    s3 = _box(ax, sx2, TOP_ROW_Y, SW, SH,
              ["\u2462  CONSTRAIN  (SAF/FTIT)",
               "tighten A\u209c by margin",
               "m\u209c = q\u209c / (w\u209c + \u03b5)",
               "\u2264 1 \u03bcs"],
              fc=STAGE_FC, ec=STAGE_EC, fs=8.5, lw=1.7)

    _latency(ax, _cx(s1), TOP_ROW_Y + SH, "18 \u03bcs")
    _latency(ax, _cx(s2), TOP_ROW_Y + SH, "10 \u03bcs")

    _arr(ax, _right(s1), _left(s2), lw=2.0, color=LINE_STAGE, ms=15,
         label="w\u209c", lbl_off=(0, 0.012), lbl_fs=8)
    _arr(ax, _right(s2), _left(s3), lw=2.0, color=LINE_STAGE, ms=15,
         label="C\u209c", lbl_off=(0, 0.012), lbl_fs=8)

    # ── Bottom row: ④ SHIELD, ⑤ CERTIFY ──────────────────────────────────
    s4 = _box(ax, sx3, BOT_ROW_Y, SW, SH,
              ["\u2463  SHIELD  (Repair)",
               "L2 project a\u2217 \u2192 A\u209c",
               "domain adapter repair()",
               "intervened \u2208 {True, False}"],
              fc=STAGE_FC, ec=STAGE_EC, fs=8.5, lw=1.7)

    s5 = _box(ax, sx4, BOT_ROW_Y, SW, SH,
              ["\u2464  CERTIFY  (CERTos)",
               "hash-linked certificate",
               "SHA-256 \u00b7 DuckDB audit trail",
               "prev_hash chain"],
              fc=CERTIFY_FC, ec=CERT_EC, fs=8.5, lw=1.7)

    _latency(ax, _cx(s4), BOT_ROW_Y + SH, "2.3 \u03bcs")
    _latency(ax, _cx(s5), BOT_ROW_Y + SH, "< 1 \u03bcs")

    # Horseshoe: ③ bottom → ④ top  (vertical bend)
    _arr(ax, _bot(s3), _top(s4), lw=2.0, color=LINE_STAGE, ms=15,
         label="A\u209c", lbl_off=(0.026, 0), lbl_fs=8)
    # ④ left → ⑤ right  (horizontal, reversed)
    _arr(ax, _left(s4), _right(s5), lw=2.0, color=LINE_STAGE, ms=15,
         label="a\u209c\u02e2\u1d43\u1da0\u1da3", lbl_off=(0, 0.012), lbl_fs=8)

    # ── Input arrows ─────────────────────────────────────────────────────
    in_w, in_h = 0.120, 0.044
    in_x = 0.018

    inp1 = _box(ax, in_x, _cy(s1) - in_h / 2, in_w, in_h,
                ["Raw Telemetry", "CPS plant  z\u209c"],
                fc=INPUT_FC, ec=INPUT_EC, fs=8.0, lw=1.2)
    _arr(ax, _right(inp1), _left(s1), lw=1.6, color=DATA_EC, ms=13)

    inp2 = _box(ax, in_x, _cy(s4) - in_h / 2, in_w, in_h,
                ["Proposed Action", "optimiser  a\u2217"],
                fc=INPUT_FC, ec=INPUT_EC, fs=8.0, lw=1.2)
    _arr(ax, _right(inp2), _left(s4), lw=1.6, color=DATA_EC, ms=13)

    # Offline artifacts → CALIBRATE (dashed: conformal quantile q)
    _arr(ax, _bot(a4), _top(s2),
         lw=1.2, ls="--", color=UQ_EC, ms=11,
         label="q  (conformal)", lbl_off=(0.038, 0), lbl_fs=7.0)

    # ── Output box ───────────────────────────────────────────────────────
    out_box = _box(ax, sx4, OUT_BOX_Y, SW, OUT_BOX_H,
                   ["Safe Action  a\u209c\u02e2\u1d43\u1da0\u1da3  +  Certificate"],
                   fc=OUTPUT_FC, ec=OUTPUT_EC, fs=8.5, lw=1.5, bold_first=False)
    _arr(ax, _bot(s5), _top(out_box),
         lw=2.0, color=OUTPUT_EC, ms=15)

    # ── Domain adapter pill row ───────────────────────────────────────────
    domains = ["Energy", "AV", "Industrial", "Healthcare", "Aerospace", "Navigation"]
    pill_w, pill_h = 0.098, 0.027
    n_d = len(domains)
    pill_total_w = n_d * pill_w + (n_d - 1) * 0.010
    pill_x0 = sx4   # start aligned with ⑤

    ax.text(pill_x0 - 0.006, DOMAIN_CY,
            "Domain Adapters:",
            ha="right", va="center",
            fontsize=7.5, color=MUTED, style="italic")

    for i, dom in enumerate(domains):
        cx = pill_x0 + i * (pill_w + 0.010) + pill_w / 2
        _pill(ax, cx, DOMAIN_CY, dom, pw=pill_w, ph=pill_h)

    # Thin arrow: domain adapters ↔ pipeline (representational)
    _arr(ax, (pill_x0 + pill_total_w / 2, DOMAIN_CY + pill_h / 2),
         (_cx(s5), BOT_ROW_Y),
         lw=1.0, ls="--", color=DOMAIN_EC, ms=10,
         label="DomainAdapter ABC", lbl_off=(0.048, 0), lbl_fs=7.0)

    # ── Safety guarantee box (right side) ────────────────────────────────
    g_x = sx2 + SW + 0.022
    g_y = BOT_ROW_Y - 0.008
    g_w = 0.970 - g_x
    g_h = (TOP_ROW_Y + SH + 0.018) - g_y

    ax.add_patch(FancyBboxPatch(
        (g_x, g_y), g_w, g_h,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        lw=2.0, ec=GUAR_EC, fc=GUARANTEE_FC, zorder=2,
    ))
    gcx = g_x + g_w / 2
    gcy = g_y + g_h / 2
    ax.text(gcx, gcy + 0.062, "Safety Guarantee",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=GUAR_EC, zorder=3)
    ax.text(gcx, gcy + 0.028,
            r"$\mathrm{E}[V] \leq \alpha\,(1-\bar{w})\,T$",
            ha="center", va="center",
            fontsize=12.5, color=TEXT, zorder=3)
    ax.text(gcx, gcy - 0.005,
            "Theorem T3",
            ha="center", va="center",
            fontsize=8, color=MUTED, style="italic", zorder=3)
    ax.text(gcx, gcy - 0.030,
            "V = true-state violations\n"
            r"$\alpha$ = miscoverage  $\bar{w}$ = mean OQE",
            ha="center", va="center",
            fontsize=7.5, color=TEXT, zorder=3)
    ax.text(gcx, gcy - 0.062,
            "Full DC\u00b3S step   P95 \u2264 41 \u03bcs",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=GUAR_EC, zorder=3)

    # =========================================================================
    # LANE C  \u2014  OPERATIONS
    # =========================================================================
    _band(ax, 0.02, OPS_BAND_Y, 0.96, 0.020,
          "OPERATIONS PLANE  \u2014  IoT  \u00b7  monitoring  \u00b7  CPSBench  \u00b7  FastAPI  \u00b7  governance",
          OPS_FILL, IOT_EC)

    OW = 0.172
    OG = 0.013
    oxs = [0.030 + i * (OW + OG) for i in range(5)]

    c1 = _box(ax, oxs[0], OPS_BOX_Y, OW, OPS_BOX_H,
              ["IoT / Edge Agent", "Kafka \u00b7 Pydantic valid.", "1/h cadence \u00b7 ACK/NACK"],
              fc=IOT_FC, ec=IOT_EC)
    c2 = _box(ax, oxs[1], OPS_BOX_Y, OW, OPS_BOX_H,
              ["Monitoring", "drift (KS/PSI) \u00b7 RMSE", "DC\u00b3S health \u00b7 retrain"],
              fc=MON_FC, ec=MON_EC)
    c3 = _box(ax, oxs[2], OPS_BOX_Y, OW, OPS_BOX_H,
              ["CPSBench", "dual-trajectory eval", "5-seed \u00b7 TSVR gate"],
              fc=BENCH_FC, ec=BENCH_EC)
    c4 = _box(ax, oxs[3], OPS_BOX_Y, OW, OPS_BOX_H,
              ["FastAPI Serving", "/dc3s/step  /forecast", "/optimize  /monitor"],
              fc=SERVE_FC, ec=SERVE_EC)
    c5 = _box(ax, oxs[4], OPS_BOX_Y, OW, OPS_BOX_H,
              ["Governance", "release manifest", "evidence map \u00b7 versions"],
              fc=GOV_FC, ec=GOV_EC)

    for a, b in [(c1, c2), (c2, c3), (c3, c4), (c4, c5)]:
        _arr(ax, _right(a), _left(b), lw=1.5, ms=13)

    # Cross-lane: safe action → IoT edge (dispatch)
    _arr(ax, _bot(out_box), _top(c1),
         lw=1.6, color=OUTPUT_EC, ls="--", ms=13,
         label="safe cmd", lbl_off=(-0.050, 0), lbl_fs=7.5)

    # Certificates → Monitoring (health stream)
    _arr(ax, (_cx(s5), BOT_ROW_Y), (_cx(c2), OPS_BOX_Y + OPS_BOX_H),
         lw=1.1, color=MON_EC, ls="--", ms=11,
         label="certs", lbl_off=(0.030, 0), lbl_fs=7.0)

    # CPSBench → Model Zoo (verification gate)
    _arr(ax, (_cx(c3), OPS_BOX_Y + OPS_BOX_H),
         (_cx(a3), OFFLINE_BOX_Y),
         lw=1.0, color=BENCH_EC, ls="--", ms=11,
         label="verify gate", lbl_off=(0.040, 0), lbl_fs=7.0)

    # =========================================================================
    # Legend bar
    # =========================================================================
    leg_y = 0.058
    ax.plot([0.03, 0.97], [leg_y + 0.025, leg_y + 0.025],
            color=LANE_EDGE, lw=0.6, zorder=0)

    items = [
        (0.03,  DATA_FC,     DATA_EC,     "Grid Data"),
        (0.15,  MODEL_FC,    MODEL_EC,    "Forecasting"),
        (0.27,  UQ_FC,       UQ_EC,       "UQ / CQR"),
        (0.38,  ARTIFACT_FC, ART_EC,      "Artifacts"),
        (0.50,  STAGE_FC,    STAGE_EC,    "DC\u00b3S Shield"),
        (0.62,  CERTIFY_FC,  CERT_EC,     "Certify / CERTos"),
        (0.74,  MON_FC,      MON_EC,      "Monitoring"),
        (0.86,  GOV_FC,      GOV_EC,      "Governance"),
    ]
    sq = 0.016
    for lx, fc, ec, lbl in items:
        ax.add_patch(FancyBboxPatch(
            (lx, leg_y + 0.004), sq, sq,
            boxstyle="round,pad=0.002,rounding_size=0.003",
            lw=0.8, ec=ec, fc=fc, zorder=2,
        ))
        ax.text(lx + sq + 0.005, leg_y + 0.012, lbl,
                fontsize=7.5, color=TEXT, va="center")

    ax.text(0.50, leg_y - 0.018,
            "Solid arrows = data / control flow  \u00b7  "
            "Dashed arrows = feedback / governance  \u00b7  "
            "Colour coding matches component family",
            ha="center", va="center", fontsize=7.5, color=MUTED)

    # =========================================================================
    # Save
    # =========================================================================
    fig.tight_layout(pad=0.3)
    out_paths: list[Path] = []
    for p in [args.out, args.svg]:
        out = Path(p)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        out_paths.append(out)

    paper_base = Path(args.paper)
    paper_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".svg"):
        pf = paper_base.with_suffix(ext)
        fig.savefig(pf, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        out_paths.append(pf)

    plt.close(fig)
    for p in out_paths:
        print(f"  + {p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
