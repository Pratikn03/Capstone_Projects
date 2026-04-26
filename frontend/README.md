# ORIUS Frontend

Smart energy management dashboard built with **Next.js 15**, **React 19**, **Vercel AI SDK**, and **Recharts**.

## Architecture

```
src/
├── app/                     # Next.js App Router
│   ├── (dashboard)/         # Protected dashboard routes
│   │   ├── page.tsx         # Main Control Center
│   │   ├── forecasting/     # Probabilistic forecasting
│   │   ├── optimization/    # Dispatch optimization
│   │   ├── anomalies/       # Anomaly detection
│   │   ├── carbon/          # Carbon impact analysis
│   │   ├── monitoring/      # MLOps & model drift
│   │   └── zone/[zoneId]/   # Zone detail view
│   ├── actions.tsx          # Server Actions (AI tool execution)
│   └── layout.tsx           # Root layout with AI provider
├── components/
│   ├── ai/                  # Generative UI components
│   │   ├── ChatAssistant.tsx # Floating query assistant
│   │   └── tools/           # Charts streamed by assistant
│   │       ├── DispatchChart.tsx
│   │       ├── ForecastChart.tsx
│   │       ├── BatterySOCChart.tsx
│   │       ├── CarbonCostPanel.tsx
│   │       └── GridStatus.tsx
│   ├── charts/              # Standalone chart components
│   │   ├── AnomalyTimeline.tsx
│   │   ├── AnomalyList.tsx
│   │   └── MLOpsMonitor.tsx
│   ├── dashboard/           # Page-specific components
│   └── ui/                  # Shared UI primitives
├── lib/
│   ├── ai/state.tsx         # AI State / UI State definitions
│   ├── api/
│   │   ├── client.ts        # Server-only FastAPI typed client
│   │   ├── schema.ts        # Zod schemas
│   │   └── mock-data.ts     # Legacy chart types and inactive demo helpers
│   └── utils.ts             # Formatting helpers
└── middleware.ts             # Auth & security headers
```

## Dashboard Panels (7 Core Views)

| # | Panel | Description |
|---|-------|-------------|
| 1 | **Probabilistic Forecast** | Load/Wind/Solar with 90% conformal prediction intervals, GBM/LSTM/TCN toggle |
| 2 | **Dispatch Optimization** | Stacked area generation mix vs net load overlay |
| 3 | **Battery SOC** | State-of-charge trajectory with charge/discharge shading |
| 4 | **Cost–Carbon Tradeoff** | Pareto frontier from multi-objective dispatch optimization |
| 5 | **Anomaly Timeline** | Forecast residual z-scores with Isolation Forest detection |
| 6 | **MLOps Monitor** | Data drift (KS), model drift (rolling RMSE), retraining signals |
| 7 | **Query Assistant** | Floating chat for natural language queries |

## Quick Start

```bash
npm install
cp .env.example .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Production Readiness

Required server-side keys/configuration:

| Variable | Required | Purpose |
|---|---:|---|
| `FASTAPI_URL` | Yes | FastAPI backend base URL. |
| `API_SECRET_KEY` | If backend enforces API key | Sent to FastAPI as `X-ORIUS-Key`. |
| `OPENAI_API_KEY` | If chat is enabled | Enables `/api/chat`; otherwise chat returns `503` instead of crashing. |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o`. |
| `FASTAPI_TIMEOUT_MS` | No | Backend request timeout, default `12000`. |
| `DASHBOARD_BASIC_AUTH` | Recommended in production | `username:password` for browser Basic Auth. |

Client-visible labels:

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_DASHBOARD_APP_LABEL` | Sidebar/topbar product label. |
| `NEXT_PUBLIC_DASHBOARD_OPERATOR_NAME` | Operator display name. |
| `NEXT_PUBLIC_DASHBOARD_OPERATOR_ROLE` | Operator role label. |

Production checks:

```bash
npm run type-check
npm run build
curl -fsS http://localhost:3000/api/health
```

The dashboard prefers live FastAPI data where available and falls back to frozen
local report/dashboard artifacts for report and data views. Live DC3S actions
remain backend-dependent and fail closed with an explicit `503`.

## Tech Stack

| Requirement | Technology |
|---|---|
| Framework | Next.js 15 (App Router, RSC) |
| AI Orchestration | Vercel AI SDK (RSC, streamUI) |
| Visualization | Recharts (SVG, composable) |
| Styling | Tailwind CSS v4 (glassmorphism) |
| Animation | Framer Motion |
| Validation | Zod (runtime type safety) |
| Backend | FastAPI (Python, Pyomo) |
