# GridPulse Frontend

AI-driven energy management dashboard built with **Next.js 15**, **React 19**, **Vercel AI SDK**, and **Recharts**.

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
│   │   ├── Copilot.tsx      # Floating AI chat
│   │   └── tools/           # Charts streamed by AI
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
│   │   ├── client.ts        # FastAPI typed client
│   │   ├── schema.ts        # Zod schemas
│   │   └── mock-data.ts     # Realistic demo data
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
| 7 | **AI Copilot** | Floating chat generating charts via natural language |

## Quick Start

```bash
npm install
cp .env.example .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

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
