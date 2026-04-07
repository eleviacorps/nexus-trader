export type TradeDirection = 'BUY' | 'SELL' | 'HOLD' | 'SKIP' | 'NEUTRAL'

export interface CandlePoint {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface ForecastPoint {
  minutes: number
  timestamp: string
  final_price?: number
  cone_upper?: number
  cone_lower?: number
  outer_upper?: number
  outer_lower?: number
  minority_price?: number
}

export interface SummaryBlock {
  call: TradeDirection | string
  summary: string
  reasoning: string
}

export interface JudgeProjectionPoint {
  minutes: number
  timestamp: string
  price: number
}

export interface JudgeContent {
  stance: TradeDirection | string
  confidence: string
  final_call: TradeDirection | string
  final_summary: string
  entry_zone: number[]
  stop_loss: number | null
  take_profit: number | null
  hold_time: string
  market_only_summary: SummaryBlock
  v18_summary: SummaryBlock
  combined_summary: SummaryBlock
  reasoning: string
  key_risk: string
  crowd_note: string
  regime_note: string
  invalidation: number | null
}

export interface JudgeEnvelope {
  available?: boolean
  provider?: string
  model?: string
  reason?: string
  error?: string
  content: JudgeContent
  projection_path?: {
    label?: string
    entry_mid?: number
    target?: number
    stop_loss?: number
    points: JudgeProjectionPoint[]
  }
  separate_from_v18?: boolean
}

export interface PaperPosition {
  trade_id: string
  symbol: string
  direction: string
  lot: number
  lot_source?: string
  requested_lot?: number | null
  entry_price: number
  entry_time?: string
  current_price?: number
  unrealized_pnl_usd?: number
  stop_loss?: number | null
  take_profit?: number | null
  sl_hit?: boolean
  tp_hit?: boolean
  exit_price?: number
  exit_time?: string
  pnl_usd?: number
}

export interface PaperState {
  summary: {
    balance: number
    equity: number
    realized_pnl: number
    unrealized_pnl: number
    total_trades: number
    open_positions: number
    win_rate: number | null
  }
  open_positions: PaperPosition[]
  closed_trades: PaperPosition[]
  updated_at?: string
}

export interface DashboardPayload {
  symbol: string
  mode?: string
  market?: {
    current_price?: number
    candles?: CandlePoint[]
  }
  realtime_chart?: {
    candles?: CandlePoint[]
  }
  simulation?: {
    direction?: TradeDirection | string
    confidence_tier?: string
    tier_label?: string
    overall_confidence?: number
    cabr_score?: number
    cpm_score?: number
    cone_width_pips?: number
    consensus_path?: number[]
    minority_path?: number[]
    cone_outer_upper?: number[]
    cone_outer_lower?: number[]
    hurst_overall?: number
    hurst_asymmetry?: number
    detected_regime?: string
    sqt_label?: string
    sqt_accuracy?: number
    should_execute?: boolean
    execution_reason?: string
    suggested_lot?: number
    mode?: string
    branch_count?: number
    crowd_persona?: string
  }
  technical_analysis?: {
    structure?: string
    location?: string
    rsi_14?: number
    atr_14?: number
    equilibrium?: number
    nearest_support?: { price?: number }
    nearest_resistance?: { price?: number }
  }
  final_forecast?: {
    points?: ForecastPoint[]
  }
  kimi_judge?: JudgeEnvelope
  local_judge?: JudgeEnvelope
  judge_comparison?: {
    agreement?: boolean
    agreement_label?: string
    kimi_call?: TradeDirection | string
    local_call?: TradeDirection | string
    summary?: string
    reasoning?: string
    preferred_source?: string
    v19_should_execute?: boolean
    v19_execution_reason?: string
  }
  v19_runtime?: {
    available?: boolean
    selected_branch_id?: number
    selected_branch_label?: string
    decision_direction?: TradeDirection | string
    cabr_score?: number
    cabr_raw_score?: number
    cpm_score?: number
    confidence_tier?: string
    sqt_label?: string
    cone_width_pips?: number
    lepl_action?: string
    lepl_probabilities?: Record<string, number>
    lepl_features?: Record<string, number | string | boolean>
    should_execute?: boolean
    execution_reason?: string
    branch_scores?: Array<{
      branch_id?: number
      branch_label?: string
      decision_direction?: TradeDirection | string
      cabr_score?: number
      cabr_raw_score?: number
    }>
  }
  feeds?: {
    news?: Array<{ title?: string; source?: string; sentiment?: number }>
    public_discussions?: Array<{ title?: string; source?: string; sentiment?: number }>
    fear_greed?: { value?: number; classification?: string }
    macro?: { macro_bias?: number; macro_shock?: number }
  }
  mfg?: {
    disagreement?: number
    consensus_drift?: number
  }
  paper_trading?: PaperState
  sqt?: {
    label?: string
    rolling_accuracy?: number
  }
}

export interface PacketLogEntry {
  request_kind?: string
  packet_bucket_15m_utc?: string
  status?: string
  model?: string
  context?: Record<string, unknown>
  numeric_glossary?: Record<string, { value?: number | string; meaning?: string }>
}

export interface PacketLogPayload {
  entries: PacketLogEntry[]
}

export interface SystemTelemetry {
  gpu_available: boolean
  gpu_name: string
  gpu_utilization_pct: number | null
  gpu_memory_used_mb: number | null
  gpu_memory_total_mb: number | null
  gpu_temperature_c: number | null
  broker_connection: string
  local_runtime: string
}
