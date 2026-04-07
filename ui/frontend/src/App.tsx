import {
  startTransition,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import {
  Activity,
  Bot,
  BrainCircuit,
  CandlestickChart,
  CircleStop,
  Cpu,
  DatabaseZap,
  GitBranchPlus,
  Link2,
  MoonStar,
  Newspaper,
  Play,
  Radar,
  RefreshCcw,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  TestTubeDiagonal,
  TrendingUpDown,
  UploadCloud,
  Wallet,
  Workflow,
} from 'lucide-react'
import { clsx } from 'clsx'

import { BranchExplorer } from './components/BranchExplorer'
import { ControlTile } from './components/ControlTile'
import { GlassCard } from './components/GlassCard'
import { GlowSlider } from './components/GlowSlider'
import { HeroCard } from './components/HeroCard'
import { StatusHeader } from './components/StatusHeader'
import { type ChartLine, TradingChart } from './components/TradingChart'
import type {
  CandlePoint,
  DashboardPayload,
  JudgeEnvelope,
  PacketLogPayload,
  PaperPosition,
  PaperState,
  SystemTelemetry,
  TradeDirection,
} from './types'

type BannerTone = 'green' | 'red' | 'blue' | 'amber'
type FocusPanel = 'execution' | 'branches' | 'briefing' | 'gpu' | 'pnl' | 'logs' | 'live'

const DEFAULT_MODEL = 'moonshotai/kimi-k2-instruct'
const DEFAULT_PROVIDER = 'nvidia_nim'

const EMPTY_TELEMETRY: SystemTelemetry = {
  gpu_available: false,
  gpu_name: 'GPU unavailable',
  gpu_utilization_pct: null,
  gpu_memory_used_mb: null,
  gpu_memory_total_mb: null,
  gpu_temperature_c: null,
  broker_connection: 'Paper broker',
  local_runtime: 'Local runtime',
}

function formatCurrency(value?: number | null) {
  if (value == null || Number.isNaN(value)) return '$-'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value)
}

function formatNumber(value?: number | null, digits = 2) {
  if (value == null || Number.isNaN(value)) return '-'
  return Number(value).toFixed(digits)
}

function formatPercent(value?: number | null, digits = 1) {
  if (value == null || Number.isNaN(value)) return '-'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

function formatSigned(value?: number | null, digits = 2) {
  if (value == null || Number.isNaN(value)) return '-'
  return `${Number(value) >= 0 ? '+' : ''}${Number(value).toFixed(digits)}`
}

function formatPrice(value?: number | null) {
  if (value == null || Number.isNaN(value)) return '-'
  return Number(value).toFixed(2)
}

function sessionForDate(date: Date) {
  const hour = date.getHours()
  if (hour >= 6 && hour < 13) return 'London Session'
  if (hour >= 13 && hour < 22) return 'New York Session'
  return 'Asia Session'
}

function marketStatusLabel(connected: boolean) {
  return connected ? 'Live Market Synced' : 'Realtime Link Reconnecting'
}

function toneClassFromValue(value: TradeDirection | string | undefined) {
  const tone = String(value ?? '').toUpperCase()
  if (tone === 'BUY' || tone === 'RUNNING' || tone === 'LIVE') return 'tone-green'
  if (tone === 'SELL' || tone === 'DANGER' || tone === 'TRAINING') return 'tone-red'
  if (tone === 'HOLD' || tone === 'SKIP' || tone === 'CAUTION') return 'tone-amber'
  return 'tone-blue'
}

function jsonFetch<T>(url: string, options?: RequestInit): Promise<T> {
  return fetch(url, options).then(async (response) => {
    const text = await response.text()
    const payload = text ? JSON.parse(text) : {}
    if (!response.ok) {
      const detail =
        typeof payload?.detail === 'string'
          ? payload.detail
          : text || `${response.status} ${response.statusText}`
      throw new Error(detail)
    }
    return payload as T
  })
}

function getCandles(payload: DashboardPayload | null): CandlePoint[] {
  if (!payload) return []
  const realtime = payload.realtime_chart?.candles
  if (Array.isArray(realtime) && realtime.length) return realtime
  return payload.market?.candles ?? []
}

function mapForecastLine(
  payload: DashboardPayload | null,
  values: number[] | undefined,
  key: 'final_price' | 'minority_price' | 'outer_upper' | 'outer_lower',
  id: string,
  color: string,
  lineStyle: 'solid' | 'dashed' | 'dotted' = 'solid',
  lineWidth = 2,
): ChartLine {
  const candles = getCandles(payload)
  const forecast = payload?.final_forecast?.points ?? []
  const points: Array<{ time: string | number; value: number }> = []
  if (candles.length > 0 && values?.length) {
    points.push({
      time: candles[candles.length - 1].timestamp,
      value: values[0] ?? candles[candles.length - 1].close,
    })
  }
  forecast.forEach((point, index) => {
    const candidate = values?.[index + 1] ?? point[key]
    if (candidate == null) return
    points.push({ time: point.timestamp, value: candidate })
  })
  return { id, name: id, color, points, lineStyle, lineWidth }
}

function buildLiveLines(payload: DashboardPayload | null): ChartLine[] {
  const simulation = payload?.simulation
  const judge = payload?.kimi_judge
  const kimiLine: ChartLine = {
    id: 'kimi',
    name: 'Kimi',
    color: '#00E38C',
    lineStyle: 'dotted',
    lineWidth: 2,
    points:
      judge?.projection_path?.points?.map((point) => ({
        time: point.timestamp,
        value: point.price,
      })) ?? [],
  }

  return [
    mapForecastLine(payload, simulation?.consensus_path, 'final_price', 'consensus', '#5BA7FF', 'solid', 3),
    mapForecastLine(payload, simulation?.minority_path, 'minority_price', 'minority', '#FFC857', 'dashed', 2),
    mapForecastLine(payload, simulation?.cone_outer_upper, 'outer_upper', 'outer-upper', '#FF4D57', 'dashed', 2),
    mapForecastLine(payload, simulation?.cone_outer_lower, 'outer_lower', 'outer-lower', '#FF4D57', 'dashed', 2),
    kimiLine,
  ]
}

function buildPredictedVsActual(payload: DashboardPayload | null): ChartLine[] {
  const candles = getCandles(payload).slice(-28)
  const actual: ChartLine = {
    id: 'actual',
    name: 'Actual',
    color: '#FFFFFF',
    lineWidth: 2,
    points: candles.map((candle) => ({ time: candle.timestamp, value: candle.close })),
  }
  const predicted = mapForecastLine(payload, payload?.simulation?.consensus_path, 'final_price', 'predicted', '#5BA7FF', 'solid', 3)
  const minority = mapForecastLine(payload, payload?.simulation?.minority_path, 'minority_price', 'branch-minority', '#FFC857', 'dashed', 2)
  return [actual, predicted, minority]
}

function buildConeLines(payload: DashboardPayload | null): ChartLine[] {
  return [
    mapForecastLine(payload, payload?.simulation?.consensus_path, 'final_price', 'cone-center', '#5BA7FF', 'solid', 3),
    mapForecastLine(payload, payload?.simulation?.cone_outer_upper, 'outer_upper', 'cone-upper', '#00E38C', 'dashed', 2),
    mapForecastLine(payload, payload?.simulation?.cone_outer_lower, 'outer_lower', 'cone-lower', '#FF4D57', 'dashed', 2),
  ]
}

function buildPnlLine(paper: PaperState | null): ChartLine[] {
  const trades = paper?.closed_trades ?? []
  let running = 0
  const points = trades.map((trade, index) => {
    running += Number(trade.pnl_usd ?? 0)
    return {
      time: trade.exit_time ?? trade.entry_time ?? `${Date.now() + index}`,
      value: running,
    }
  })
  if (!points.length) points.push({ time: `${Date.now()}`, value: 0 })
  return [
    {
      id: 'pnl',
      name: 'PnL',
      color: points[points.length - 1].value >= 0 ? '#00E38C' : '#FF4D57',
      lineWidth: 3,
      points,
    },
  ]
}

function mergeOpenPositions(paper: PaperState | null): PaperPosition[] {
  return paper?.open_positions ?? []
}

function App() {
  const [clock, setClock] = useState(() => new Date())
  const [dashboard, setDashboard] = useState<DashboardPayload | null>(null)
  const [paperState, setPaperState] = useState<PaperState | null>(null)
  const [packetLog, setPacketLog] = useState<PacketLogPayload>({ entries: [] })
  const [telemetry, setTelemetry] = useState<SystemTelemetry>(EMPTY_TELEMETRY)
  const [banner, setBanner] = useState<{ tone: BannerTone; text: string }>({
    tone: 'blue',
    text: 'Initializing the local AI trading terminal.',
  })
  const [apiLatencyMs, setApiLatencyMs] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const [symbol, setSymbol] = useState('XAUUSD')
  const [mode, setMode] = useState<'frequency' | 'precision'>('frequency')
  const [model, setModel] = useState(DEFAULT_MODEL)
  const [provider] = useState(DEFAULT_PROVIDER)
  const [trainingMode, setTrainingMode] = useState(false)
  const [focusedPanel, setFocusedPanel] = useState<FocusPanel>('live')
  const [direction, setDirection] = useState<'BUY' | 'SELL'>('BUY')
  const [leverage, setLeverage] = useState(200)
  const [stopPips, setStopPips] = useState(20)
  const [takeProfitPips, setTakeProfitPips] = useState(30)
  const [manualLot, setManualLot] = useState('')
  const [tradeNote, setTradeNote] = useState('')
  const [riskPercent, setRiskPercent] = useState(1.2)
  const [gpuLimit, setGpuLimit] = useState(82)
  const [positionSize, setPositionSize] = useState(5)
  const [confidenceThreshold, setConfidenceThreshold] = useState(58)
  const [tradeFrequency, setTradeFrequency] = useState(15)
  const [liveState, setLiveState] = useState({
    connected: false,
    price: null as number | null,
    bar_countdown: 900,
    positions: [] as PaperPosition[],
  })
  const initialLoadRef = useRef(false)

  const newsItems = useDeferredValue(dashboard?.feeds?.news ?? [])
  const discussionItems = useDeferredValue(dashboard?.feeds?.public_discussions ?? [])
  const closedTrades = useDeferredValue(paperState?.closed_trades ?? [])

  useEffect(() => {
    const timer = window.setInterval(() => setClock(new Date()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  const refreshDesk = useCallback(async (forceKimi = false) => {
    const startedAt = performance.now()
    setLoading(true)
    try {
      const baseQuery = new URLSearchParams({
        symbol,
        mode,
        llm_provider: provider,
        llm_model: model,
      })

      const [deskPayload, paperPayload, logPayload, telemetryPayload] = await Promise.all([
        jsonFetch<DashboardPayload>(`/api/dashboard/live?${baseQuery.toString()}`),
        jsonFetch<PaperState>(`/api/paper/state?symbol=${encodeURIComponent(symbol)}`),
        jsonFetch<PacketLogPayload>('/api/llm/kimi-log?limit=12'),
        jsonFetch<SystemTelemetry>('/api/system/telemetry').catch(() => EMPTY_TELEMETRY),
      ])

      let nextPayload = deskPayload
      if (forceKimi || !deskPayload.kimi_judge?.available) {
        const kimiQuery = new URLSearchParams(baseQuery)
        if (forceKimi) kimiQuery.set('force', '1')
        const kimiResponse = await jsonFetch<{ kimi_judge: JudgeEnvelope }>(
          `/api/llm/kimi-live?${kimiQuery.toString()}`,
        )
        nextPayload = { ...deskPayload, kimi_judge: kimiResponse.kimi_judge }
      }

      const latency = Math.round(performance.now() - startedAt)
      startTransition(() => {
        setDashboard(nextPayload)
        setPaperState(paperPayload)
        setPacketLog(logPayload)
        setTelemetry(telemetryPayload)
        setApiLatencyMs(latency)
        setBanner({
          tone: nextPayload.kimi_judge?.available ? 'green' : 'amber',
          text: nextPayload.kimi_judge?.available
            ? `Desk refreshed on ${nextPayload.kimi_judge.model ?? model}.`
            : nextPayload.kimi_judge?.error ||
              nextPayload.kimi_judge?.reason ||
              'Simulator updated with cached Kimi context.',
        })
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown refresh error'
      setBanner({ tone: 'red', text: message })
    } finally {
      setLoading(false)
    }
  }, [mode, model, provider, symbol])

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    void refreshDesk(true)
  }, [refreshDesk])

  useEffect(() => {
    const interval = window.setInterval(() => {
      void refreshDesk(false)
    }, Math.max(15000, tradeFrequency * 1000))
    return () => window.clearInterval(interval)
  }, [tradeFrequency, refreshDesk])

  const handleSocketMessage = useCallback((payload: {
    price?: number
    bar_countdown?: number
    positions?: PaperPosition[]
  }) => {
    startTransition(() => {
      setLiveState((current) => ({
        connected: true,
        price: payload.price ?? current.price,
        bar_countdown: payload.bar_countdown ?? current.bar_countdown,
        positions: payload.positions ?? current.positions,
      }))
    })
  }, [])

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/live?symbol=${symbol}`)

    socket.onopen = () => {
      startTransition(() => {
        setLiveState((current) => ({ ...current, connected: true }))
      })
    }

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as {
          price?: number
          bar_countdown?: number
          positions?: PaperPosition[]
        }
        handleSocketMessage(payload)
      } catch {
        setBanner({ tone: 'amber', text: 'Realtime socket sent an unreadable payload.' })
      }
    }

    socket.onclose = () => {
      startTransition(() => {
        setLiveState((current) => ({ ...current, connected: false }))
      })
    }

    return () => socket.close()
  }, [symbol, handleSocketMessage])

  const currentPrice = liveState.price ?? dashboard?.market?.current_price ?? null
  const activePaper = paperState ?? dashboard?.paper_trading ?? null
  const activePositions = mergeOpenPositions(activePaper)
  const judge = dashboard?.kimi_judge?.content
  const latestPacket = packetLog.entries[packetLog.entries.length - 1]
  const currentSession = sessionForDate(clock)

  const confidenceLabel =
    judge?.confidence ??
    dashboard?.simulation?.confidence_tier?.replaceAll('_', ' ').toUpperCase() ??
    'LOW'

  const portfolioStats = activePaper?.summary
  const liveLines = useMemo(() => buildLiveLines(dashboard), [dashboard])
  const predictedActualLines = useMemo(() => buildPredictedVsActual(dashboard), [dashboard])
  const coneLines = useMemo(() => buildConeLines(dashboard), [dashboard])
  const pnlLines = useMemo(() => buildPnlLine(activePaper), [activePaper])

  const branchReasons = [
    `Strong ${dashboard?.simulation?.detected_regime ?? 'regime'} match across the 15-minute selector.`,
    `CABR ${formatPercent(dashboard?.simulation?.cabr_score)} confirms the chosen path is the strongest ranked branch.`,
    `Cone width ${formatNumber(dashboard?.simulation?.cone_width_pips, 1)} pips remains inside the current volatility envelope.`,
    `${dashboard?.technical_analysis?.location ?? 'Market'} location and ${dashboard?.technical_analysis?.structure ?? 'structure'} reduce branch dislocation.`,
  ]

  const handleTileClick = async (tile: string) => {
    switch (tile) {
      case 'start-training':
        setTrainingMode(true)
        setFocusedPanel('gpu')
        setBanner({ tone: 'green', text: 'Training mode armed on the local terminal.' })
        return
      case 'stop-training':
        setTrainingMode(false)
        setFocusedPanel('gpu')
        setBanner({ tone: 'amber', text: 'Training mode idled. Live desk remains available.' })
        return
      case 'backtest':
        setFocusedPanel('pnl')
        setBanner({ tone: 'blue', text: 'Backtest deck focused. Refreshing the latest runtime snapshot.' })
        await refreshDesk(false)
        return
      case 'live-trade':
      case 'risk-settings':
      case 'broker-link':
        setFocusedPanel('execution')
        return
      case 'strategy-config':
        setMode((current) => (current === 'frequency' ? 'precision' : 'frequency'))
        setBanner({ tone: 'blue', text: 'Strategy mode toggled between frequency and precision.' })
        return
      case 'news-feed':
        setFocusedPanel('briefing')
        return
      case 'data-sync':
      case 'reload-market-data':
        setFocusedPanel('live')
        await refreshDesk(true)
        return
      case 'gpu-monitor':
        setFocusedPanel('gpu')
        return
      case 'logs':
        setFocusedPanel('logs')
        return
      case 'branch-explorer':
      case 'top-futures':
      case 'minority-scenario':
        setFocusedPanel('branches')
        return
      case 'dark-mode':
        setBanner({ tone: 'blue', text: 'Dark OLED theme is locked in for this terminal.' })
        return
      default:
        return
    }
  }

  const applyKimiToExecution = () => {
    if (!judge) return
    const stance = String(judge.stance ?? '').toUpperCase()
    if (stance === 'BUY' || stance === 'SELL') setDirection(stance)
    if (judge.entry_zone?.length === 2 && judge.stop_loss != null && judge.take_profit != null) {
      const midpoint = (judge.entry_zone[0] + judge.entry_zone[1]) / 2
      const pipSize = symbol === 'EURUSD' ? 0.0001 : symbol === 'BTCUSD' ? 1 : 0.1
      setStopPips(Math.max(1, Math.abs(midpoint - judge.stop_loss) / pipSize))
      setTakeProfitPips(Math.max(1, Math.abs(judge.take_profit - midpoint) / pipSize))
    }
    setFocusedPanel('execution')
    setBanner({
      tone:
        judge.final_call?.toUpperCase() === 'BUY'
          ? 'green'
          : judge.final_call?.toUpperCase() === 'SELL'
            ? 'red'
            : 'amber',
      text: judge.final_summary || 'Kimi guidance applied to the execution console.',
    })
  }

  const openPaperTrade = async () => {
    try {
      await jsonFetch('/api/paper/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          direction,
          entry_price: currentPrice,
          confidence_tier: dashboard?.simulation?.confidence_tier ?? 'moderate',
          sqt_label: dashboard?.sqt?.label ?? dashboard?.simulation?.sqt_label ?? 'NEUTRAL',
          mode,
          leverage,
          stop_pips: stopPips,
          take_profit_pips: takeProfitPips,
          manual_lot: manualLot ? Number(manualLot) : null,
          note: tradeNote,
        }),
      })
      setBanner({ tone: 'green', text: `${direction} paper trade opened on ${symbol}.` })
      await refreshDesk(false)
    } catch (error) {
      setBanner({
        tone: 'red',
        text: error instanceof Error ? error.message : 'Unable to open paper trade.',
      })
    }
  }

  const closePaperTrade = async (tradeId: string) => {
    try {
      await jsonFetch('/api/paper/close', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trade_id: tradeId, exit_price: currentPrice }),
      })
      setBanner({ tone: 'amber', text: `Trade ${tradeId.slice(0, 8)} was closed.` })
      await refreshDesk(false)
    } catch (error) {
      setBanner({
        tone: 'red',
        text: error instanceof Error ? error.message : 'Unable to close trade.',
      })
    }
  }

  const resetPaperDesk = async () => {
    const confirmed = window.confirm('Reset the local paper desk back to $1,000?')
    if (!confirmed) return
    await jsonFetch('/api/paper/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ starting_balance: 1000 }),
    })
    setBanner({ tone: 'amber', text: 'Paper desk reset to the base balance.' })
    await refreshDesk(false)
  }

  const actionTiles = [
    { id: 'start-training', label: 'Start Training', icon: Play, active: trainingMode, tone: 'green' as const },
    { id: 'stop-training', label: 'Stop Training', icon: CircleStop, active: !trainingMode, tone: 'red' as const },
    { id: 'backtest', label: 'Backtest', icon: TestTubeDiagonal, active: focusedPanel === 'pnl', tone: 'blue' as const },
    { id: 'live-trade', label: 'Live Trade', icon: Radar, active: focusedPanel === 'execution', tone: 'red' as const },
    { id: 'strategy-config', label: 'Strategy Config', icon: Settings2, active: mode === 'precision', tone: 'blue' as const },
    { id: 'risk-settings', label: 'Risk Settings', icon: ShieldAlert, active: focusedPanel === 'execution', tone: 'amber' as const },
    { id: 'news-feed', label: 'News Feed', icon: Newspaper, active: focusedPanel === 'briefing', tone: 'blue' as const },
    { id: 'data-sync', label: 'Data Sync', icon: UploadCloud, active: loading, tone: 'green' as const },
    { id: 'gpu-monitor', label: 'GPU Monitor', icon: Cpu, active: focusedPanel === 'gpu', tone: 'green' as const },
    { id: 'logs', label: 'Logs', icon: DatabaseZap, active: focusedPanel === 'logs', tone: 'amber' as const },
    { id: 'broker-link', label: 'Broker Link', icon: Link2, active: true, tone: 'green' as const },
    { id: 'dark-mode', label: 'Dark Mode', icon: MoonStar, active: true, tone: 'blue' as const },
    { id: 'branch-explorer', label: 'Branch Explorer', icon: GitBranchPlus, active: focusedPanel === 'branches', tone: 'blue' as const },
    { id: 'top-futures', label: 'Top-3 Futures', icon: TrendingUpDown, active: focusedPanel === 'branches', tone: 'green' as const },
    { id: 'minority-scenario', label: 'Minority Scenario', icon: Workflow, active: focusedPanel === 'branches', tone: 'amber' as const },
    { id: 'reload-market-data', label: 'Reload Market Data', icon: RefreshCcw, active: loading, tone: 'blue' as const },
  ]

  return (
    <div className="min-h-screen px-6 pb-10 pt-6 text-white md:px-8 xl:px-10">
      <div className="mx-auto flex w-full max-w-[1800px] flex-col gap-6">
        <StatusHeader
          clockLabel={clock.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          dateLabel={clock.toLocaleDateString([], { month: 'short', day: 'numeric' })}
          sessionLabel={currentSession}
          marketStatus={marketStatusLabel(liveState.connected)}
          gpuLabel={
            telemetry.gpu_available
              ? `${telemetry.gpu_name} ${formatNumber(telemetry.gpu_utilization_pct, 0)}%`
              : 'GPU telemetry offline'
          }
          gpuMemoryLabel={
            telemetry.gpu_memory_total_mb
              ? `${formatNumber(telemetry.gpu_memory_used_mb, 0)}/${formatNumber(telemetry.gpu_memory_total_mb, 0)} MB`
              : 'GPU memory unavailable'
          }
          modelLabel={model}
          latencyLabel={`${apiLatencyMs || 0} ms`}
          brokerLabel={telemetry.broker_connection}
          modeLabel={trainingMode ? 'Training Mode' : 'Live Mode'}
        />

        <div className={`terminal-banner terminal-banner-${banner.tone}`}>
          <Sparkles size={16} />
          <span>{banner.text}</span>
        </div>

        <section className="grid gap-6 2xl:grid-cols-2">
          <HeroCard
            title="Portfolio Summary"
            subtitle="Live local paper-book telemetry with equity, realized movement, and active terminal exposure."
            icon={<Wallet size={24} />}
            primaryValue={formatCurrency(portfolioStats?.equity)}
            primaryLabel="Current Equity"
            accent="green"
            stats={[
              {
                label: 'Daily PnL',
                value: formatCurrency((portfolioStats?.realized_pnl ?? 0) + (portfolioStats?.unrealized_pnl ?? 0)),
                tone: (portfolioStats?.realized_pnl ?? 0) + (portfolioStats?.unrealized_pnl ?? 0) >= 0 ? 'green' : 'red',
              },
              { label: 'Win Rate', value: formatPercent(portfolioStats?.win_rate), tone: 'blue' },
              { label: 'Active Positions', value: String(portfolioStats?.open_positions ?? 0), tone: 'amber' },
              { label: 'Closed Trades', value: String(portfolioStats?.total_trades ?? 0), tone: 'neutral' },
            ]}
          />
          <HeroCard
            title="AI Signal Status"
            subtitle="The top-layer AI judge merged with the V18 branch selector for the current desktop terminal."
            icon={<Bot size={24} />}
            primaryValue={String(judge?.final_call ?? dashboard?.simulation?.direction ?? 'HOLD')}
            primaryLabel="Final Desk Call"
            accent="blue"
            stats={[
              { label: 'Confidence', value: confidenceLabel, tone: 'blue' },
              { label: 'Model', value: model.split('/').at(-1) ?? model, tone: 'neutral' },
              { label: 'Selector', value: dashboard?.simulation?.tier_label ?? 'Calibrating', tone: 'green' },
              { label: 'Mode', value: mode.toUpperCase(), tone: 'amber' },
            ]}
          />
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
          <GlassCard
            title="Action Matrix"
            subtitle="Local terminal actions with tactile soft controls for the trading desk, branch explorer, GPU runtime, and execution flow."
            icon={<BrainCircuit size={20} />}
          >
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              {actionTiles.map((tile) => (
                <ControlTile
                  key={tile.id}
                  label={tile.label}
                  icon={tile.icon}
                  active={tile.active}
                  tone={tile.tone}
                  onClick={() => void handleTileClick(tile.id)}
                />
              ))}
            </div>
          </GlassCard>

          <GlassCard
            title="Execution Console"
            subtitle="Manual local trade placement with Kimi-assisted setup, equity-scaled lot sizing, and tactile broker controls."
            icon={<CandlestickChart size={20} />}
            className={clsx(focusedPanel === 'execution' && 'panel-focused')}
          >
            <div className="grid gap-6">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="field-shell">
                  <label className="field-label">Instrument</label>
                  <select value={symbol} onChange={(event) => setSymbol(event.target.value)} className="field-input">
                    <option value="XAUUSD">XAUUSD</option>
                    <option value="EURUSD">EURUSD</option>
                    <option value="BTCUSD">BTCUSD</option>
                  </select>
                </div>
                <div className="field-shell">
                  <label className="field-label">Direction</label>
                  <select value={direction} onChange={(event) => setDirection(event.target.value as 'BUY' | 'SELL')} className="field-input">
                    <option value="BUY">BUY</option>
                    <option value="SELL">SELL</option>
                  </select>
                </div>
                <div className="field-shell">
                  <label className="field-label">Mode</label>
                  <select value={mode} onChange={(event) => setMode(event.target.value as 'frequency' | 'precision')} className="field-input">
                    <option value="frequency">Frequency</option>
                    <option value="precision">Precision</option>
                  </select>
                </div>
                <div className="field-shell">
                  <label className="field-label">Model Version</label>
                  <input value={model} onChange={(event) => setModel(event.target.value)} className="field-input" />
                </div>
                <div className="field-shell">
                  <label className="field-label">Leverage</label>
                  <select value={leverage} onChange={(event) => setLeverage(Number(event.target.value))} className="field-input">
                    <option value={50}>1:50</option>
                    <option value={100}>1:100</option>
                    <option value={200}>1:200</option>
                  </select>
                </div>
                <div className="field-shell">
                  <label className="field-label">Manual Lot</label>
                  <input value={manualLot} onChange={(event) => setManualLot(event.target.value)} placeholder="Auto if blank" className="field-input" />
                </div>
                <div className="field-shell">
                  <label className="field-label">Stop Pips</label>
                  <input value={stopPips} onChange={(event) => setStopPips(Number(event.target.value))} type="number" className="field-input" />
                </div>
                <div className="field-shell">
                  <label className="field-label">Take Profit Pips</label>
                  <input value={takeProfitPips} onChange={(event) => setTakeProfitPips(Number(event.target.value))} type="number" className="field-input" />
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="stat-chip">
                  <div className="text-[11px] tracking-[0.24em] text-white/36 uppercase">Live Price</div>
                  <div className="mt-3 font-mono text-3xl font-semibold text-white">{formatPrice(currentPrice)}</div>
                </div>
                <div className="stat-chip">
                  <div className="text-[11px] tracking-[0.24em] text-white/36 uppercase">Suggested Lot</div>
                  <div className="mt-3 font-mono text-3xl font-semibold text-[#00E38C]">
                    {formatNumber(dashboard?.simulation?.suggested_lot, 2)} lot
                  </div>
                </div>
                <div className="stat-chip">
                  <div className="text-[11px] tracking-[0.24em] text-white/36 uppercase">Kimi Summary</div>
                  <div className={`mt-3 font-mono text-2xl font-semibold ${toneClassFromValue(judge?.final_call)}`}>
                    {judge?.final_call ?? 'SKIP'}
                  </div>
                </div>
              </div>

              <div className="field-shell">
                <label className="field-label">Trade Note</label>
                <textarea
                  value={tradeNote}
                  onChange={(event) => setTradeNote(event.target.value)}
                  className="field-input min-h-[104px] resize-none"
                  placeholder="Desktop note for the local execution log"
                />
              </div>

              <div className="flex flex-wrap gap-3">
                <button type="button" className="terminal-button terminal-button-green" onClick={applyKimiToExecution}>
                  Apply Kimi Setup
                </button>
                <button type="button" className="terminal-button terminal-button-blue" onClick={() => void openPaperTrade()}>
                  Open Paper Trade
                </button>
                <button type="button" className="terminal-button terminal-button-ghost" onClick={() => void resetPaperDesk()}>
                  Reset Desk
                </button>
              </div>
            </div>
          </GlassCard>
        </section>

        <GlassCard
          title="Slider Bank"
          subtitle="Pill-shaped local control surfaces for risk, GPU ceilings, position sizing, confidence gating, and desk cadence."
          icon={<SlidersHorizontal size={20} />}
        >
          <div className="grid gap-4 xl:grid-cols-5">
            <GlowSlider
              label="Risk %"
              value={riskPercent}
              min={0.25}
              max={3}
              step={0.05}
              tone="green"
              valueLabel={`${riskPercent.toFixed(2)}%`}
              onChange={setRiskPercent}
            />
            <GlowSlider
              label="GPU Usage Limit"
              value={gpuLimit}
              min={35}
              max={100}
              tone="amber"
              valueLabel={`${gpuLimit.toFixed(0)}%`}
              onChange={setGpuLimit}
            />
            <GlowSlider
              label="Position Size"
              value={positionSize}
              min={1}
              max={20}
              tone="green"
              valueLabel={`${positionSize.toFixed(0)} / 20`}
              onChange={setPositionSize}
            />
            <GlowSlider
              label="Confidence Threshold"
              value={confidenceThreshold}
              min={40}
              max={90}
              tone="blue"
              valueLabel={`${confidenceThreshold.toFixed(0)}%`}
              onChange={setConfidenceThreshold}
            />
            <GlowSlider
              label="Trade Frequency"
              value={tradeFrequency}
              min={15}
              max={60}
              step={5}
              tone="red"
              valueLabel={`${tradeFrequency.toFixed(0)}s`}
              onChange={setTradeFrequency}
            />
          </div>
        </GlassCard>

        <section className="grid gap-6 xl:grid-cols-[1.25fr_0.95fr]">
          <TradingChart
            title="Live Price Chart"
            subtitle="Realtime market structure with the main V18 branch, minority scenario, confidence rails, and a separate Kimi projection."
            candles={getCandles(dashboard)}
            lines={liveLines}
            height={420}
            aside={
              <div className="flex flex-wrap gap-2 text-[11px] tracking-[0.18em] text-white/48 uppercase">
                <span className="status-pill status-pill-blue">Consensus</span>
                <span className="status-pill status-pill-amber">Minority</span>
                <span className="status-pill status-pill-red">Cone</span>
                <span className="status-pill status-pill-green">Kimi</span>
              </div>
            }
          />

          <div className={clsx(focusedPanel === 'branches' && 'panel-focused')}>
            <BranchExplorer
              consensusPath={dashboard?.simulation?.consensus_path ?? []}
              minorityPath={dashboard?.simulation?.minority_path ?? []}
              outerUpper={dashboard?.simulation?.cone_outer_upper ?? []}
              outerLower={dashboard?.simulation?.cone_outer_lower ?? []}
              regimeLabel={dashboard?.simulation?.detected_regime ?? 'Unknown regime'}
              selectorLabel={dashboard?.simulation?.tier_label ?? 'Selector calibrating'}
              reasons={branchReasons}
            />
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-3">
          <TradingChart
            title="Predicted vs Actual"
            subtitle="Recent realized price action against the main path and minority future branch."
            lines={predictedActualLines}
            height={300}
          />

          <GlassCard
            title="Neural Briefing"
            subtitle="Kimi summaries, live headlines, and public discussion bias in one premium local intelligence pane."
            icon={<Newspaper size={20} />}
            className={clsx(focusedPanel === 'briefing' && 'panel-focused')}
          >
            <div className="grid gap-4">
              <div className="summary-block">
                <div className="summary-label">Final Call</div>
                <div className={`summary-value ${toneClassFromValue(judge?.final_call)}`}>{judge?.final_call ?? 'SKIP'}</div>
                <p className="summary-copy">{judge?.final_summary ?? 'Waiting for the latest Kimi desk packet.'}</p>
              </div>
              <div className="grid gap-3">
                {[judge?.market_only_summary, judge?.v18_summary, judge?.combined_summary]
                  .filter(Boolean)
                  .map((block) => (
                    <div key={block?.summary} className="summary-card">
                      <div className={`summary-call ${toneClassFromValue(block?.call)}`}>{block?.call ?? 'SKIP'}</div>
                      <p className="text-sm text-white/72">{block?.summary}</p>
                      <p className="text-xs leading-6 text-white/48">{block?.reasoning}</p>
                    </div>
                  ))}
              </div>
              <div className="grid gap-3">
                {newsItems.slice(0, 3).map((item, index) => (
                  <div key={`${item.title}-${index}`} className="news-card">
                    <div className="news-title">{item.title}</div>
                    <div className="news-meta">
                      {item.source ?? 'local feed'} • sentiment {formatSigned(item.sentiment ? item.sentiment * 100 : 0, 0)}%
                    </div>
                  </div>
                ))}
                {discussionItems.slice(0, 2).map((item, index) => (
                  <div key={`${item.title}-${index}`} className="news-card">
                    <div className="news-title">{item.title}</div>
                    <div className="news-meta">
                      public discussion • {item.source ?? 'desk'} • {formatSigned(item.sentiment ? item.sentiment * 100 : 0, 0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </GlassCard>

          <GlassCard
            title="GPU / Training Monitor"
            subtitle="Local GPU telemetry and the desk runtime state for your machine."
            icon={<Cpu size={20} />}
            className={clsx(focusedPanel === 'gpu' && 'panel-focused')}
          >
            <div className="grid gap-4">
              <div className="monitor-bar">
                <div className="monitor-label">GPU Utilization</div>
                <div className="monitor-track">
                  <div className="monitor-fill bg-[#00E38C]" style={{ width: `${telemetry.gpu_utilization_pct ?? 0}%` }} />
                </div>
                <div className="monitor-value">{formatNumber(telemetry.gpu_utilization_pct, 0)}%</div>
              </div>
              <div className="monitor-bar">
                <div className="monitor-label">Memory Pressure</div>
                <div className="monitor-track">
                  <div
                    className="monitor-fill bg-[#5BA7FF]"
                    style={{
                      width: `${
                        telemetry.gpu_memory_total_mb
                          ? ((telemetry.gpu_memory_used_mb ?? 0) / telemetry.gpu_memory_total_mb) * 100
                          : 0
                      }%`,
                    }}
                  />
                </div>
                <div className="monitor-value">
                  {telemetry.gpu_memory_total_mb
                    ? `${formatNumber(telemetry.gpu_memory_used_mb, 0)} / ${formatNumber(telemetry.gpu_memory_total_mb, 0)} MB`
                    : 'Unavailable'}
                </div>
              </div>
              <div className="monitor-bar">
                <div className="monitor-label">Training Runtime</div>
                <div className="monitor-track">
                  <div
                    className={clsx('monitor-fill', trainingMode ? 'bg-[#FF4D57]' : 'bg-[#FFC857]')}
                    style={{ width: trainingMode ? '82%' : '35%' }}
                  />
                </div>
                <div className="monitor-value">{trainingMode ? 'Running locally' : 'Idle / live routing'}</div>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="stat-chip">
                  <div className="text-[11px] tracking-[0.24em] text-white/36 uppercase">GPU</div>
                  <div className="mt-2 text-lg font-semibold text-white">{telemetry.gpu_name}</div>
                </div>
                <div className="stat-chip">
                  <div className="text-[11px] tracking-[0.24em] text-white/36 uppercase">Temperature</div>
                  <div className="mt-2 font-mono text-lg font-semibold text-white">
                    {telemetry.gpu_temperature_c != null ? `${formatNumber(telemetry.gpu_temperature_c, 0)}°C` : 'Unavailable'}
                  </div>
                </div>
              </div>
            </div>
          </GlassCard>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
          <TradingChart
            title="PnL History"
            subtitle="Closed paper-book performance with cumulative gain or drawdown over recent exits."
            lines={pnlLines}
            height={280}
            aside={<div className="status-pill status-pill-blue">{closedTrades.length} closed trades</div>}
          />

          <TradingChart
            title="Confidence Cone"
            subtitle="Branch center line with the upper and lower envelope of the active cone."
            lines={coneLines}
            height={280}
          />
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
          <GlassCard
            title="Open Positions"
            subtitle="Live local paper positions with close controls and a condensed closed-trade ledger."
            icon={<Activity size={20} />}
            className={clsx(focusedPanel === 'execution' && 'panel-focused')}
          >
            <div className="grid gap-5">
              <div className="overflow-x-auto rounded-[24px] border border-white/8">
                <table className="desk-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Side</th>
                      <th>Lot</th>
                      <th>Entry</th>
                      <th>PnL</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {activePositions.length ? (
                      activePositions.map((position) => (
                        <tr key={position.trade_id}>
                          <td className="font-mono">{position.trade_id.slice(0, 8)}</td>
                          <td className={toneClassFromValue(position.direction)}>{position.direction}</td>
                          <td className="font-mono">{formatNumber(position.lot, 2)}</td>
                          <td className="font-mono">{formatPrice(position.entry_price)}</td>
                          <td className={Number(position.unrealized_pnl_usd ?? 0) >= 0 ? 'tone-green' : 'tone-red'}>
                            {formatCurrency(position.unrealized_pnl_usd)}
                          </td>
                          <td>
                            <button
                              type="button"
                              className="table-action"
                              onClick={() => void closePaperTrade(position.trade_id)}
                            >
                              Close
                            </button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={6} className="py-5 text-center text-sm text-white/42">
                          No open paper positions on this local desk.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="grid gap-3">
                {closedTrades.slice(-4).reverse().map((trade) => (
                  <div key={trade.trade_id} className="news-card">
                    <div className="flex items-center justify-between gap-3">
                      <div className="font-mono text-sm text-white/82">{trade.trade_id.slice(0, 8)}</div>
                      <div className={Number(trade.pnl_usd ?? 0) >= 0 ? 'tone-green' : 'tone-red'}>
                        {formatCurrency(trade.pnl_usd)}
                      </div>
                    </div>
                    <div className="news-meta">
                      {trade.direction} • entry {formatPrice(trade.entry_price)} • exit {formatPrice(trade.exit_price)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </GlassCard>

          <GlassCard
            title="Packet Log + Numeric Guide"
            subtitle="Latest Kimi packet metadata and a local explanation of the most recent numeric glossary rows."
            icon={<DatabaseZap size={20} />}
            className={clsx(focusedPanel === 'logs' && 'panel-focused')}
          >
            <div className="grid gap-4">
              <div className="summary-block">
                <div className="summary-label">Latest Packet</div>
                <div className="summary-value tone-blue">
                  {latestPacket?.packet_bucket_15m_utc ?? 'No packet yet'}
                </div>
                <p className="summary-copy">
                  {latestPacket?.request_kind ?? 'Kimi packet log'} • {latestPacket?.model ?? model} • {latestPacket?.status ?? 'pending'}
                </p>
              </div>
              <div className="grid max-h-[320px] gap-3 overflow-y-auto pr-1">
                {Object.entries(latestPacket?.numeric_glossary ?? {})
                  .slice(0, 10)
                  .map(([key, value]) => (
                    <div key={key} className="summary-card">
                      <div className="text-[11px] tracking-[0.22em] text-white/38 uppercase">{key}</div>
                      <div className="mt-2 font-mono text-lg font-semibold text-white">
                        {String(value?.value ?? '-')}
                      </div>
                      <p className="text-xs leading-6 text-white/48">{value?.meaning}</p>
                    </div>
                  ))}
              </div>
            </div>
          </GlassCard>
        </section>

        {loading ? (
          <div className="pb-4 text-center text-sm tracking-[0.22em] text-white/34 uppercase">
            Refreshing terminal state...
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default App
