import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { CandlestickSeries, ColorType, LineSeries, LineStyle, createChart } from 'lightweight-charts'
import type { ISeriesApi, LineWidth, UTCTimestamp } from 'lightweight-charts'
import type { ReactNode } from 'react'

import type { CandlePoint } from '../types'

export interface ChartLinePoint {
  time: string | number
  value: number
}

export interface ChartLine {
  id: string
  name: string
  color: string
  points: ChartLinePoint[]
  lineWidth?: number
  lineStyle?: 'solid' | 'dashed' | 'dotted'
}

interface TradingChartProps {
  title: string
  subtitle: string
  candles?: CandlePoint[]
  lines: ChartLine[]
  height?: number
  aside?: ReactNode
}

function toUnixTimestamp(value: string | number): UTCTimestamp | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? (value as UTCTimestamp) : null
  }
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? (Math.floor(parsed / 1000) as UTCTimestamp) : null
}

function lineStyleFor(style?: 'solid' | 'dashed' | 'dotted') {
  if (style === 'dashed') return LineStyle.Dashed
  if (style === 'dotted') return LineStyle.Dotted
  return LineStyle.Solid
}

export function TradingChart({
  title,
  subtitle,
  candles = [],
  lines,
  height = 360,
  aside,
}: TradingChartProps) {
  type ChartApi = ReturnType<typeof createChart>
  type CandleSeriesApi = ISeriesApi<'Candlestick'>
  type LineSeriesApi = ISeriesApi<'Line'>

  const hostRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<ChartApi | null>(null)
  const candleSeriesRef = useRef<CandleSeriesApi | null>(null)
  const lineSeriesRef = useRef<Map<string, LineSeriesApi>>(new Map())
  const seededRef = useRef(false)
  const [chartError, setChartError] = useState<string | null>(null)
  const scheduleChartError = (message: string) => {
    window.setTimeout(() => {
      setChartError((current) => current ?? message)
    }, 0)
  }

  useEffect(() => {
    if (!hostRef.current || chartRef.current) return
    let resizeObserver: ResizeObserver | null = null
    try {
      const chart = createChart(hostRef.current, {
        width: hostRef.current.clientWidth || 600,
        height,
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: 'rgba(255,255,255,0.58)',
          fontFamily: 'JetBrains Mono, monospace',
        },
        grid: {
          vertLines: { color: 'rgba(255,255,255,0.04)' },
          horzLines: { color: 'rgba(255,255,255,0.04)' },
        },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.08)' },
        timeScale: {
          borderColor: 'rgba(255,255,255,0.08)',
          timeVisible: true,
          secondsVisible: false,
        },
        crosshair: { mode: 0 },
      })
      chartRef.current = chart

      resizeObserver = new ResizeObserver(() => {
        if (!hostRef.current || !chartRef.current) return
        chartRef.current.applyOptions({
          width: hostRef.current.clientWidth || 600,
          height,
        })
      })
      resizeObserver.observe(hostRef.current)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown chart bootstrap error.'
      scheduleChartError(message)
      return
    }

    return () => {
      resizeObserver?.disconnect()
      const chart = chartRef.current
      const lineSeriesMap = lineSeriesRef.current
      if (!chart) return
      lineSeriesMap.forEach((series) => chart.removeSeries(series))
      lineSeriesMap.clear()
      if (candleSeriesRef.current) {
        chart.removeSeries(candleSeriesRef.current)
        candleSeriesRef.current = null
      }
      chart.remove()
      chartRef.current = null
      seededRef.current = false
    }
  }, [height])

  useEffect(() => {
    if (!chartRef.current || chartError) return
    try {
      const chart = chartRef.current
      const preserved = seededRef.current ? chart.timeScale().getVisibleLogicalRange() : null

      if (candles.length > 0 && !candleSeriesRef.current) {
        candleSeriesRef.current = chart.addSeries(CandlestickSeries, {
          upColor: '#00E38C',
          downColor: '#FF4D57',
          borderUpColor: '#00E38C',
          borderDownColor: '#FF4D57',
          wickUpColor: '#00E38C',
          wickDownColor: '#FF4D57',
        })
      }

      if (candleSeriesRef.current) {
        const candleData: Array<{ time: UTCTimestamp; open: number; high: number; low: number; close: number }> = []
        candles.forEach((item) => {
          const time = toUnixTimestamp(item.timestamp)
          if (!time) return
          candleData.push({
            time,
            open: Number(item.open),
            high: Number(item.high),
            low: Number(item.low),
            close: Number(item.close),
          })
        })
        candleSeriesRef.current.setData(candleData)
      }

      lines.forEach((line) => {
        if (!lineSeriesRef.current.has(line.id)) {
          const series = chart.addSeries(LineSeries, {
            color: line.color,
            lineWidth: (line.lineWidth ?? 2) as LineWidth,
            lineStyle: lineStyleFor(line.lineStyle),
          })
          lineSeriesRef.current.set(line.id, series)
        }
        const series = lineSeriesRef.current.get(line.id)
        if (!series) return
        const data: Array<{ time: UTCTimestamp; value: number }> = []
        line.points.forEach((item) => {
          const time = toUnixTimestamp(item.time)
          if (!time || !Number.isFinite(item.value)) return
          data.push({ time, value: Number(item.value) })
        })
        series.setData(data)
      })

      const activeIds = new Set(lines.map((line) => line.id))
      lineSeriesRef.current.forEach((series, key) => {
        if (activeIds.has(key)) return
        chart.removeSeries(series)
        lineSeriesRef.current.delete(key)
      })

      if (preserved && Number.isFinite(preserved.from) && Number.isFinite(preserved.to)) {
        chart.timeScale().setVisibleLogicalRange(preserved)
      } else if (!seededRef.current) {
        chart.timeScale().fitContent()
        seededRef.current = true
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown chart update error.'
      scheduleChartError(message)
    }
  }, [candles, chartError, lines])

  if (chartError) {
    return (
      <motion.section
        whileHover={{ y: -3 }}
        transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
        className="glass-card chart-card"
      >
        <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold tracking-[0.22em] text-white/92 uppercase">
              {title}
            </h3>
            <p className="mt-1 text-sm leading-6 text-white/58">{subtitle}</p>
          </div>
          {aside}
        </div>
        <div className="flex h-[240px] items-center justify-center rounded-[26px] border border-white/8 bg-black/25 px-6 text-center">
          <div>
            <div className="text-sm tracking-[0.22em] text-white/40 uppercase">Chart Fallback Active</div>
            <p className="mt-3 max-w-xl text-sm leading-7 text-white/66">
              The realtime chart renderer hit a browser-side error, so this panel stayed visible instead of crashing the whole desk.
            </p>
            <p className="mt-3 font-mono text-xs leading-6 text-[#FFC857]">{chartError}</p>
          </div>
        </div>
      </motion.section>
    )
  }

  return (
    <motion.section
      whileHover={{ y: -3 }}
      transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
      className="glass-card chart-card"
    >
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold tracking-[0.22em] text-white/92 uppercase">
            {title}
          </h3>
          <p className="mt-1 text-sm leading-6 text-white/58">{subtitle}</p>
        </div>
        {aside}
      </div>
      <div ref={hostRef} style={{ height }} className="w-full" />
    </motion.section>
  )
}
