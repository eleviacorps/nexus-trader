import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { ColorType, LineStyle, createChart } from 'lightweight-charts'
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

function toUnixTimestamp(value: string | number): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? Math.floor(parsed / 1000) : null
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
  const hostRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<any>(null)
  const candleSeriesRef = useRef<any>(null)
  const lineSeriesRef = useRef<Map<string, any>>(new Map())
  const seededRef = useRef(false)

  useEffect(() => {
    if (!hostRef.current || chartRef.current) return
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

    const resizeObserver = new ResizeObserver(() => {
      if (!hostRef.current || !chartRef.current) return
      chartRef.current.applyOptions({
        width: hostRef.current.clientWidth || 600,
        height,
      })
    })
    resizeObserver.observe(hostRef.current)

    return () => {
      resizeObserver.disconnect()
      lineSeriesRef.current.forEach((series) => chart.removeSeries(series))
      lineSeriesRef.current.clear()
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
    if (!chartRef.current) return
    const chart = chartRef.current
    const preserved = seededRef.current ? chart.timeScale().getVisibleLogicalRange() : null

    if (candles.length > 0 && !candleSeriesRef.current) {
      candleSeriesRef.current = chart.addCandlestickSeries({
        upColor: '#00E38C',
        downColor: '#FF4D57',
        borderUpColor: '#00E38C',
        borderDownColor: '#FF4D57',
        wickUpColor: '#00E38C',
        wickDownColor: '#FF4D57',
      })
    }

    if (candleSeriesRef.current) {
      const candleData = candles
        .map((item) => {
          const time = toUnixTimestamp(item.timestamp)
          if (!time) return null
          return {
            time,
            open: Number(item.open),
            high: Number(item.high),
            low: Number(item.low),
            close: Number(item.close),
          }
        })
        .filter(Boolean)
      candleSeriesRef.current.setData(candleData)
    }

    lines.forEach((line) => {
      if (!lineSeriesRef.current.has(line.id)) {
        const series = chart.addLineSeries({
          color: line.color,
          lineWidth: line.lineWidth ?? 2,
          lineStyle: lineStyleFor(line.lineStyle),
        })
        lineSeriesRef.current.set(line.id, series)
      }
      const series = lineSeriesRef.current.get(line.id)
      const data = line.points
        .map((item) => {
          const time = toUnixTimestamp(item.time)
          if (!time || !Number.isFinite(item.value)) return null
          return { time, value: Number(item.value) }
        })
        .filter(Boolean)
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
  }, [candles, lines])

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
