import { motion } from 'framer-motion'
import { GitBranchPlus, Target } from 'lucide-react'

interface BranchExplorerProps {
  consensusPath: number[]
  minorityPath: number[]
  outerUpper: number[]
  outerLower: number[]
  regimeLabel: string
  selectorLabel: string
  reasons: string[]
}

function buildPoints(values: number[], width: number, height: number, min: number, max: number) {
  return values.map((value, index) => {
    const x = (index / Math.max(values.length - 1, 1)) * width
    const normalized = (value - min) / Math.max(max - min, 1)
    const y = height - normalized * height
    return { x, y }
  })
}

function buildPath(points: Array<{ x: number; y: number }>) {
  if (!points.length) return ''
  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ')
}

function buildAreaPath(upper: Array<{ x: number; y: number }>, lower: Array<{ x: number; y: number }>) {
  if (!upper.length || !lower.length) return ''
  return `${buildPath(upper)} ${lower
    .slice()
    .reverse()
    .map((point) => `L ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ')} Z`
}

export function BranchExplorer({
  consensusPath,
  minorityPath,
  outerUpper,
  outerLower,
  regimeLabel,
  selectorLabel,
  reasons,
}: BranchExplorerProps) {
  const width = 760
  const height = 260
  const merged = [...consensusPath, ...minorityPath, ...outerUpper, ...outerLower].filter(Number.isFinite)
  const min = Math.min(...merged, 0)
  const max = Math.max(...merged, 1)
  const upperPoints = buildPoints(outerUpper, width, height, min, max)
  const lowerPoints = buildPoints(outerLower, width, height, min, max)
  const consensusPoints = buildPoints(consensusPath, width, height, min, max)
  const minorityPoints = buildPoints(minorityPath, width, height, min, max)
  const upperPath = buildPath(upperPoints)
  const lowerPath = buildPath(lowerPoints)
  const consensusSvg = buildPath(consensusPoints)
  const minoritySvg = buildPath(minorityPoints)
  const areaPath = buildAreaPath(upperPoints, lowerPoints)

  const topFutures = [consensusPath, outerUpper, outerLower]
    .map((path, index) => ({
      label: index === 0 ? 'Top Branch' : `Future ${index + 1}`,
      final: path.at(-1),
    }))
    .filter((item) => item.final != null)

  return (
    <motion.section
      whileHover={{ y: -3 }}
      transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
      className="glass-card branch-card"
    >
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="icon-badge">
              <GitBranchPlus size={20} />
            </div>
            <div>
              <h3 className="text-base font-semibold tracking-[0.22em] text-white/92 uppercase">
                Future Branches
              </h3>
              <p className="mt-1 text-sm text-white/58">
                Top branch, minority branch, and confidence envelope for the current 15-minute horizon.
              </p>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          <div className="status-pill status-pill-blue">{selectorLabel}</div>
          <div className="status-pill status-pill-amber">{regimeLabel}</div>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.3fr)_320px]">
        <div className="rounded-[26px] border border-white/8 bg-black/30 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
          <svg viewBox={`0 0 ${width} ${height}`} className="h-[260px] w-full">
            <defs>
              <linearGradient id="coneFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="rgba(91,167,255,0.24)" />
                <stop offset="100%" stopColor="rgba(91,167,255,0.02)" />
              </linearGradient>
            </defs>
            {Array.from({ length: 6 }).map((_, index) => (
              <line
                key={`grid-${index}`}
                x1="0"
                y1={(index / 5) * height}
                x2={width}
                y2={(index / 5) * height}
                stroke="rgba(255,255,255,0.05)"
                strokeWidth="1"
              />
            ))}
            {areaPath ? (
              <path
                d={areaPath}
                fill="rgba(91,167,255,0.08)"
                stroke="none"
              />
            ) : null}
            <path d={upperPath} fill="none" stroke="#5BA7FF" strokeDasharray="8 6" strokeWidth="2" />
            <path d={lowerPath} fill="none" stroke="#5BA7FF" strokeDasharray="8 6" strokeWidth="2" />
            <path d={consensusSvg} fill="none" stroke="#00E38C" strokeWidth="3" strokeLinecap="round" />
            <path d={minoritySvg} fill="none" stroke="#FFC857" strokeWidth="2.5" strokeDasharray="5 5" strokeLinecap="round" />
          </svg>
        </div>

        <div className="grid gap-4">
          <div className="rounded-[26px] border border-white/8 bg-white/[0.04] p-4">
            <div className="mb-3 flex items-center gap-3">
              <Target size={16} className="text-[#00E38C]" />
              <div className="text-[11px] tracking-[0.24em] text-white/42 uppercase">
                Top 3 Futures
              </div>
            </div>
            <div className="grid gap-3">
              {topFutures.map((branch) => (
                <div key={branch.label} className="stat-chip">
                  <div className="text-[11px] tracking-[0.24em] text-white/36 uppercase">
                    {branch.label}
                  </div>
                  <div className="mt-2 font-mono text-xl font-semibold text-white">
                    {branch.final?.toFixed(2)}
                  </div>
                </div>
              ))}
              <div className="stat-chip border-[#FFC85733]">
                <div className="text-[11px] tracking-[0.24em] text-white/36 uppercase">
                  Minority Branch
                </div>
                <div className="mt-2 font-mono text-xl font-semibold text-[#FFC857]">
                  {minorityPath.at(-1)?.toFixed(2) ?? '-'}
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-[26px] border border-white/8 bg-white/[0.04] p-4">
            <div className="text-[11px] tracking-[0.24em] text-white/42 uppercase">
              Chosen Because
            </div>
            <ul className="mt-4 grid gap-3 text-sm leading-6 text-white/66">
              {reasons.map((reason) => (
                <li key={reason} className="flex items-start gap-3">
                  <span className="mt-2 h-1.5 w-1.5 rounded-full bg-[#5BA7FF]" />
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </motion.section>
  )
}
