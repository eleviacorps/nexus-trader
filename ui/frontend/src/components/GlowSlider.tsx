import { motion } from 'framer-motion'

interface GlowSliderProps {
  label: string
  valueLabel: string
  value: number
  min: number
  max: number
  step?: number
  tone?: 'green' | 'red' | 'blue' | 'amber'
  onChange: (value: number) => void
}

export function GlowSlider({
  label,
  valueLabel,
  value,
  min,
  max,
  step = 1,
  tone = 'green',
  onChange,
}: GlowSliderProps) {
  const fill = ((value - min) / Math.max(max - min, 1)) * 100

  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
      className="slider-card"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="text-[11px] tracking-[0.24em] text-white/42 uppercase">{label}</div>
        <div className={`font-mono text-sm font-semibold tone-${tone}`}>{valueLabel}</div>
      </div>
      <div className="mt-5">
        <div className={`glow-slider-track glow-slider-track-${tone}`}>
          <div className="glow-slider-fill" style={{ width: `${fill}%` }} />
        </div>
        <input
          className={`glow-slider glow-slider-${tone}`}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </div>
    </motion.div>
  )
}
