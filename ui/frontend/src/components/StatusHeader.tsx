import { motion } from 'framer-motion'
import { Cpu, Gauge, Link2, RadioTower, ShieldCheck } from 'lucide-react'
import type { ReactNode } from 'react'

interface StatusHeaderProps {
  clockLabel: string
  dateLabel: string
  sessionLabel: string
  marketStatus: string
  gpuLabel: string
  gpuMemoryLabel: string
  modelLabel: string
  latencyLabel: string
  brokerLabel: string
  modeLabel: string
}

function Pill({
  icon,
  label,
  accent = 'blue',
}: {
  icon: ReactNode
  label: string
  accent?: 'blue' | 'green' | 'amber' | 'red'
}) {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
      className={`status-pill status-pill-${accent}`}
    >
      {icon}
      <span>{label}</span>
    </motion.div>
  )
}

export function StatusHeader(props: StatusHeaderProps) {
  return (
    <header className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-start">
      <div>
        <div className="font-mono text-[clamp(3.5rem,6vw,5.2rem)] font-semibold leading-none tracking-[-0.08em] text-white">
          {props.clockLabel}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-sm tracking-[0.24em] text-white/58 uppercase">
          <span>{props.dateLabel}</span>
          <span className="h-1 w-1 rounded-full bg-white/18" />
          <span>{props.sessionLabel}</span>
          <span className="h-1 w-1 rounded-full bg-white/18" />
          <span>{props.marketStatus}</span>
        </div>
      </div>

      <div className="flex flex-wrap justify-start gap-3 xl:justify-end">
        <Pill icon={<Cpu size={14} />} label={props.gpuLabel} accent="green" />
        <Pill icon={<Gauge size={14} />} label={props.gpuMemoryLabel} accent="blue" />
        <Pill icon={<RadioTower size={14} />} label={props.modelLabel} accent="blue" />
        <Pill icon={<Gauge size={14} />} label={props.latencyLabel} accent="amber" />
        <Pill icon={<Link2 size={14} />} label={props.brokerLabel} accent="green" />
        <Pill icon={<ShieldCheck size={14} />} label={props.modeLabel} accent="red" />
      </div>
    </header>
  )
}
