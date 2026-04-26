import { motion } from 'framer-motion'
import { ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'

import { GlassCard } from './GlassCard'

interface HeroStat {
  label: string
  value: string
  tone?: 'green' | 'red' | 'blue' | 'amber' | 'neutral'
}

interface HeroCardProps {
  title: string
  subtitle: string
  icon: ReactNode
  primaryValue: string
  primaryLabel: string
  stats: HeroStat[]
  accent?: 'green' | 'red' | 'blue' | 'amber'
}

export function HeroCard({
  title,
  subtitle,
  icon,
  primaryValue,
  primaryLabel,
  stats,
  accent = 'blue',
}: HeroCardProps) {
  return (
    <GlassCard
      className={`hero-card hero-card-${accent}`}
      title={title}
      subtitle={subtitle}
      icon={icon}
      aside={
        <motion.div whileHover={{ x: 2 }} className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-white/48">
          <ChevronRight size={16} />
        </motion.div>
      }
    >
      <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_minmax(260px,0.9fr)] xl:items-end">
        <div>
          <div className="font-mono text-[clamp(2.3rem,4vw,4.2rem)] font-semibold leading-none tracking-[-0.08em] text-white">
            {primaryValue}
          </div>
          <p className="mt-3 text-sm tracking-[0.22em] text-white/44 uppercase">
            {primaryLabel}
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {stats.map((stat) => (
            <div key={stat.label} className="stat-chip">
              <div className="text-[11px] tracking-[0.24em] text-white/38 uppercase">
                {stat.label}
              </div>
              <div className={`mt-2 font-mono text-xl font-semibold ${stat.tone ? `tone-${stat.tone}` : 'text-white/90'}`}>
                {stat.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </GlassCard>
  )
}
