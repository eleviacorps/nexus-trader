import { motion } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'

interface ControlTileProps {
  label: string
  icon: LucideIcon
  active?: boolean
  tone?: 'green' | 'red' | 'blue' | 'amber' | 'neutral'
  onClick?: () => void
}

export function ControlTile({
  label,
  icon: Icon,
  active = false,
  tone = 'neutral',
  onClick,
}: ControlTileProps) {
  return (
    <motion.button
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.97 }}
      transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
      onClick={onClick}
      className={`control-tile control-tile-${tone} ${active ? 'control-tile-active' : ''}`}
      type="button"
    >
      <div className="control-tile-icon">
        <Icon size={18} strokeWidth={1.9} />
      </div>
      <span className="control-tile-label">{label}</span>
    </motion.button>
  )
}
