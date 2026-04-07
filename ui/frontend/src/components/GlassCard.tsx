import { motion } from 'framer-motion'
import { clsx } from 'clsx'
import type { PropsWithChildren, ReactNode } from 'react'

interface GlassCardProps extends PropsWithChildren {
  title?: string
  subtitle?: string
  icon?: ReactNode
  aside?: ReactNode
  className?: string
  contentClassName?: string
  interactive?: boolean
}

export function GlassCard({
  title,
  subtitle,
  icon,
  aside,
  className,
  contentClassName,
  interactive = true,
  children,
}: GlassCardProps) {
  return (
    <motion.section
      whileHover={interactive ? { y: -3 } : undefined}
      transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
      className={clsx('glass-card relative overflow-hidden', className)}
    >
      {(title || subtitle || icon || aside) && (
        <div className="mb-5 flex items-start justify-between gap-4">
          <div className="flex min-w-0 items-start gap-4">
            {icon ? <div className="icon-badge shrink-0">{icon}</div> : null}
            <div className="min-w-0">
              {title ? (
                <h2 className="text-base font-semibold tracking-[0.22em] text-white/92 uppercase">
                  {title}
                </h2>
              ) : null}
              {subtitle ? (
                <p className="mt-1 max-w-3xl text-sm leading-6 text-white/62">
                  {subtitle}
                </p>
              ) : null}
            </div>
          </div>
          {aside ? <div className="shrink-0">{aside}</div> : null}
        </div>
      )}
      <div className={clsx('relative z-[1]', contentClassName)}>{children}</div>
    </motion.section>
  )
}
