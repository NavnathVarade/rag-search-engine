import { cva, type VariantProps } from 'class-variance-authority'
import { Loader2 } from 'lucide-react'
import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        primary: 'bg-accent text-accent-ink hover:bg-accent/90',
        outline: 'border bg-surface text-ink hover:bg-raised',
        ghost: 'text-soft hover:bg-raised hover:text-ink',
        danger: 'bg-bad text-white hover:bg-bad/90',
      },
      size: { sm: 'h-8 px-3', md: 'h-9 px-4', lg: 'h-11 px-5 text-[15px]', icon: 'h-9 w-9' },
    },
    defaultVariants: { variant: 'primary', size: 'md' },
  },
)

interface Props extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> { loading?: boolean }

export const Button = forwardRef<HTMLButtonElement, Props>(({ className, variant, size, loading, children, disabled, ...p }, ref) => (
  <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} disabled={disabled || loading} {...p}>
    {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
    {children}
  </button>
))
Button.displayName = 'Button'
