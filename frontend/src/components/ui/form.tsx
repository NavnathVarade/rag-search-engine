import { forwardRef, useId, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

const field = 'w-full rounded-md border bg-surface px-3 text-sm text-ink placeholder:text-faint transition-colors hover:border-faint/60 disabled:opacity-50'

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input ref={ref} className={cn(field, 'h-9', className)} {...p} />
))
Input.displayName = 'Input'

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...p }, ref) => (
  <textarea ref={ref} className={cn(field, 'min-h-[120px] py-2 leading-relaxed', className)} {...p} />
))
Textarea.displayName = 'Textarea'

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(({ className, ...p }, ref) => (
  <select ref={ref} className={cn(field, 'h-9 pr-8', className)} {...p} />
))
Select.displayName = 'Select'

/** Label + control + optional hint, wired together with ids for screen readers. */
export function Field({ label, hint, children }: { label: string; hint?: string; children: (id: string, describedBy?: string) => ReactNode }) {
  const id = useId()
  const hintId = hint ? `${id}-hint` : undefined
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      {children(id, hintId)}
      {hint && <p id={hintId} className="text-xs text-faint">{hint}</p>}
    </div>
  )
}
