import * as AlertDialog from '@radix-ui/react-alert-dialog'
import * as Dialog from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import type { ReactNode } from 'react'
import { Button } from './button'

const overlay = 'fixed inset-0 z-50 bg-ink/40 backdrop-blur-[2px]'

export function Confirm({ open, onOpenChange, title, body, confirmLabel, onConfirm, busy }: {
  open: boolean; onOpenChange: (o: boolean) => void; title: string; body: ReactNode; confirmLabel: string; onConfirm: () => void; busy?: boolean
}) {
  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className={overlay} />
        <AlertDialog.Content className="fixed left-1/2 top-1/2 z-50 w-[calc(100vw-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-xl border bg-surface p-6 shadow-xl animate-rise">
          <AlertDialog.Title className="text-lg font-semibold tracking-tight">{title}</AlertDialog.Title>
          <AlertDialog.Description className="mt-2 text-sm text-soft">{body}</AlertDialog.Description>
          <div className="mt-6 flex justify-end gap-2">
            <AlertDialog.Cancel asChild><Button variant="outline">Cancel</Button></AlertDialog.Cancel>
            <Button variant="danger" loading={busy} onClick={onConfirm}>{confirmLabel}</Button>
          </div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  )
}

export function Drawer({ open, onOpenChange, title, description, children }: {
  open: boolean; onOpenChange: (o: boolean) => void; title: string; description?: string; children: ReactNode
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className={overlay} />
        <Dialog.Content className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l bg-surface shadow-xl animate-rise">
          <div className="flex items-start justify-between gap-4 border-b px-5 py-4">
            <div className="min-w-0">
              <Dialog.Title className="truncate text-base font-semibold tracking-tight">{title}</Dialog.Title>
              <Dialog.Description className="mt-0.5 text-sm text-soft">{description ?? 'Document details'}</Dialog.Description>
            </div>
            <Dialog.Close asChild><Button variant="ghost" size="icon" aria-label="Close"><X className="h-4 w-4" /></Button></Dialog.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-5">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
