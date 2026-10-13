import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { cn } from '@/lib/utils'
import { Card, CardHeader } from './ui/primitives'

export interface Series { key: string; label: string; color: string }

/** Grouped bar chart following the validated two-slot palette (blue, then orange). A table view is always
 *  available so no value depends on colour or hover alone. */
export function BarChartCard({ title, description, data, series, format = (n) => String(n), height, domain, horizontal = false }: {
  title: string; description?: string; data: Record<string, string | number | null>[]; series: Series[]; format?: (n: number) => string; height?: number; domain?: [number, number]; horizontal?: boolean
}) {
  const [table, setTable] = useState(false)
  // Horizontal bars keep long category names readable; height grows with the number of rows.
  const h = height ?? (horizontal ? Math.max(200, data.length * (series.length * 18 + 18) + 48) : 280)
  const toggle = 'rounded-md px-2.5 py-1 text-xs font-medium'
  return (
    <Card>
      <CardHeader title={title} description={description} action={
        <div role="group" aria-label="View" className="flex rounded-md border p-0.5">
          <button className={cn(toggle, !table ? 'bg-raised text-ink' : 'text-soft')} aria-pressed={!table} onClick={() => setTable(false)}>Chart</button>
          <button className={cn(toggle, table ? 'bg-raised text-ink' : 'text-soft')} aria-pressed={table} onClick={() => setTable(true)}>Table</button>
        </div>} />
      <div className="p-4">
        {table ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="border-b text-left text-soft"><th className="py-2 pr-4 font-medium">Name</th>{series.map((s) => <th key={s.key} className="py-2 pr-4 text-right font-medium">{s.label}</th>)}</tr></thead>
              <tbody className="divide-y">{data.map((d) => <tr key={String(d.name)}><td className="py-2 pr-4">{d.name}</td>{series.map((s) => <td key={s.key} className="py-2 pr-4 text-right font-mono tabular-nums">{typeof d[s.key] === 'number' ? format(d[s.key] as number) : '–'}</td>)}</tr>)}</tbody>
            </table>
          </div>
        ) : (
          <div style={{ height: h }} role="img" aria-label={`${title}. Switch to Table for the exact values.`}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'} margin={{ top: 8, right: 16, left: 0, bottom: 4 }} barGap={2} barCategoryGap="24%">
                <CartesianGrid horizontal={horizontal ? false : true} vertical={horizontal ? true : false} stroke="hsl(var(--line))" strokeDasharray="3 3" />
                {horizontal ? (
                  <>
                    <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: 'hsl(var(--soft))', fontSize: 12 }} tickFormatter={format} domain={domain} />
                    <YAxis type="category" dataKey="name" tickLine={false} axisLine={{ stroke: 'hsl(var(--line))' }} tick={{ fill: 'hsl(var(--soft))', fontSize: 12 }} width={178} interval={0} />
                  </>
                ) : (
                  <>
                    <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: 'hsl(var(--line))' }} tick={{ fill: 'hsl(var(--soft))', fontSize: 12 }} interval={0} />
                    <YAxis tickLine={false} axisLine={false} tick={{ fill: 'hsl(var(--soft))', fontSize: 12 }} tickFormatter={format} width={64} domain={domain} />
                  </>
                )}
                <Tooltip cursor={{ fill: 'hsl(var(--raised))' }} formatter={(v: number) => format(v)}
                  contentStyle={{ background: 'hsl(var(--surface))', border: '1px solid hsl(var(--line))', borderRadius: 8, fontSize: 13, color: 'hsl(var(--ink))' }} labelStyle={{ color: 'hsl(var(--ink))', fontWeight: 600 }} />
                {series.length > 1 && <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: 'hsl(var(--soft))' }} />}
                {series.map((s) => <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]} barSize={series.length > 1 ? 14 : 20} isAnimationActive={false} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </Card>
  )
}
