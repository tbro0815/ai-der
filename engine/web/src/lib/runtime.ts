import type { HealthResponse } from "./api"

export function decodeTokensPerSecond(tokens: number, firstTokenAtMs: number | null, nowMs: number): number | null {
  if (firstTokenAtMs === null || tokens < 2 || nowMs <= firstTokenAtMs) return null
  return (tokens - 1) * 1000 / (nowMs - firstTokenAtMs)
}

/** Whether the speed meter shows the gateway's API average instead of the
 *  dashboard's own last run: its last API run finished later (both on this clock). */
export function showApiSpeed(apiRunAt: number | null, ownRunAt: number | null): boolean {
  return apiRunAt !== null && (ownRunAt === null || apiRunAt > ownRunAt)
}

export function activeRequests(health: HealthResponse | null): number {
  return Number(health?.scheduler?.active || 0)
}

export function supportsCacheSlots(health: HealthResponse | null): boolean {
  return typeof health?.kv_slots === "number" && health.kv_slots > 0
}
