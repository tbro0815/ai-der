import type { HealthResponse } from "./api"

export function decodeTokensPerSecond(tokens: number, firstTokenAtMs: number | null, nowMs: number): number | null {
  if (firstTokenAtMs === null || tokens < 2 || nowMs <= firstTokenAtMs) return null
  return (tokens - 1) * 1000 / (nowMs - firstTokenAtMs)
}

export function activeRequests(health: HealthResponse | null): number {
  return Number(health?.scheduler?.active || 0)
}

export function supportsCacheSlots(health: HealthResponse | null): boolean {
  return typeof health?.kv_slots === "number" && health.kv_slots > 0
}
