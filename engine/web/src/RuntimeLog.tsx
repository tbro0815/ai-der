import { useEffect, useRef, useState } from "react"
import { Cpu, MemoryStick, MonitorDot, ScrollText } from "lucide-react"

import { getTelemetry, type RuntimeUtilization, type TelemetryEvent } from "@/lib/api"
import { useLocale } from "./i18n"

/* proxy backends report some counts as null (no timings from upstream) */
const integer = (value?: number | null) => (value ?? 0).toLocaleString()
const decimal = (value?: number | null) => (value ?? 0).toFixed(1)

export function telemetryEventText(event: TelemetryEvent) {
  switch (event.event) {
    case "request_received":
      return `message received · ${decimal((event.prompt_bytes || 0) / 1024)} KB · max ${integer(event.max_tokens)} output · slot ${(event.cache_slot || 0) + 1}`
    case "prefill_started":
      return `prefill started · context ${integer(event.prompt_tokens)} / ${integer(event.context_window)} · max ${integer(event.max_tokens)} output`
    case "prefill_progress":
      return `${event.phase === "prefix" ? "engine preparation (unseen stable prefix)" : "prefill"} · ${integer(event.done_tokens)} / ${integer(event.prompt_tokens)} tokens · ${decimal(event.tokens_per_second)} tok/s · ETA ${event.eta_seconds != null ? Math.round(event.eta_seconds) + "s" : "?"}`
    case "prefill_finished": {
      const cache = event.prompt_tokens ? 100 * (event.cached_tokens || 0) / event.prompt_tokens : 0
      return `prefill finished · ${integer(event.prefilled_tokens)} processed · ${integer(event.cached_tokens)} cached (${decimal(cache)}%) · ${decimal(event.prefill_tokens_per_second)} tok/s · ${decimal(event.prefill_seconds)}s`
    }
    case "generation_finished":
      return `generation finished · ${integer(event.completion_tokens)} tokens · ${decimal(event.tokens_per_second)} tok/s · expert cache ${decimal(event.expert_cache_hit_percent)}% · ${decimal(event.rss_gb)} GB RSS${event.length_limited ? " · limit reached" : ""}`
    case "request_cancelled":
      return "generation cancelled"
    case "request_failed":
      return `request failed · ${event.error || "engine error"}`
  }
}

const eventLabel = (event: TelemetryEvent) => ({
  request_received: "MESSAGE",
  prefill_started: "PREFILL",
  prefill_progress: "PREFILL",
  prefill_finished: "PREFILL",
  generation_finished: "DECODE",
  request_cancelled: "CANCEL",
  request_failed: "ERROR",
}[event.event])

export function RuntimeLog({ baseUrl, apiKey, connected }: { baseUrl: string; apiKey: string; connected: boolean }) {
  const { t } = useLocale()
  const [events, setEvents] = useState<TelemetryEvent[]>([])
  const [runtime, setRuntime] = useState<RuntimeUtilization>({})
  const streamRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getTelemetry(baseUrl, apiKey)
        if (!disposed) { setEvents(result.events); setRuntime(result.runtime || {}) }
      } catch { /* engine restarting — keep the last useful log */ }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 1000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [apiKey, baseUrl, connected])

  useEffect(() => {
    if (streamRef.current) streamRef.current.scrollTop = streamRef.current.scrollHeight
  }, [events])

  return (
    <div className="log-page">
      <div className="log-head">
        <div className="section-title"><ScrollText className="size-4" /> {t("log.title")}</div>
        <div className="log-runtime" aria-label="Current hardware utilization">
          <span><Cpu className="size-3" /> {t("log.cpu")} <strong>{runtime.cpu_percent == null ? "—" : `${decimal(runtime.cpu_percent)}%`}</strong></span>
          <span><MonitorDot className="size-3" /> {t("log.gpu")} <strong>{runtime.gpu_percent == null ? "—" : `${decimal(runtime.gpu_percent)}%`}</strong></span>
          <span><MemoryStick className="size-3" /> {t("log.vram")} <strong>{runtime.vram_used_gb == null ? "—" : `${decimal(runtime.vram_used_gb)} / ${decimal(runtime.vram_total_gb)} GB`}</strong></span>
        </div>
      </div>
      <div className="log-stream" ref={streamRef} role="log" aria-live="polite">
        {!events.length ? <p className="runtime-unavailable">{connected ? t("log.empty") : t("log.connectHint")}</p> : (
          <div className="log-lines">
            {events.map((event) => (
              <div className={`log-line ${event.event}`} key={event.seq}>
                <time>{new Date(event.ts * 1000).toLocaleTimeString([], { hour12: false })}</time>
                <b>{eventLabel(event)}</b>
                <code>#{event.request_id}</code>
                <span>{telemetryEventText(event)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
