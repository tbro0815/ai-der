import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  ArrowUp,
  BrainCircuit,
  CircleStop,
  Clock,
  Cpu,
  Database,
  Gauge,
  Globe,
  HardDrive,
  Server,
  KeyRound,
  AlertTriangle,
  Layers,
  Link2,
  LoaderCircle,
  MemoryStick,
  ImagePlus,
  MessageSquareText,
  MonitorDot,
  RefreshCw,
  ScrollText,
  SlidersHorizontal,
  Timer,
  Trash2,
  Waypoints,
  X,
  Zap,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { type PrefillProgress, getHealth, getServerSettings, listModels, resetCache, restartServer, streamChat, updateApiDefaults, updateServerBackend, updateVllmProfile, updateServerSettings, withSystemPrompt, type ChatMessage, type HealthResponse, type ServerSettings, type StreamChatResult } from "@/lib/api"
import { activeRequests, decodeTokensPerSecond, supportsCacheSlots } from "@/lib/runtime"
import { Brain } from "./Brain"
import { Profiling } from "./Profiling"
import { RuntimeLog } from "./RuntimeLog"
import { Attribution } from "./Attribution"
import { loadPublicSettings, persistPublicSettings, type PublicSettings } from "@/lib/storage"
import { Markdown } from "@/components/Markdown"
import { cn } from "@/lib/utils"
import { useLocale } from "./i18n"

const message = (role: ChatMessage["role"], content: string): ChatMessage => {
  let id: string
  try { id = crypto.randomUUID() } catch { id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16) }) }
  return { id, role, content }
}

const DEFAULT_SYSTEM_PROMPT = `You are AI-DER a concise reasoning agent. When thinking (inside <think>…</think> blocks), you must use highly compressed, dense, and telegraphic logic.

- Your thinking budget is a maximum of 4096 tokens if needed, keep reasoning as short as possible though. Don't overthink.
- Eliminate all conversational filler, preambles, and meta-commentary (e.g., do not say "Let me double check this" or "Now I will look at option B". Verification steps stay mandatory, express them minimally instead (e.g. 'Check: 3×7=21 ✓').
- Express self-correction using direct symbols like "X -> Y" or "Correction: [Fact]".
- Use short bullet points and fragments instead of full paragraphs.
- Compress your line of thought as much as possible while maintaining the exact same rigor of self-questioning.
- After </think>: answer directly. No restating the question, no summary of your reasoning, no hedging preamble, no closing offer. Code without surrounding explanation unless asked. Target the minimal complete answer.`

const formatEta = (seconds: number) => {
  const total = Math.max(0, Math.round(seconds))
  return total >= 60 ? `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")} min` : `${total}s`
}

export default function App() {
  const { t, locale, setLocale, locales } = useLocale()

  const servedByEngine = typeof window !== "undefined" && window.location.port !== "5173" && window.location.protocol.startsWith("http")
  const defaultBase = servedByEngine ? `${window.location.origin}/v1` : "http://127.0.0.1:8000/v1"
  const [initialSettings] = useState(() => {
    const defaults: PublicSettings = {
      baseUrl: defaultBase,
      model: "qwen3.8-flash-next-aider",
      temperatureThinking: 1.0,
      temperatureInstruct: 0.7,
      maxTokens: 32768,
      thinking: false,
      reasoningEffort: "xhigh",
      preserveThinking: false,
      reasoningBudget: 8192,
      speculativeDecoding: true,
      gpuRouter: true,
      cacheSlot: 0,
      autoScroll: true,
      systemPrompt: DEFAULT_SYSTEM_PROMPT,
      useSystemPrompt: true,
      endpointSystemPrompt: false,
    }
    const saved = loadPublicSettings(localStorage, defaults)
    if (servedByEngine && saved.baseUrl === "http://127.0.0.1:8000/v1" && defaultBase !== saved.baseUrl) {
      return { ...saved, baseUrl: defaultBase }
    }
    return saved
  })
  const [baseUrl, setBaseUrl] = useState(initialSettings.baseUrl)
  const [apiKey, setApiKey] = useState("")
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState(initialSettings.model)
  const [temperatureThinking, setTemperatureThinking] = useState(initialSettings.temperatureThinking)
  const [temperatureInstruct, setTemperatureInstruct] = useState(initialSettings.temperatureInstruct)
  const [maxTokens, setMaxTokens] = useState(initialSettings.maxTokens)
  const [thinking, setThinking] = useState(initialSettings.thinking)
  const [reasoningEffort, setReasoningEffort] = useState(initialSettings.reasoningEffort)
  /* the sampling preset follows the reasoning toggle (Qwen3.8 model card):
     thinking 1.0 / top_p 0.95, instruct 0.7 / top_p 0.80 / presence 1.5; top_k 20 */
  const temperature = thinking ? temperatureThinking : temperatureInstruct
  const setTemperature = thinking ? setTemperatureThinking : setTemperatureInstruct
  const [preserveThinking, setPreserveThinking] = useState(initialSettings.preserveThinking)
  const [reasoningBudget, setReasoningBudget] = useState(initialSettings.reasoningBudget)
  const [speculativeDecoding, setSpeculativeDecoding] = useState(initialSettings.speculativeDecoding)
  const [gpuRouter, setGpuRouter] = useState(initialSettings.gpuRouter)
  const [cacheSlot, setCacheSlot] = useState(initialSettings.cacheSlot)
  const [autoScroll, setAutoScroll] = useState(initialSettings.autoScroll)
  const [systemPrompt, setSystemPrompt] = useState(initialSettings.systemPrompt)
  const [useSystemPrompt, setUseSystemPrompt] = useState(initialSettings.useSystemPrompt)
  const [endpointSystemPrompt, setEndpointSystemPrompt] = useState(initialSettings.endpointSystemPrompt)
  const [serverSettings, setServerSettingsRaw] = useState<ServerSettings | null>()
  /* Safari was observed ending up with `undefined` here after the settings
     PATCH resolved, hiding everything gated on the settings object.  Accept
     only real objects; anything else is logged and ignored, and the restart
     capability is remembered from the first response that carried it. */
  const [restartSupported, setRestartSupported] = useState(false)
  const setServerSettings = (value: ServerSettings | null | undefined) => {
    if (value === null) { setServerSettingsRaw(null); return }
    if (!value || typeof value !== "object") {
      console.warn("[ai-der] ignoring non-object server settings response", value)
      return
    }
    setServerSettingsRaw(value)
    if (typeof value.restart_supported === "boolean") setRestartSupported(value.restart_supported)
  }
  const [conversations, setConversations] = useState<Record<number, ChatMessage[]>>({ 0: [] })
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState("")
  const [lastRun, setLastRun] = useState<StreamChatResult | null>(null)
  const [draft, setDraft] = useState("")
  /* Immagini in attesa di partire col prossimo messaggio. Si tengono come
     data: URI perche' e' quello che il server accetta e quello che il browser
     puo' mostrare in anteprima senza inventarsi un percorso su disco. */
  const [attachments, setAttachments] = useState<{ name: string; url: string }[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  /* P6.5: the AI-DER engine has its own tower; a proxy backend reports
     `images` when it was started with one (llama.cpp --mmproj, vLLM VISION=1) */
  const imagesSupported = !health?.backend || health.backend.images !== false
  const attachFiles = async (files: FileList | File[] | null) => {
    if (!imagesSupported) { setError("chat.imagesUnsupported"); return }
    if (!files) return
    const all = Array.from(files)
    const images = all.filter((file) => file.type.startsWith("image/"))
    /* an audio attachment gets its own message, whether alone or next to
       images (the images still attach); other non-image files a generic one */
    if (all.some((file) => file.type.startsWith("audio/"))) setError("chat.audioUnsupported")
    else if (all.length && !images.length) setError("chat.attachNotImage")
    if (!images.length) return
    const read = await Promise.all(
      images.map(
        (file) =>
          new Promise<{ name: string; url: string }>((resolve, reject) => {
            const reader = new FileReader()
            reader.onload = () => resolve({ name: file.name, url: String(reader.result) })
            reader.onerror = () => reject(reader.error)
            reader.readAsDataURL(file)
          }),
      ),
    )
    setAttachments((current) => [...current, ...read])
  }
  const [loading, setLoading] = useState(false)
  const [tokenCount, setTokenCount] = useState(0)
  const [tokPerSec, setTokPerSec] = useState<number | null>(null)
  const [ttft, setTtft] = useState<number | null>(null)
  /* P9 preparation UX: while a turn waits for its first token, poll the live
     prefill position so the wait is explained (unseen stable prefix vs prompt,
     tokens done, ETA) instead of a silent spinner. */
  const [progress, setProgress] = useState<PrefillProgress | null>(null)
  useEffect(() => {
    if (!loading || ttft !== null) { setProgress(null); return }
    let disposed = false
    const poll = async () => {
      try {
        const result = await getHealth(baseUrl, apiKey)
        if (!disposed) setProgress(result.progress ?? null)
      } catch { /* health errors are reported by the main poll */ }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 1000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [loading, ttft, baseUrl, apiKey])
  const [totalTokens, setTotalTokens] = useState({ prompt: 0, completion: 0 })
  const [connecting, setConnecting] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [switchingBackend, setSwitchingBackend] = useState(false)
  const backendLabel = (id: string) => ({ aider: "AI-DER", llamacpp: "llama.cpp", vllm: "vLLM" } as Record<string, string>)[id] || id
  const chooseBackend = async (backend: string) => {
    if (!serverSettings || backend === serverSettings.backend_next) return
    setSwitchingBackend(true)
    try {
      setServerSettings(await updateServerBackend(baseUrl, apiKey, backend))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSwitchingBackend(false)
    }
  }
  const chooseVllmProfile = async (profile: string) => {
    if (!serverSettings || profile === serverSettings.vllm_profile) return
    if (profile === "fast" && !window.confirm(t("sidebar.vllmFastConfirm"))) return
    setSwitchingBackend(true)
    try {
      setServerSettings(await updateVllmProfile(baseUrl, apiKey, profile))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSwitchingBackend(false)
    }
  }
  const [connected, setConnected] = useState(false)
  const [view, setView] = useState<"chat" | "brain" | "profiling" | "log">("chat")
  /* The Profiling pane renders the engine's per-turn PROF phase lines, which
     the qwen38 engine does not emit (nor do the proxy backends); it stays in
     the tree for development and is hidden from users until it carries data. */
  const SHOW_PROFILING = false
  /* The Brain pane draws the expert map only the AI-DER engine emits; the
     proxy backends (llama.cpp, vLLM) have none, so the tab is hidden there
     and the view falls back to the chat when the backend changes. */
  const showBrain = !health?.backend || health.backend.id === "aider"
  useEffect(() => { if (!showBrain && view === "brain") setView("chat") }, [showBrain, view])
  const [attributionOpen, setAttributionOpen] = useState(false)
  const [error, setError] = useState("")
  const autoConnected = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  const probeRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const messages = conversations[cacheSlot] || []
  const kvSlots = Math.max(1, health?.kv_slots || 1)
  const active = activeRequests(health)
  const capacity = health?.scheduler?.capacity || kvSlots
  const failures = health?.scheduler ? health.scheduler.rejected + health.scheduler.timed_out + health.scheduler.cancelled : 0
  const outputCeiling = Math.min(32768, serverSettings?.max_output_tokens || 32768)
  const supportsServerSettings = serverSettings !== undefined && serverSettings !== null

  const updateMessages = (next: ChatMessage[] | ((current: ChatMessage[]) => ChatMessage[])) =>
    setConversations((current) => ({
      ...current,
      [cacheSlot]: typeof next === "function" ? next(current[cacheSlot] || []) : next,
    }))

  // EFFECT #1
  useEffect(() => {
    persistPublicSettings(localStorage, {
      baseUrl, model, temperatureThinking, temperatureInstruct, maxTokens, thinking, reasoningEffort, preserveThinking, reasoningBudget, cacheSlot, autoScroll,
      speculativeDecoding, gpuRouter, systemPrompt, useSystemPrompt, endpointSystemPrompt,
    })
  }, [autoScroll, baseUrl, cacheSlot, endpointSystemPrompt, gpuRouter, maxTokens, model, preserveThinking, reasoningBudget, reasoningEffort, speculativeDecoding, systemPrompt, temperatureInstruct, temperatureThinking, thinking, useSystemPrompt])

  // EFFECT #2
  useEffect(() => {
    setConnected(false)
    setHealth(null)
    setServerSettings(undefined)
    setHealthError("")
  }, [baseUrl, apiKey])

  // EFFECT #3
  useEffect(() => () => {
    probeRef.current?.abort()
    abortRef.current?.abort()
  }, [])

  // EFFECT #4
  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getHealth(baseUrl, apiKey)
        if (!disposed) { setHealth(result); setHealthError("") }
      } catch (cause) {
        if (!disposed) setHealthError(cause instanceof Error ? cause.message : "status.runtimeUnavailable")
      }
    }
    const timer = window.setInterval(() => void poll(), 5000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [apiKey, baseUrl, connected])

  // EFFECT #5
  useEffect(() => {
    if (cacheSlot >= kvSlots) setCacheSlot(0)
  }, [cacheSlot, kvSlots])

  // EFFECT #6
  useEffect(() => { setLastRun(null) }, [cacheSlot])

  // EFFECT #7
  useEffect(() => {
    if (autoScroll) bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [autoScroll, messages])

  // The sidebar's sampling and reasoning values are also the server's defaults
  // for API clients that omit the field (an explicit client value wins);
  // "Apply to all API clients" below concerns the system prompt only.
  useEffect(() => {
    if (!connected || !supportsServerSettings) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      void updateApiDefaults(baseUrl, apiKey, {
        reasoning: thinking, reasoning_effort: reasoningEffort, thinking_budget: reasoningBudget,
        preserve_thinking: preserveThinking, temperature: null,
        temperature_thinking: temperatureThinking, temperature_instruct: temperatureInstruct,
        speculative_decoding: speculativeDecoding, gpu_router: gpuRouter,
      }, controller.signal)
        .then(setServerSettings)
        .catch((cause) => {
          if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "status.serverError")
        })
    }, 300)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [apiKey, baseUrl, connected, gpuRouter, preserveThinking, reasoningBudget, reasoningEffort, speculativeDecoding, supportsServerSettings, temperatureInstruct, temperatureThinking, thinking])

  // The gateway persists this setting; the small debounce avoids writing its
  // config file on every keystroke in the system-prompt textarea.
  useEffect(() => {
    if (!connected || !supportsServerSettings) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      void updateServerSettings(baseUrl, apiKey, systemPrompt, endpointSystemPrompt, controller.signal)
        .then(setServerSettings)
        .catch((cause) => {
          if (!controller.signal.aborted) {
            setServerSettings(null)
            setError(cause instanceof Error ? cause.message : "status.serverError")
          }
        })
    }, 300)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [apiKey, baseUrl, connected, endpointSystemPrompt, supportsServerSettings, systemPrompt])

  const connect = async () => {
    probeRef.current?.abort()
    const controller = new AbortController()
    probeRef.current = controller
    setConnecting(true)
    setError("")
    try {
      const found = await listModels(baseUrl, apiKey, controller.signal)
      setModels(found)
      if (found.length && !found.includes(model)) setModel(found[0])
      try {
        const settings = await getServerSettings(baseUrl, apiKey, controller.signal)
        setServerSettings(settings)
        setSystemPrompt(settings.system_prompt)
        setEndpointSystemPrompt(settings.prepend_system_prompt)
      } catch {
        setServerSettings(null)
      }
      setConnected(true)
      try {
        setHealth(await getHealth(baseUrl, apiKey, controller.signal))
        setHealthError("")
      } catch (cause) {
        if (!controller.signal.aborted) {
          setHealth(null)
          setHealthError(cause instanceof Error ? cause.message : "status.runtimeUnavailable")
        }
      }
    } catch (cause) {
      if (controller.signal.aborted) return
      setConnected(false)
      setError(cause instanceof Error ? cause.message : "status.serverError")
    } finally {
      if (probeRef.current === controller) { probeRef.current = null; setConnecting(false) }
    }
  }

  const restart = async () => {
    if (!window.confirm(t("sidebar.restartConfirm"))) return
    setRestarting(true)
    setError("")
    try {
      const accepted = await restartServer(baseUrl, apiKey)
      setConnected(false)
      setHealth(null)
      setError("status.restarting")
      window.setTimeout(() => { setRestarting(false); void connect() }, accepted.retry_after_seconds * 1000)
    } catch (cause) {
      setRestarting(false)
      setError(cause instanceof Error ? cause.message : "status.serverError")
    }
  }

  // Auto-connect once when the UI is served by the engine itself. In an
  // effect, not in the render body: a side effect during render breaks under
  // StrictMode double-rendering and concurrent re-renders.
  useEffect(() => {
    if (servedByEngine && !autoConnected.current && !connected) {
      autoConnected.current = true
      connect()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servedByEngine, connected])

  const canSend = useMemo(() => draft.trim() && model && !loading, [draft, loading, model])

  const clear = async () => {
    setError("")
    updateMessages([])
    setTokPerSec(null)
    setTtft(null)
    setTokenCount(0)
    setTotalTokens({ prompt: 0, completion: 0 })
    setLastRun(null)
    try {
      await resetCache(baseUrl, apiKey, cacheSlot)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "status.serverError")
    }
  }

  const send = async () => {
    const content = draft.trim()
    /* Un'immagine da sola e' una domanda valida: "questa cosa e'?" si puo'
       chiedere anche senza scrivere niente. */
    if ((!content && !attachments.length) || loading) return
    const user = message("user", content)
    if (attachments.length) user.images = attachments.map((item) => item.url)
    const assistant = message("assistant", "")
    const history = [...messages, user]
    /* a proxy backend cannot take the conversation's earlier images (it would
       answer 400 on every turn): send the text only and say so once per send */
    const imagesOmitted = !imagesSupported && history.some((item) => item.images && item.images.length)
    const requestMessages = withSystemPrompt(
      imagesOmitted ? history.map((item) => (item.images && item.images.length ? { ...item, images: undefined } : item)) : history,
      systemPrompt, useSystemPrompt)
    setDraft("")
    setAttachments([])
    setError(imagesOmitted ? "chat.imagesOmitted" : "")
    updateMessages([...history, assistant])
    setLoading(true)
    setTokenCount(0)
    setTokPerSec(null)
    setTtft(null)
    const t0 = performance.now()
    let firstTokenAt: number | null = null
    let count = 0
    const controller = new AbortController()
    abortRef.current = controller
    const recordToken = () => {
      const now = performance.now()
      if (firstTokenAt === null) {
        firstTokenAt = now
        setTtft(now - t0)
      }
      count++
      setTokenCount(count)
      const rate = decodeTokensPerSecond(count, firstTokenAt, now)
      if (rate !== null) setTokPerSec(rate)
    }
    try {
      const result = await streamChat({
        baseUrl,
        apiKey,
        model,
        messages: requestMessages,
        temperature,
        maxTokens,
        enableThinking: thinking,
        reasoningEffort,
        preserveThinking,
        reasoningBudget,
        speculativeDecoding,
        gpuRouter,
        cacheSlot: supportsCacheSlots(health) ? cacheSlot : undefined,
        signal: controller.signal,
        /* Reasoning tokens are tokens: they count toward the rate, and the
           first one is the real time-to-first-token — the answer's first
           token arrives much later on a reasoning model. */
        onReasoning: (delta) => {
          recordToken()
          updateMessages((current) => current.map((item) =>
            item.id === assistant.id ? { ...item, reasoning: (item.reasoning ?? "") + delta } : item,
          ))
        },
        onDelta: (delta) => {
          recordToken()
          updateMessages((current) => current.map((item) =>
            item.id === assistant.id ? { ...item, content: item.content + delta } : item,
          ))
        },
      })
      const finalRate = decodeTokensPerSecond(
        result.usage?.completion_tokens ?? count, firstTokenAt, performance.now(),
      )
      if (finalRate !== null) setTokPerSec(finalRate)
      if (result.usage) setTotalTokens(prev => ({
        prompt: prev.prompt + (result.usage?.prompt_tokens || 0),
        completion: prev.completion + (result.usage?.completion_tokens || 0),
      }))
      setLastRun(result)
      setConnected(true)
    } catch (cause) {
      if (controller.signal.aborted) {
        updateMessages((current) => current.filter((item) => item.id !== assistant.id || item.content || item.reasoning))
      } else {
        setError(cause instanceof Error ? cause.message : "status.generationFailed")
        updateMessages((current) => current.filter((item) => item.id !== assistant.id || item.content || item.reasoning))
      }
    } finally {
      abortRef.current = null
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark"><Waypoints className="size-5" /></div>
          <div><h1>{t("brand.name")}</h1><p>{t("brand.subtitle")}</p><p className="brand-tagline">{t("brand.tagline")}</p></div>
        </div>

        <section className="side-section">
          <div className="section-title"><Link2 className="size-3.5" /> {t("sidebar.connection")}</div>
          <label>{t("sidebar.endpoint")}<Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} /></label>
          <label>{t("sidebar.apiKey")}<div className="relative"><KeyRound className="field-icon" /><Input className="pl-9" type="password" value={apiKey} placeholder={t("sidebar.apiKeyPlaceholder")} onChange={(event) => setApiKey(event.target.value)} /></div><span className="field-help">{t("sidebar.apiKeyHelp")}</span></label>
          <Button type="button" variant="secondary" onClick={connect} disabled={connecting}>
            {connecting ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
            {t("sidebar.probe")}
          </Button>
          <div className={cn("connection-state", connected && "connected")} aria-live="polite"><span />{connected ? t("status.connected") : t("status.notConnected")}</div>
        </section>

        <section className="side-section runtime-section" aria-live="polite">
          <div className="section-title"><Activity className="size-3.5" /> {t("sidebar.runtime")}</div>
          {health?.hwinfo ? <div className="hw-panel">
            {health.hwinfo.cpu ? <div className="hw-row"><Cpu className="size-3.5" /><span>{health.hwinfo.cpu}</span></div> : null}
            {health.hwinfo.gpus > 0 ? <div className="hw-row"><MonitorDot className="size-3.5" /><span>{health.hwinfo.gpu || "GPU"}<small>{health.hwinfo.gpus}× · {health.hwinfo.vram_total_gb.toFixed(0)} GB VRAM</small></span></div> : null}
            <div className="hw-row"><MemoryStick className="size-3.5" /><span>{health.hwinfo.ram_total_gb.toFixed(0)} GB usable RAM<small>{typeof health.hwinfo.ram_used_gb === "number" && typeof health.hwinfo.ram_cache_gb === "number"
              ? `${health.hwinfo.ram_used_gb.toFixed(0)} GB in use · ${health.hwinfo.ram_cache_gb.toFixed(0)} GB page cache`
              : `${health.hwinfo.ram_avail_gb.toFixed(0)} GB free`}</small></span></div>
            <div className="hw-row"><HardDrive className="size-3.5" /><span>{health.hwinfo.cores} cores</span></div>
            {health.backend ? <div className="hw-row"><Server className="size-3.5" /><span>{health.backend.label}<small>{health.backend.model}</small></span></div> : null}
          </div> : null}
          {health?.scheduler ? <>
            <div className="runtime-grid">
              <div><span>{t("dashboard.active")}</span><strong>{active}<small> / {capacity}</small></strong></div>
              <div><span>{t("dashboard.queued")}</span><strong>{health.scheduler.queued}<small> / {health.scheduler.max_queue}</small></strong></div>
              <div><span>{t("dashboard.completed")}</span><strong>{health.scheduler.completed}</strong></div>
              <div><span>{t("dashboard.failures")}</span><strong>{failures}</strong></div>
            </div>
            {health.tiers ? (() => {
              const ti = health.tiers
              // qwen38 keeps no RAM copy of experts (ram is always 0): the bar is
              // VRAM-resident vs memory-mapped, and a second band shows how much
              // of the mapped container the page cache currently holds.
              const total = Math.max(ti.vram + ti.ram + ti.disk, 1)
              const mm = ti.mmap_total_gb && ti.mmap_resident_gb !== undefined ? ti : null
              return <div className="tier-panel">
                <div className="tier-bar" role="img" aria-label={t("tier.ariaLabel", { vram: ti.vram, disk: ti.disk })}>
                  <span className="tier-vram" style={{ width: `${(100 * ti.vram) / total}%` }} />
                  <span className="tier-disk" style={{ width: `${(100 * (ti.disk + ti.ram)) / total}%` }} />
                </div>
                <div className="tier-legend">
                  <span><i className="tier-vram" />{t("tier.vram")} <strong>{ti.vram.toLocaleString()}</strong><small>{ti.vram_gb.toFixed(1)} GB</small></span>
                  <span><i className="tier-disk" />{t("tier.disk")} <strong>{(ti.disk + ti.ram).toLocaleString()}</strong></span>
                </div>
                {mm ? <>
                  <div className="tier-bar" role="img" aria-label={t("tier.mmapAriaLabel", { res: mm.mmap_resident_gb!.toFixed(1), total: mm.mmap_total_gb!.toFixed(1) })}>
                    <span className="tier-ram" style={{ width: `${Math.min(100, (100 * mm.mmap_resident_gb!) / mm.mmap_total_gb!)}%` }} />
                  </div>
                  <div className="tier-legend">
                    <span><i className="tier-ram" />{t("tier.mmap")} <strong>{mm.mmap_resident_gb!.toFixed(1)}</strong><small>/ {mm.mmap_total_gb!.toFixed(1)} GB</small></span>
                  </div>
                </> : null}
              </div>
            })() : null}
            {/* Expert-tier hit rate, split by WHY the expert was already in
                VRAM. Only engines with a prefetcher report it (qwen38), so the
                whole block is absent otherwise rather than showing zeros. */}
            {health.prefetch ? (() => {
              const pf = health.prefetch
              const hits = Math.max(pf.hits, 1)
              return <div className="tier-panel">
                <div className="tier-bar" role="img" aria-label={t("prefetch.ariaLabel", { hit: pf.hit_percent.toFixed(1) })}>
                  <span className="tier-vram" style={{ width: `${(100 * pf.demand_hits) / hits}%` }} />
                  <span className="tier-ram" style={{ width: `${(100 * pf.pinned_hits) / hits}%` }} />
                  <span className="tier-disk" style={{ width: `${(100 * pf.prefetch_hits) / hits}%` }} />
                </div>
                <div className="tier-legend">
                  <span><i className="tier-vram" />{t("prefetch.demand")} <strong>{pf.demand_hits.toLocaleString()}</strong></span>
                  <span><i className="tier-ram" />{t("prefetch.pinned")} <strong>{pf.pinned_hits.toLocaleString()}</strong><small>{pf.pinned_experts}/layer</small></span>
                  <span><i className="tier-disk" />{t("prefetch.prefetched")} <strong>{pf.prefetch_hits.toLocaleString()}</strong><small>{pf.completed.toLocaleString()} up</small></span>
                </div>
                <div className="runtime-foot">{t("prefetch.hitRate")} <code>{pf.hit_percent.toFixed(1)}%</code></div>
              </div>
            })() : null}
            {totalTokens.prompt + totalTokens.completion > 0 ? <div className="session-stats">
              <span><Database className="size-3" /> {t("dashboard.session")} <strong>{totalTokens.prompt.toLocaleString()}</strong> {t("dashboard.prompt")} + <strong>{totalTokens.completion.toLocaleString()}</strong> {t("dashboard.completion")}</span>
            </div> : null}
            <div className="runtime-foot"><span className="runtime-dot" /> {t("sidebar.schedulerOnline")} <code>{kvSlots} KV</code></div>
          </> : <p className="runtime-unavailable">{connected ? (healthError ? t(healthError) : t("status.runtimeUnavailable")) : t("sidebar.runtimeProbe")}</p>}
        </section>

        <section className="side-section">
          <div className="section-title"><SlidersHorizontal className="size-3.5" /> {t("sidebar.inference")}</div>
          {/* the model id is fixed per deployment (one public id on every backend);
              the runtime panel names what is serving, so the selector is gone */}
          {health?.kv_slots && health.kv_slots > 1 ? <label>{t("sidebar.kvSession")}<select value={cacheSlot} onChange={(event) => setCacheSlot(Number(event.target.value))} disabled={loading}>
            {Array.from({ length: kvSlots }, (_, slot) => <option key={slot} value={slot}>{t("sidebar.sessionLabel", { slot: slot + 1 })}</option>)}
          </select><span className="field-help">{t("sidebar.kvSessionHelp")}</span></label> : null}
          <label><span className="label-line"><span>{t("sidebar.temperature")}</span><code>{temperature.toFixed(1)}</code></span><input className="range" type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
          <label>{t("sidebar.maxTokens")}<Input type="number" min={1} max={outputCeiling} value={maxTokens} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value)) { const next = Math.min(outputCeiling, Math.max(1, Math.round(value))); setMaxTokens(next); setReasoningBudget((current) => Math.min(current, next)) } }} /></label>
          <button type="button" className={cn("toggle-row", thinking && "active")} aria-pressed={thinking} onClick={() => setThinking((value) => !value)}>
            <span><BrainCircuit className="size-4" /> {t("sidebar.reasoning")}</span><i><b /></i>
          </button>
          <label>{t("sidebar.reasoningEffort")}<select value={reasoningEffort} disabled={!thinking} onChange={(event) => setReasoningEffort(event.target.value)}>
            {["low", "medium", "xhigh"].map((level) => <option key={level} value={level}>{t(`sidebar.effort.${level}`)}</option>)}
          </select><span className="field-help">{t("sidebar.reasoningEffortHelp")}</span></label>
          <button type="button" className={cn("toggle-row", preserveThinking && "active")} aria-pressed={preserveThinking} onClick={() => setPreserveThinking((value) => !value)}>
            <span><BrainCircuit className="size-4" /> {t("sidebar.preserveThinking")}</span><i><b /></i>
          </button>
          <label>{t("sidebar.reasoningBudget")}<Input type="number" min={0} max={maxTokens} value={reasoningBudget} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value)) setReasoningBudget(Math.min(maxTokens, Math.max(0, Math.round(value)))) }} /><span className="field-help">{t("sidebar.reasoningBudgetHelp")}</span></label>
          <button type="button" className={cn("toggle-row", speculativeDecoding && "active")} aria-pressed={speculativeDecoding} onClick={() => setSpeculativeDecoding((value) => !value)}>
            <span><Zap className="size-4" /> {t("sidebar.speculativeDecoding")}</span><i><b /></i>
          </button>
          <span className="field-help">{t("sidebar.speculativeDecodingHelp")}</span>
          <button type="button" className={cn("toggle-row", gpuRouter && "active")} aria-pressed={gpuRouter} onClick={() => setGpuRouter((value) => !value)}>
            <span><Waypoints className="size-4" /> {t("sidebar.gpuRouter")}</span><i><b /></i>
          </button>
          <span className="field-help">{t("sidebar.gpuRouterHelp")}</span>
          <button type="button" className={cn("toggle-row", autoScroll && "active")} aria-pressed={autoScroll} onClick={() => setAutoScroll((value) => !value)}>
            <span><ScrollText className="size-4" /> {t("sidebar.autoScrolling")}</span><i><b /></i>
          </button>
        </section>

        <section className="side-section">
          <div className="section-title"><MessageSquareText className="size-3.5" /> {t("sidebar.systemPrompt")}</div>
          <label>{t("sidebar.systemPromptNewChats")}<Textarea className="system-prompt" value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} placeholder={t("sidebar.systemPromptPlaceholder")} /><span className="field-help">{t("sidebar.systemPromptHelp")}</span></label>
          <button type="button" className={cn("toggle-row", useSystemPrompt && "active")} aria-pressed={useSystemPrompt} onClick={() => setUseSystemPrompt((value) => !value)}>
            <span><MessageSquareText className="size-4" /> {t("sidebar.useSystemPrompt")}</span><i><b /></i>
          </button>
          <span className="field-help">{t("sidebar.useSystemPromptHelp")}</span>
          <button type="button" className={cn("toggle-row", endpointSystemPrompt && "active")} aria-pressed={endpointSystemPrompt} disabled={connected && serverSettings === null} onClick={() => setEndpointSystemPrompt((value) => !value)}>
            <span><Link2 className="size-4" /> {t("sidebar.endpointSystemPrompt")}</span><i><b /></i>
          </button>
          <span className="field-help">{serverSettings === null ? t("sidebar.endpointSettingsUnavailable") : t("sidebar.endpointSystemPromptHelp")}</span>
        </section>

        <details className="side-section extra-section">
          <summary className="section-title"><SlidersHorizontal className="size-3.5" /> {t("sidebar.extra")}</summary>
          <dl className="endpoint-settings">
            <div><dt>base_url</dt><dd>{baseUrl}</dd></div>
            <div><dt>model id</dt><dd>{model}</dd></div>
            <div><dt>context_window</dt><dd>{(serverSettings?.context_window || 131072).toLocaleString()}</dd></div>
            <div><dt>max_tokens</dt><dd>{outputCeiling.toLocaleString()}</dd></div>
            <div><dt>temperature</dt><dd>{temperature.toFixed(1)}</dd></div>
            <div><dt>top_p</dt><dd>{thinking ? "0.95" : "0.80"}</dd></div>
            <div><dt>top_k</dt><dd>20</dd></div>
            <div><dt>presence_penalty</dt><dd>{thinking ? "0.0" : "1.5"}</dd></div>
            <div><dt>enable_thinking</dt><dd>{String(thinking)}</dd></div>
            <div><dt>reasoning_effort</dt><dd>{thinking ? reasoningEffort : "—"}</dd></div>
            <div><dt>preserve_thinking</dt><dd>{String(preserveThinking)}</dd></div>
            <div><dt>thinking_budget</dt><dd>{reasoningBudget.toLocaleString()}</dd></div>
            <div><dt>speculative_decoding</dt><dd>{String(speculativeDecoding)}</dd></div>
            <div><dt>gpu_router</dt><dd>{String(gpuRouter)}</dd></div>
            <div><dt>cache_slot</dt><dd>{cacheSlot}</dd></div>
            <div><dt>restart_supported</dt><dd>{restartSupported ? "true" : serverSettings === undefined ? "loading" : serverSettings === null ? "unavailable" : "false"}</dd></div>
            <div><dt>backend</dt><dd>{backendLabel(health?.backend ? health.backend.id : serverSettings?.backend || "—")}</dd></div>
          </dl>
          <span className="field-help">{t("sidebar.extraDefaultsHelp")}</span>
          {serverSettings?.backends && serverSettings.backends.length > 1 ? (
            <label className="backend-choice">{t("sidebar.backend")}
              <select value={serverSettings.backend_next || serverSettings.backend} disabled={switchingBackend} onChange={(event) => void chooseBackend(event.target.value)}>
                {serverSettings.backends.map((id) => <option key={id} value={id}>{backendLabel(id)}</option>)}
              </select>
              <span className="field-help">{serverSettings.backend_error ? `${t("sidebar.backendFailed")} ${serverSettings.backend_error}` : serverSettings.backend_next && serverSettings.backend_next !== serverSettings.backend ? t("sidebar.backendPending") : t("sidebar.backendHelp")}</span>
            </label>
          ) : null}
          {serverSettings?.backends?.includes("vllm") && serverSettings.vllm_profiles ? (
            <label className="backend-choice">{t("sidebar.vllmProfile")}
              <select value={serverSettings.vllm_profile || "long"} disabled={switchingBackend} onChange={(event) => void chooseVllmProfile(event.target.value)}>
                {Object.entries(serverSettings.vllm_profiles).map(([id, info]) => <option key={id} value={id}>{id} · {Math.round(info.context / 1024)}K</option>)}
              </select>
              <span className="field-help">{serverSettings.backend === "vllm" && serverSettings.vllm_profile_active && serverSettings.vllm_profile_active !== (serverSettings.vllm_profile || "long") ? t("sidebar.vllmProfilePending") : t("sidebar.vllmProfileHelp")}</span>
            </label>
          ) : null}
          {restartSupported ? (
            <div className="restart-action">
              <Button type="button" variant="destructive" size="sm" onClick={() => void restart()} disabled={restarting || loading}>
                {restarting ? <LoaderCircle className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
                {t(restarting ? "sidebar.restartingServer" : "sidebar.restartServer")}
              </Button>
              <span className="field-help">{t("sidebar.restartServerHelp")}</span>
            </div>
          ) : null}
        </details>

        <div className="sidebar-foot">
          <div><Cpu className="size-3.5" /><span>{t("sidebar.transport")}</span></div>
          <div>
            <button type="button" className="attribution-link" onClick={() => setAttributionOpen(true)}>
              <ScrollText className="size-3.5" /><span>{t("attribution.link")}</span>
            </button>
          </div>
          <div className="locale-switcher">
            <Globe className="size-3.5" />
            <select value={locale} onChange={(e) => setLocale(e.target.value)}>
              {locales.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
            </select>
          </div>
        </div>
      </aside>

      <main className="chat-panel">
        <header className="topbar">
          <div><span className="eyebrow">{t("topbar.activeModel")}</span><strong>{health?.backend?.model || model}</strong></div>
          <div className="view-tabs">
            <button className={view === "chat" ? "active" : ""} onClick={() => setView("chat")}><MessageSquareText className="size-3.5" /> {t("nav.chat")}</button>
            {showBrain ? <button className={view === "brain" ? "active" : ""} onClick={() => setView("brain")}><BrainCircuit className="size-3.5" /> {t("nav.brain")}</button> : null}
            {SHOW_PROFILING ? <button className={view === "profiling" ? "active" : ""} onClick={() => setView("profiling")}><Gauge className="size-3.5" /> {t("nav.profiling")}</button> : null}
            <button className={view === "log" ? "active" : ""} onClick={() => setView("log")}><ScrollText className="size-3.5" /> {t("nav.log")}</button>
          </div>
          <div className="top-actions">
              {loading && tokenCount > 0 ? <Badge className="badge-live"><Zap className="size-3 flash" /> {t("topbar.tokens", { n: tokenCount })}</Badge> : null}
              {!loading && tokPerSec != null ? <Badge className="badge-speed"><Gauge className="size-3" /> {t("topbar.tokPerSec", { n: tokPerSec.toFixed(1) })}</Badge> : null}
              {!loading && ttft != null ? <Badge><Timer className="size-3" /> TTFT {(ttft/1000).toFixed(1)}s</Badge> : null}
              {!loading && lastRun?.usage ? <Badge><Layers className="size-3" /> {lastRun.usage.prompt_tokens}→{lastRun.usage.completion_tokens}</Badge> : null}
              {!loading && lastRun?.finishReason === "length" ? <Badge className="badge-warn" title={t("topbar.truncatedHelp")}><AlertTriangle className="size-3" /> {t("topbar.truncated")}</Badge> : null}
              {lastRun?.queueWaitMs != null ? <Badge><Clock className="size-3" /> queue {Math.round(lastRun.queueWaitMs)}ms</Badge> : null}
              <Badge><MonitorDot className="size-3" /> {t("topbar.slot", { n: cacheSlot + 1 })}</Badge>
              <Button variant="ghost" size="sm" onClick={() => void clear()} disabled={!connected || loading}><Trash2 className="size-3.5" /> {t("topbar.clear")}</Button>
            </div>
        </header>

        {view === "brain" && showBrain ? <Brain baseUrl={baseUrl} apiKey={apiKey} connected={connected} />
          : view === "profiling" && SHOW_PROFILING ? <Profiling baseUrl={baseUrl} apiKey={apiKey} connected={connected} />
          : view === "log" ? <RuntimeLog baseUrl={baseUrl} apiKey={apiKey} connected={connected} /> : <>

        <div className="conversation">
          {!messages.length ? (
            <div className="empty-state">
              <div className="orb"><Waypoints /></div>
              <h2 className="empty-wordmark">{t("brand.name")}</h2>
              <div className="suggestions">
                {[t("prompts.routing"), t("prompts.benchmark"), t("prompts.caching")].map((item) => <button key={item} onClick={() => setDraft(item)}>{item}<ArrowUp className="size-3.5 rotate-45" /></button>)}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((item) => (
                <article key={item.id} className={cn("message", item.role)}>
                  <div className="avatar">{item.role === "user" ? "Y" : <Waypoints className="size-4" />}</div>
                  <div><div className="message-meta">{item.role === "user" ? t("chat.you") : t("chat.assistant")}</div><div className="message-body">{item.reasoning
                    ? <details className="reasoning" open={!item.content}>
                        <summary>{t("sidebar.reasoning")}</summary>
                        <div className="reasoning-body">{item.reasoning}</div>
                      </details>
                    : null}{item.content
                    ? (item.role === "assistant"
                        /* User turns stay literal: the person typed those
                           characters and expects to see them back. Only the
                           model's output is markdown. */
                        ? <Markdown source={item.content} />
                        : item.content)
                    : <span className="typing" aria-label="Generating"><i /><i /><i /></span>}</div>
                  {item.images && item.images.length ? (
                    /* the attachments of that turn, kept on the message (they are
                       resent with the history) and shown where they were sent */
                    <div className="message-images">
                      {item.images.map((url, index) => (
                        <a key={index} href={url} target="_blank" rel="noreferrer"><img src={url} alt={`attachment ${index + 1}`} /></a>
                      ))}
                    </div>
                  ) : null}</div>
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {error && <div className="error-banner" role="alert">{t(error)}</div>}
          <div className="composer">
            <Textarea value={draft}
              onPaste={(event) => { const files = Array.from(event.clipboardData.files); if (files.length) { event.preventDefault(); void attachFiles(files) } }}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => { if (event.dataTransfer.files.length) { event.preventDefault(); void attachFiles(event.dataTransfer.files) } }} onChange={(event) => setDraft(event.target.value)} placeholder={t("chat.placeholder")} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send() } }} />
            {attachments.length > 0 && (
              <div className="attachments">
                {attachments.map((item, index) => (
                  <span key={item.url + index} className="attachment">
                    <img src={item.url} alt={item.name} />
                    <button type="button" aria-label={t("chat.removeImage")}
                      onClick={() => setAttachments((current) => current.filter((_, at) => at !== index))}>x</button>
                  </span>
                ))}
              </div>
            )}
            {loading && ttft === null ? (
              <div className="prep-status" role="status" aria-live="polite">
                <Timer className="size-3.5" />
                <span>{progress
                  ? (progress.phase === "prefix" ? t("prep.prefix") : t("prep.prompt"))
                    + " · " + t("prep.progress", {
                        done: progress.done_tokens.toLocaleString(),
                        total: progress.prompt_tokens.toLocaleString(),
                        rate: progress.tokens_per_second.toFixed(1),
                        eta: progress.eta_seconds != null ? formatEta(progress.eta_seconds) : "?" })
                    + (progress.restored_tokens > 0 ? " · " + t("prep.restored", { n: progress.restored_tokens.toLocaleString() }) : "")
                  : t("prep.waiting")}</span>
              </div>
            ) : null}
            <div className="composer-foot"><span><MessageSquareText className="size-3.5" /> {t("chat.inputHint")}</span><input ref={fileInputRef} type="file" accept="image/*" multiple hidden onChange={(event) => { void attachFiles(event.target.files); event.target.value = "" }} /><Button variant="ghost" size="icon" aria-label={imagesSupported ? t("chat.attachImage") : t("chat.imagesUnsupported")} title={imagesSupported ? undefined : t("chat.imagesUnsupported")} disabled={!imagesSupported} onClick={() => fileInputRef.current?.click()}><ImagePlus className="size-4" /></Button>{loading ? <Button variant="destructive" size="icon" aria-label={t("chat.stop")} onClick={() => abortRef.current?.abort()}><CircleStop className="size-4" /></Button> : <Button size="icon" aria-label={t("chat.send")} disabled={!canSend} onClick={() => void send()}><ArrowUp className="size-4" /></Button>}</div>
          </div>
        </div>
        </>}
      </main>

      {attributionOpen ? <Attribution onClose={() => setAttributionOpen(false)} /> : null}
    </div>
  )
}
