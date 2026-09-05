export type ChatRole = "system" | "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  /* Immagini allegate al turno, come data: URI. Restano sul messaggio e non
     dentro `content` perche' la cronologia le deve poter rimandare al server
     insieme al testo: un secondo turno che parla della foto senza la foto
     riceverebbe una risposta su niente. */
  images?: string[]
  /* Reasoning models stream their thinking on a separate delta field before
     the answer. Kept apart from `content` so it can be rendered separately
     and included in later prompts only when preserve_thinking is enabled. */
  reasoning?: string
}

export function withSystemPrompt(messages: ChatMessage[], prompt: string, enabled: boolean) {
  return enabled && prompt.trim()
    ? [{ id: "system-prompt", role: "system" as const, content: prompt }, ...messages]
    : messages
}

interface OpenAIError {
  error?: { message?: string }
}

export interface SchedulerHealth {
  active: boolean | number
  capacity?: number
  queued: number
  max_queue: number
  queue_timeout_seconds: number
  admitted: number
  completed: number
  rejected: number
  timed_out: number
  cancelled: number
}

export interface TiersHealth {
  vram: number
  ram: number
  disk: number
  vram_gb: number
  ram_gb: number
  /** page-cache residency of the memory-mapped expert container (qwen38) */
  mmap_resident_gb?: number
  mmap_total_gb?: number
}

export interface HwinfoHealth {
  cores: number
  ram_total_gb: number
  /** live on every poll since 2026-09-03: MemTotal-MemAvailable and the page cache (mapped experts live there) */
  ram_used_gb?: number
  ram_cache_gb?: number
  ram_avail_gb: number
  gpus: number
  vram_total_gb: number
  cpu: string
  gpu: string
}

/** qwen38's expert prefetcher (PFETCH line, P4/P7). Optional everywhere: an
 *  engine without a prefetcher never sends it and the panel stays hidden.
 *  Counters are cumulative since the engine started. */
export interface PrefetchHealth {
  hits: number
  miss: number
  demand_hits: number
  pinned_hits: number
  prefetch_hits: number
  issued: number
  completed: number
  evicted_unused: number
  pinned_experts: number
  hit_percent: number
}

export interface HealthResponse {
  status: string
  scheduler?: SchedulerHealth
  kv_slots?: number
  tiers?: TiersHealth
  hwinfo?: HwinfoHealth
  prefetch?: PrefetchHealth
  /** live prefill position of the in-flight request (absent when idle) */
  progress?: PrefillProgress
  /** the inference backend serving this gateway (colibri, llama.cpp, vLLM) */
  backend?: BackendHealth
}

export interface PrefillProgress {
  request_id: string
  phase: "prefix" | "prompt"
  prompt_tokens: number
  restored_tokens: number
  done_tokens: number
  remaining_tokens: number
  prefix_tokens: number
  elapsed_seconds: number
  tokens_per_second: number
  eta_seconds: number | null
}

export interface ServerSettings {
  model: string
  context_window: number
  max_output_tokens: number
  system_prompt: string
  prepend_system_prompt: boolean
  restart_supported?: boolean
  /** P9 items 6/7: the running backend, the persisted choice for the next
   *  restart and the backends this deployment has configured. */
  backend?: string
  backend_next?: string
  backends?: string[]
  /** set when the chosen backend failed to start and AI-DER serves instead */
  backend_error?: string
  /** vLLM launcher profile: the persisted choice (next start), the running one, the catalogue */
  vllm_profile?: string
  vllm_profile_active?: string | null
  vllm_profiles?: Record<string, { context: number; note: string }>
  /** dashboard "Extra": defaults for API clients that omit the field */
  api_defaults?: ApiDefaults
}

export interface ApiDefaults {
  reasoning?: boolean | null
  reasoning_effort?: string | null
  thinking_budget?: number | null
  preserve_thinking?: boolean | null
  temperature?: number | null
  top_p?: number | null
  temperature_thinking?: number | null
  temperature_instruct?: number | null
  top_p_thinking?: number | null
  top_p_instruct?: number | null
  speculative_decoding?: boolean | null
  gpu_router?: boolean | null
}

export interface BackendHealth {
  id: string
  label: string
  model: string
  upstream?: string
  context_window?: number
  /** P6.5: false when a proxy backend runs without its vision tower */
  images?: boolean
}

export interface RestartAccepted {
  status: "restarting"
  retry_after_seconds: number
}

export interface ProfileTurn {
  wall_s: number
  prompt_tokens: number
  completion_tokens: number
  expert_disk_s: number
  expert_wait_s: number
  expert_matmul_s: number
  attention_s: number
  lm_head_s: number
  forwards: number
}

export interface ProfileResponse {
  seq: number
  turns: ProfileTurn[]
}

export type TelemetryEventName =
  | "request_received"
  | "prefill_started"
  | "prefill_progress"
  | "prefill_finished"
  | "generation_finished"
  | "request_cancelled"
  | "request_failed"

export interface TelemetryEvent {
  seq: number
  ts: number
  event: TelemetryEventName
  request_id: string
  prompt_bytes?: number
  prompt_tokens?: number
  context_window?: number
  max_tokens?: number
  cache_slot?: number
  cached_tokens?: number
  prefilled_tokens?: number
  prefill_seconds?: number
  prefill_tokens_per_second?: number
  phase?: "prefix" | "prompt"
  restored_tokens?: number
  done_tokens?: number
  remaining_tokens?: number
  eta_seconds?: number | null
  elapsed_seconds?: number
  completion_tokens?: number
  tokens_per_second?: number
  expert_cache_hit_percent?: number
  rss_gb?: number
  length_limited?: boolean
  error?: string
}

export interface RuntimeUtilization {
  backend?: string
  cpu_percent?: number
  gpu_percent?: number
  vram_used_gb?: number
  vram_total_gb?: number
}

export interface TelemetryResponse {
  seq: number
  events: TelemetryEvent[]
  runtime: RuntimeUtilization
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface StreamChatResult {
  finishReason: string | null
  usage: TokenUsage | null
  requestId: string | null
  queueWaitMs: number | null
}

export function endpoint(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

export function serverEndpoint(baseUrl: string, path: string) {
  return endpoint(baseUrl.replace(/\/v1\/?$/, ""), path)
}

function headers(apiKey: string) {
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  }
}

async function responseError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    return body.error?.message || fallback
  } catch {
    return fallback
  }
}

export async function listModels(baseUrl: string, apiKey: string, signal?: AbortSignal) {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string }> }
  return (body.data || []).map((model) => model.id)
}

export async function getHealth(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "health"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as HealthResponse
}

export async function getProfile(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ProfileResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "profile"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ProfileResponse
}

export async function getTelemetry(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<TelemetryResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "telemetry"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as TelemetryResponse
}

export async function resetCache(baseUrl: string, apiKey: string, cacheSlot: number) {
  const response = await fetch(endpoint(baseUrl, `cache/slots/${cacheSlot}`), {
    method: "DELETE",
    headers: headers(apiKey),
  })
  if (!response.ok) throw new Error(await responseError(response))
}

export async function restartServer(baseUrl: string, apiKey: string): Promise<RestartAccepted> {
  const response = await fetch(endpoint(baseUrl, "admin/restart"), {
    method: "POST",
    headers: headers(apiKey),
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as RestartAccepted
}

export async function getServerSettings(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<ServerSettings> {
  const response = await fetch(endpoint(baseUrl, "settings"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ServerSettings
}

export async function updateServerSettings(
  baseUrl: string,
  apiKey: string,
  systemPrompt: string,
  prependSystemPrompt: boolean,
  signal?: AbortSignal,
): Promise<ServerSettings> {
  const response = await fetch(endpoint(baseUrl, "settings"), {
    method: "PATCH",
    headers: headers(apiKey),
    signal,
    body: JSON.stringify({
      system_prompt: systemPrompt,
      prepend_system_prompt: prependSystemPrompt,
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ServerSettings
}

export async function updateVllmProfile(
  baseUrl: string,
  apiKey: string,
  profile: string,
  signal?: AbortSignal,
): Promise<ServerSettings> {
  const response = await fetch(endpoint(baseUrl, "settings"), {
    method: "PATCH",
    headers: headers(apiKey),
    signal,
    body: JSON.stringify({ vllm_profile: profile }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ServerSettings
}

export async function updateApiDefaults(
  baseUrl: string,
  apiKey: string,
  apiDefaults: ApiDefaults,
  signal?: AbortSignal,
): Promise<ServerSettings> {
  const response = await fetch(endpoint(baseUrl, "settings"), {
    method: "PATCH",
    headers: headers(apiKey),
    signal,
    body: JSON.stringify({ api_defaults: apiDefaults }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ServerSettings
}

export async function updateServerBackend(
  baseUrl: string,
  apiKey: string,
  backend: string,
  signal?: AbortSignal,
): Promise<ServerSettings> {
  const response = await fetch(endpoint(baseUrl, "settings"), {
    method: "PATCH",
    headers: headers(apiKey),
    signal,
    body: JSON.stringify({ backend }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ServerSettings
}

export function extractSSE(buffer: string) {
  const frames = buffer.split(/\r?\n\r?\n/)
  const rest = frames.pop() || ""
  const data = frames.flatMap((frame) =>
    frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()),
  )
  return { data, rest }
}

export interface StreamChatOptions {
  baseUrl: string
  apiKey: string
  model: string
  messages: ChatMessage[]
  temperature: number
  maxTokens: number
  enableThinking: boolean
  preserveThinking: boolean
  reasoningBudget: number
  reasoningEffort?: string
  speculativeDecoding: boolean
  gpuRouter?: boolean
  cacheSlot?: number
  signal: AbortSignal
  onDelta: (text: string) => void
  onReasoning?: (text: string) => void
}

export async function streamChat(options: StreamChatOptions): Promise<StreamChatResult> {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: headers(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      /* Un turno con immagini viaggia nella forma a parti dell'API OpenAI;
         senza, resta la stringa di sempre e nessun server vede una differenza. */
      messages: options.messages.map(({ role, content, images, reasoning }) => ({
        role,
        content: images && images.length
          ? [
              ...(content ? [{ type: "text", text: content }] : []),
              ...images.map((url) => ({ type: "image_url", image_url: { url } })),
            ]
          : content,
        ...(options.preserveThinking && role === "assistant" && reasoning
          ? { reasoning_content: reasoning }
          : {}),
      })),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.enableThinking,
      ...(options.enableThinking && options.reasoningEffort ? { reasoning_effort: options.reasoningEffort } : {}),
      preserve_thinking: options.preserveThinking,
      thinking_budget: options.reasoningBudget,
      speculative_decoding: options.speculativeDecoding,
      gpu_router: options.gpuRouter !== false,
      ...(options.cacheSlot === undefined ? {} : { cache_slot: options.cacheSlot }),
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let finishReason: string | null = null
  let usage: TokenUsage | null = null

  const consume = (data: string) => {
    if (data === "[DONE]") return
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: string; reasoning_content?: string }; finish_reason?: string | null }>
      usage?: TokenUsage | null
    }
    const choice = event.choices?.[0]
    const text = choice?.delta?.content
    if (text) options.onDelta(text)
    const reasoning = choice?.delta?.reasoning_content
    if (reasoning) options.onReasoning?.(reasoning)
    if (choice?.finish_reason) finishReason = choice.finish_reason
    if (event.usage) usage = event.usage
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const parsed = extractSSE(buffer)
    buffer = parsed.rest
    parsed.data.forEach(consume)
    if (done) break
  }

  const queueWaitHeader = response.headers.get("x-colibri-queue-wait-ms")
  const parsedQueueWait = queueWaitHeader === null ? null : Number(queueWaitHeader)
  return {
    finishReason,
    usage,
    requestId: response.headers.get("x-request-id"),
    queueWaitMs: parsedQueueWait !== null && Number.isFinite(parsedQueueWait) ? parsedQueueWait : null,
  }
}
