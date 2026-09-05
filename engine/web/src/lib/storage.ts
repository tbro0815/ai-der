export interface StringStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export interface PublicSettings {
  baseUrl: string
  model: string
  /** per-mode sampling (Qwen3.8 model card): thinking 1.0, instruct 0.7 */
  temperatureThinking: number
  temperatureInstruct: number
  maxTokens: number
  thinking: boolean
  /** Qwen3.8 reasoning effort: low | medium | xhigh (model default xhigh) */
  reasoningEffort: string
  preserveThinking: boolean
  /** dashboard web_search tool: Serper API key (stored here by the user's choice) and its switch */
  serperApiKey: string
  webSearch: boolean
  reasoningBudget: number
  speculativeDecoding: boolean
  gpuRouter: boolean
  cacheSlot: number
  autoScroll: boolean
  systemPrompt: string
  useSystemPrompt: boolean
  endpointSystemPrompt: boolean
}

const SETTINGS_KEY = "aider.settings.v1"

export function loadPublicSettings(
  storage: Pick<StringStorage, "getItem">,
  defaults: PublicSettings,
): PublicSettings {
  try {
    const raw = storage.getItem(SETTINGS_KEY)
    const saved = raw ? JSON.parse(raw) as Partial<PublicSettings> & { temperature?: number, preserveThinkingV2?: boolean } : {}
    const number = (value: unknown, fallback: number, min: number, max: number) =>
      typeof value === "number" && Number.isFinite(value)
        ? Math.min(max, Math.max(min, value))
        : fallback
    return {
      baseUrl: typeof saved.baseUrl === "string" && saved.baseUrl ? saved.baseUrl : storage.getItem("colibri.baseUrl") || defaults.baseUrl,
      model: typeof saved.model === "string" && saved.model ? saved.model : storage.getItem("colibri.model") || defaults.model,
      // the old single slider value migrates into the mode it was set in
      temperatureThinking: number(saved.temperatureThinking,
        typeof saved.temperature === "number" && saved.thinking === true ? saved.temperature : defaults.temperatureThinking, 0, 2),
      temperatureInstruct: number(saved.temperatureInstruct,
        typeof saved.temperature === "number" && saved.thinking !== true ? saved.temperature : defaults.temperatureInstruct, 0, 2),
      maxTokens: Math.round(number(saved.maxTokens, defaults.maxTokens, 1, 32768)),
      thinking: typeof saved.thinking === "boolean" ? saved.thinking : defaults.thinking,
      reasoningEffort: saved.reasoningEffort === "low" || saved.reasoningEffort === "medium" || saved.reasoningEffort === "xhigh" ? saved.reasoningEffort : defaults.reasoningEffort,
      // 2026-09-05: the default became "off"; a value saved under the old default is
      // reset once (the V2 marker is written with every save from then on)
      preserveThinking: saved.preserveThinkingV2 === true && typeof saved.preserveThinking === "boolean"
        ? saved.preserveThinking : defaults.preserveThinking,
      serperApiKey: typeof saved.serperApiKey === "string" ? saved.serperApiKey.trim() : defaults.serperApiKey,
      webSearch: typeof saved.webSearch === "boolean" ? saved.webSearch : defaults.webSearch,
      reasoningBudget: Math.round(number(saved.reasoningBudget, defaults.reasoningBudget, 0, 32768)),
      speculativeDecoding: typeof saved.speculativeDecoding === "boolean" ? saved.speculativeDecoding : defaults.speculativeDecoding,
      gpuRouter: typeof saved.gpuRouter === "boolean" ? saved.gpuRouter : defaults.gpuRouter,
      cacheSlot: Math.round(number(saved.cacheSlot, defaults.cacheSlot, 0, 15)),
      autoScroll: typeof saved.autoScroll === "boolean" ? saved.autoScroll : defaults.autoScroll,
      systemPrompt: typeof saved.systemPrompt === "string" ? saved.systemPrompt : defaults.systemPrompt,
      useSystemPrompt: typeof saved.useSystemPrompt === "boolean" ? saved.useSystemPrompt : defaults.useSystemPrompt,
      endpointSystemPrompt: typeof saved.endpointSystemPrompt === "boolean" ? saved.endpointSystemPrompt : defaults.endpointSystemPrompt,
    }
  } catch {
    return defaults
  }
}

export function persistPublicSettings(storage: StringStorage, settings: PublicSettings) {
  try {
    storage.setItem(SETTINGS_KEY, JSON.stringify({ ...settings, preserveThinkingV2: true }))
    // API credentials intentionally remain memory-only. Remove values left by
    // older web releases and the two superseded single-value settings.
    storage.removeItem("colibri.apiKey")
    storage.removeItem("colibri.baseUrl")
    storage.removeItem("colibri.model")
  } catch { /* restricted storage mode */ }
}
