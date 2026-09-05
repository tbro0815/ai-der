import { describe, expect, it, vi } from "vitest"

import { loadPublicSettings, persistPublicSettings, type PublicSettings, type StringStorage } from "./storage"

function memoryStorage(initial: Record<string, string> = {}): StringStorage & { values: Map<string, string> } {
  const values = new Map(Object.entries(initial))
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value) },
    removeItem: (key) => { values.delete(key) },
  }
}

describe("browser settings persistence", () => {
  const settings: PublicSettings = {
    baseUrl: "https://localhost/v1",
    model: "test-model",
    temperatureThinking: 1,
    temperatureInstruct: 0.5,
    maxTokens: 32768,
    reasoningEffort: "xhigh",
    thinking: true,
    preserveThinking: true,
    reasoningBudget: 8192,
    speculativeDecoding: true,
    gpuRouter: true,
    cacheSlot: 2,
    autoScroll: false,
    systemPrompt: "Be concise.",
    useSystemPrompt: true,
    endpointSystemPrompt: true,
  }

  it("round-trips sidebar selections but removes legacy credentials", () => {
    const storage = memoryStorage({ "colibri.apiKey": "legacy-secret", "colibri.model": "old" })
    persistPublicSettings(storage, settings)
    expect(Object.fromEntries(storage.values)).toEqual({
      "aider.settings.v1": JSON.stringify({ ...settings, preserveThinkingV2: true }),
    })
    expect(loadPublicSettings(storage, { ...settings, model: "fallback" })).toEqual(settings)
  })
  it("resets preserveThinking saved under the old default, keeps it once re-saved", () => {
    const old = memoryStorage({ "aider.settings.v1": JSON.stringify({ preserveThinking: true }) })
    expect(loadPublicSettings(old, { ...settings, preserveThinking: false }).preserveThinking).toBe(false)
    const kept = memoryStorage({ "aider.settings.v1": JSON.stringify({ preserveThinking: true, preserveThinkingV2: true }) })
    expect(loadPublicSettings(kept, { ...settings, preserveThinking: false }).preserveThinking).toBe(true)
  })

  it("does not attempt to write an API key", () => {
    const storage = memoryStorage()
    const setItem = vi.spyOn(storage, "setItem")
    persistPublicSettings(storage, settings)
    expect(setItem).not.toHaveBeenCalledWith("colibri.apiKey", expect.anything())
  })

  it("uses defaults when storage access is unavailable", () => {
    expect(loadPublicSettings({ getItem: () => { throw new Error("denied") } }, settings)).toEqual(settings)
  })
})
