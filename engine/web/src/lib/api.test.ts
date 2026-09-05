import { afterEach, describe, expect, it, vi } from "vitest"

import { extractSSE, getHealth, getProfile, getServerSettings, getTelemetry, resetCache, restartServer, serverEndpoint, streamChat, updateServerSettings, withSystemPrompt } from "./api"

afterEach(() => vi.unstubAllGlobals())

describe("extractSSE", () => {
  it("keeps an incomplete frame for the next network chunk", () => {
    const parsed = extractSSE('data: {"choices":[]}\n\ndata: {"cho')
    expect(parsed.data).toEqual(['{"choices":[]}'])
    expect(parsed.rest).toBe('data: {"cho')
  })

  it("supports CRLF and multiple data frames", () => {
    const parsed = extractSSE("data: one\r\n\r\ndata: two\r\n\r\n")
    expect(parsed.data).toEqual(["one", "two"])
    expect(parsed.rest).toBe("")
  })
})

describe("runtime API", () => {
  it.each([
    ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/health"],
    ["https://example.test/api/v1/", "https://example.test/api/health"],
    ["https://example.test/api", "https://example.test/api/health"],
  ])("resolves the health endpoint outside the OpenAI v1 prefix", (baseUrl, expected) => {
    expect(serverEndpoint(baseUrl, "health")).toBe(expected)
  })

  it("requests health with the configured bearer credential", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "ok", scheduler: { active: true } })))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getHealth("http://localhost:8000/v1/", "secret")).resolves.toMatchObject({ status: "ok" })
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/health", expect.objectContaining({
      headers: expect.objectContaining({ Authorization: "Bearer secret" }),
    }))
  })

  it("requests the profiling history next to the OpenAI v1 prefix", async () => {
    const turn = {
      wall_s: 2.5, prompt_tokens: 7, completion_tokens: 12,
      expert_disk_s: 0.4, expert_wait_s: 0.1, expert_matmul_s: 0.9,
      attention_s: 0.6, lm_head_s: 0.2, forwards: 15,
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ seq: 1, turns: [turn] })))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getProfile("http://localhost:8000/v1/")).resolves.toEqual({ seq: 1, turns: [turn] })
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/profile", expect.anything())
  })

  it("requests the live telemetry log next to the OpenAI v1 prefix", async () => {
    const payload = {
      seq: 1,
      events: [{ seq: 1, ts: 1, event: "prefill_started", request_id: "7" }],
      runtime: { cpu_percent: 12.5, gpu_percent: 88 },
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(payload)))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getTelemetry("http://localhost:8000/v1/", "secret")).resolves.toEqual(payload)
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/telemetry",
      expect.objectContaining({ headers: expect.objectContaining({ Authorization: "Bearer secret" }) }),
    )
  })

  it("deletes the selected cache slot", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
    vi.stubGlobal("fetch", fetchMock)

    await resetCache("http://localhost:8000/v1", "secret", 3)
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/v1/cache/slots/3",
      expect.objectContaining({
        method: "DELETE",
        headers: expect.objectContaining({ Authorization: "Bearer secret" }),
      }),
    )
  })

  it("requests a supervised server restart", async () => {
    const accepted = { status: "restarting", retry_after_seconds: 35 }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(accepted), { status: 202 }))
    vi.stubGlobal("fetch", fetchMock)

    await expect(restartServer("http://localhost:8000/v1", "secret")).resolves.toEqual(accepted)
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/v1/admin/restart",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ Authorization: "Bearer secret" }),
      }),
    )
  })

  it("reads and updates the authenticated server settings resource", async () => {
    const settings = {
      model: "test-model", context_window: 131072, max_output_tokens: 32768,
      system_prompt: "Be concise.", prepend_system_prompt: true, restart_supported: true,
    }
    const fetchMock = vi.fn().mockImplementation(
      () => Promise.resolve(new Response(JSON.stringify(settings))),
    )
    vi.stubGlobal("fetch", fetchMock)

    await expect(getServerSettings("http://localhost:8000/v1", "secret")).resolves.toEqual(settings)
    await expect(updateServerSettings("http://localhost:8000/v1", "secret", "Be concise.", true)).resolves.toEqual(settings)
    expect(fetchMock.mock.calls[1]).toEqual([
      "http://localhost:8000/v1/settings",
      expect.objectContaining({
        method: "PATCH",
        headers: expect.objectContaining({ Authorization: "Bearer secret" }),
        body: JSON.stringify({ system_prompt: "Be concise.", prepend_system_prompt: true }),
      }),
    ])
  })
})

describe("chat request extensions", () => {
  const completedStream = () => new Response("data: [DONE]\n\n", {
    headers: { "content-type": "text/event-stream" },
  })

  async function requestBody(cacheSlot?: number, preserveThinking = true, reasoningBudget = 8) {
    const fetchMock = vi.fn().mockResolvedValue(completedStream())
    vi.stubGlobal("fetch", fetchMock)
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [{ id: "assistant-1", role: "assistant", content: "Answer", reasoning: "work" }],
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      preserveThinking,
      reasoningBudget,
      speculativeDecoding: true,
    gpuRouter: true,
      cacheSlot,
      signal: new AbortController().signal,
      onDelta: () => undefined,
    })
    return JSON.parse(fetchMock.mock.calls[0][1].body as string) as Record<string, unknown>
  }

  it("only prepends the configured system prompt when enabled", () => {
    const messages = [{ id: "user-1", role: "user" as const, content: "Hello" }]
    expect(withSystemPrompt(messages, "Be concise.", false)).toBe(messages)
    expect(withSystemPrompt(messages, "Be concise.", true)).toEqual([
      { id: "system-prompt", role: "system", content: "Be concise." },
      ...messages,
    ])
  })

  it("omits cache_slot for a generic OpenAI-compatible backend", async () => {
    expect(await requestBody()).not.toHaveProperty("cache_slot")
  })

  it("sends cache_slot zero when colibrì advertises KV slots", async () => {
    expect(await requestBody(0)).toMatchObject({ cache_slot: 0 })
  })

  it("sends the Qwen reasoning controls", async () => {
    expect(await requestBody()).toMatchObject({
      preserve_thinking: true,
      thinking_budget: 8,
      speculative_decoding: true,
      gpu_router: true,
      messages: [{ reasoning_content: "work" }],
    })
  })

  it("omits historical reasoning when preserve_thinking is off", async () => {
    const body = await requestBody(undefined, false, 0)
    expect(body).toMatchObject({ preserve_thinking: false, thinking_budget: 0 })
    expect(body.messages).toEqual([{ role: "assistant", content: "Answer" }])
  })
})
