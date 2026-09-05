import { describe, expect, it } from "vitest"

import { telemetryEventText } from "./RuntimeLog"

describe("telemetryEventText", () => {
  it("reports prompt-cache reuse and uncached prefill throughput", () => {
    expect(telemetryEventText({
      seq: 3, ts: 1, event: "prefill_finished", request_id: "7",
      prompt_tokens: 100, cached_tokens: 75, prefilled_tokens: 25,
      prefill_seconds: 1, prefill_tokens_per_second: 25,
    })).toBe("prefill finished · 25 processed · 75 cached (75.0%) · 25.0 tok/s · 1.0s")
  })

  it("reports generation size, inference rate, and expert-cache use", () => {
    expect(telemetryEventText({
      seq: 4, ts: 2, event: "generation_finished", request_id: "7",
      completion_tokens: 64, tokens_per_second: 13.25,
      expert_cache_hit_percent: 96.5, rss_gb: 101.2, length_limited: false,
    })).toContain("64 tokens · 13.3 tok/s · expert cache 96.5%")
  })
})
