import { describe, expect, it } from "vitest"

import { PROMPT_CARDS, pickSuggestions } from "./prompts"

describe("suggestion cards", () => {
  it("keeps exactly 50 cards with unique ids and texts", () => {
    expect(PROMPT_CARDS).toHaveLength(50)
    expect(new Set(PROMPT_CARDS.map((c) => c.id)).size).toBe(50)
    expect(new Set(PROMPT_CARDS.map((c) => c.text)).size).toBe(50)
    for (const card of PROMPT_CARDS) {
      expect(card.id).toMatch(/^[a-z0-9-]+$/)
      expect(card.text.length).toBeGreaterThan(15)
      expect(card.text.length).toBeLessThan(110)
    }
  })

  it("picks three distinct cards", () => {
    const picked = pickSuggestions()
    expect(picked).toHaveLength(3)
    expect(new Set(picked.map((c) => c.id)).size).toBe(3)
  })

  it("is driven by the injected random source", () => {
    const first = pickSuggestions(3, () => 0)
    expect(first.map((c) => c.id)).toEqual(PROMPT_CARDS.slice(0, 3).map((c) => c.id))
    const last = pickSuggestions(3, () => 0.999999)
    expect(last[0].id).toBe(PROMPT_CARDS[49].id)
  })
})
