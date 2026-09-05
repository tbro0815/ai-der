/** Suggestion cards for an empty chat: 50 prompts, three drawn at random whenever a
 *  conversation starts fresh. Rotation policy: every MINOR release replaces ten cards
 *  (docs/releasing.md, tools/rotate_prompts.py picks the ten to retire). Keep exactly
 *  50 entries with unique ids and texts; prompts.test.ts enforces that. */
export interface PromptCard {
  id: string
  text: string
}

export const PROMPT_CARDS: readonly PromptCard[] = [
  { id: "routing", text: "Explain how expert routing works in a mixture-of-experts model" },
  { id: "c-bench", text: "Write a small C benchmark for memcpy throughput and explain the result" },
  { id: "caching", text: "Compare RAM and VRAM caching strategies for expert weights" },
  { id: "context-budget", text: "I have a 128K context window. How should an agent budget it across tools, history and answer?" },
  { id: "quant", text: "What does 4-bit quantization do to a weight matrix, with a worked example" },
  { id: "pcie", text: "Estimate how many 2.5 MB experts per second fit through PCIe 4 x16" },
  { id: "mmap", text: "When is mmap with MADV_RANDOM the wrong choice for reading large files?" },
  { id: "page-cache", text: "How does the Linux page cache decide what to evict? Keep it practical" },
  { id: "csv-clean", text: "Clean a messy CSV export in Python: mixed date formats, stray quotes, duplicate rows" },
  { id: "regex", text: "Write a regex that matches ISO 8601 timestamps and explain each part" },
  { id: "git-bisect", text: "Show me how to use git bisect to find the commit that broke a test" },
  { id: "makefile", text: "Write a Makefile for a C project with a debug and a release target" },
  { id: "python-typing", text: "Add type hints to a Python function that parses a config dict; explain TypedDict versus dataclass" },
  { id: "rust-borrow", text: "Explain the Rust borrow checker using a function that returns a slice" },
  { id: "ts-generics", text: "Show a TypeScript generic that types a fetch wrapper's JSON result" },
  { id: "react-effect", text: "When does a React useEffect run twice, and how do I fix it properly?" },
  { id: "css-grid", text: "Lay out a responsive three-column card grid with CSS grid, no framework" },
  { id: "sql-window", text: "Explain SQL window functions with a running total example" },
  { id: "bash-safe", text: "Rewrite this pattern safely: for f in $(ls *.txt); do ... done" },
  { id: "cron-vs-timer", text: "Compare cron and systemd timers for a nightly backup job, with examples" },
  { id: "git-worktree", text: "When does git worktree beat branches and stashes? Show a workflow" },
  { id: "tls", text: "Explain what happens during a TLS 1.3 handshake in ten lines" },
  { id: "hash-map", text: "Implement an open-addressing hash map in C with linear probing" },
  { id: "bloom", text: "Explain a Bloom filter and size one for ten million URLs at 1% false positives" },
  { id: "endianness", text: "Explain endianness with a 32-bit integer written to disk on x86 and read on ARM" },
  { id: "unicode", text: "Explain the difference between a code point, a grapheme and a byte in UTF-8" },
  { id: "json-schema", text: "Write a JSON schema for a config file with nested optional sections" },
  { id: "unit-tests", text: "Propose unit tests for a function that parses durations like 1h30m" },
  { id: "code-review", text: "Review this idea: caching API responses in localStorage for a week" },
  { id: "refactor", text: "How would you split a 2,000-line Python module into packages?" },
  { id: "logging", text: "Design structured logging for a CLI tool: fields, levels, rotation" },
  { id: "concurrency", text: "Explain a mutex, a semaphore and a condition variable with one example each" },
  { id: "gpu-occupancy", text: "What limits CUDA kernel occupancy and how would I measure it?" },
  { id: "attention", text: "Explain scaled dot-product attention with a 3-token example" },
  { id: "tokenizer", text: "How does a BPE tokenizer split an unknown word? Walk through one" },
  { id: "moe-vs-dense", text: "Why can a 100B mixture-of-experts model run where a 100B dense model cannot?" },
  { id: "speculative", text: "Explain speculative decoding and when it does not help" },
  { id: "prompt-cache", text: "What is prompt prefix caching and why does a changed system prompt defeat it?" },
  { id: "vision", text: "How does a vision-language model turn an image into tokens?" },
  { id: "eval", text: "Design a small evaluation set for a coding assistant, with pass criteria" },
  { id: "ssd", text: "What matters more for model loading, SSD bandwidth or IOPS? Explain" },
  { id: "ddr5", text: "How much faster is DDR5 than DDR4 for a bandwidth-bound memcpy?" },
  { id: "thermal", text: "My GPU throttles at 83 C under sustained load. What are my options, cheapest first?" },
  { id: "nvme-health", text: "How do I read NVMe SMART data on Linux and which fields predict failure?" },
  { id: "ssh-keys", text: "Explain SSH key types and set up a key-only login step by step" },
  { id: "networking", text: "Explain what a subnet mask does with the example 192.168.1.0/24" },
  { id: "markdown-table", text: "Turn this list into a Markdown table: CPU, RAM, GPU, SSD, price" },
  { id: "summary", text: "Summarize the trade-offs of monorepos versus many repositories" },
  { id: "learning", text: "Give me a two-week plan to learn CUDA programming from C knowledge" },
  { id: "debug", text: "My program segfaults only under valgrind. Where do I start?" },
]

/** Three distinct cards; `random` is injectable for tests. */
export function pickSuggestions(count = 3, random: () => number = Math.random): PromptCard[] {
  const pool = [...PROMPT_CARDS]
  const picked: PromptCard[] = []
  while (picked.length < count && pool.length) {
    const index = Math.min(pool.length - 1, Math.floor(random() * pool.length))
    picked.push(pool.splice(index, 1)[0])
  }
  return picked
}
