/** Suggestion cards for an empty chat: 50 prompts, three drawn at random whenever a
 *  conversation starts fresh. Rotation policy: every MINOR release replaces ten cards
 *  (docs/releasing.md, tools/rotate_prompts.py picks the ten to retire). Keep exactly
 *  50 entries with unique ids and texts; prompts.test.ts enforces that. */
export interface PromptCard {
  id: string
  text: string
}

export const PROMPT_CARDS: readonly PromptCard[] = [
  { id: "moe-offload", text: "Why does offloading MoE experts to RAM slow decoding more than prefill?" },
  { id: "c-bench", text: "Write a small C benchmark for memcpy throughput and explain the result" },
  { id: "caching", text: "Compare RAM and VRAM caching strategies for expert weights" },
  { id: "context-budget", text: "I have a 128K context window. How should an agent budget it across tools, history and answer?" },
  { id: "cron-systemd", text: "Turn a cron job into a systemd timer with logging and a retry on failure" },
  { id: "pcie", text: "Estimate how many 2.5 MB experts per second fit through PCIe 4 x16" },
  { id: "mmap", text: "When is mmap with MADV_RANDOM the wrong choice for reading large files?" },
  { id: "pinned-memory", text: "What does pinned host memory buy a CUDA transfer, and what does it cost the OS?" },
  { id: "json-diff", text: "Write a Python function that diffs two nested JSON objects and lists the changed paths" },
  { id: "http-cache", text: "Explain HTTP caching headers: Cache-Control, ETag and when each applies" },
  { id: "ssh-config", text: "Write an ~/.ssh/config with a jump host, per-host keys and connection sharing" },
  { id: "makefile", text: "Write a Makefile for a C project with a debug and a release target" },
  { id: "pytest-fixtures", text: "Write pytest fixtures for a temp directory and a fake HTTP server, then a test using both" },
  { id: "async-compare", text: "Compare async in Rust and Python: what does an executor actually do?" },
  { id: "ts-unions", text: "Model API responses in TypeScript with discriminated unions and an exhaustive switch" },
  { id: "http-retry", text: "Write a Python HTTP client with retries, exponential backoff and a total timeout" },
  { id: "css-grid", text: "Lay out a responsive three-column card grid with CSS grid, no framework" },
  { id: "sql-window", text: "Explain SQL window functions with a running total example" },
  { id: "ssh-tunnel", text: "Set up an SSH tunnel that forwards a remote port to localhost and restarts when it drops" },
  { id: "sqlite-wal", text: "When should SQLite run in WAL mode, and what changes for concurrent readers?" },
  { id: "git-bisect", text: "Walk me through git bisect with a script that finds a performance regression" },
  { id: "tls", text: "Explain what happens during a TLS 1.3 handshake in ten lines" },
  { id: "hash-map", text: "Implement an open-addressing hash map in C with linear probing" },
  { id: "bloom", text: "Explain a Bloom filter and size one for ten million URLs at 1% false positives" },
  { id: "endianness", text: "Explain endianness with a 32-bit integer written to disk on x86 and read on ARM" },
  { id: "unicode", text: "Explain the difference between a code point, a grapheme and a byte in UTF-8" },
  { id: "json-schema", text: "Write a JSON schema for a config file with nested optional sections" },
  { id: "german-tokens", text: "Why does a tokenizer split long German compound words into many tokens?" },
  { id: "backup-321", text: "Design a 3-2-1 backup plan for a home server with two disks and a cloud bucket" },
  { id: "refactor", text: "How would you split a 2,000-line Python module into packages?" },
  { id: "logging", text: "Design structured logging for a CLI tool: fields, levels, rotation" },
  { id: "deadlock", text: "Show a two-lock deadlock in Python and two ways to prevent it" },
  { id: "quant-kl", text: "What does KL divergence against the full-precision model tell me about a quantized model?" },
  { id: "attention", text: "Explain scaled dot-product attention with a 3-token example" },
  { id: "tokenizer", text: "How does a BPE tokenizer split an unknown word? Walk through one" },
  { id: "long-book", text: "How should I summarize a 300-page book with a 190K-token context model?" },
  { id: "speculative", text: "Explain speculative decoding and when it does not help" },
  { id: "bash-trap", text: "Write a bash script that cleans up its temp files on exit, error or Ctrl+C" },
  { id: "ocr-pipeline", text: "Design a pipeline that extracts tables from scanned PDFs into CSV" },
  { id: "kv-cache-size", text: "How much memory does a KV cache need for a 128K context? Walk through the formula" },
  { id: "ssd", text: "What matters more for model loading, SSD bandwidth or IOPS? Explain" },
  { id: "numa", text: "Explain NUMA on a dual-socket server and how to pin a process to one node" },
  { id: "sql-index", text: "Why is my SQL query not using the index? List the usual causes" },
  { id: "nvme-health", text: "How do I read NVMe SMART data on Linux and which fields predict failure?" },
  { id: "ssh-keys", text: "Explain SSH key types and set up a key-only login step by step" },
  { id: "networking", text: "Explain what a subnet mask does with the example 192.168.1.0/24" },
  { id: "rate-limit", text: "Implement a token bucket rate limiter in Python and explain its parameters" },
  { id: "summary", text: "Summarize the trade-offs of monorepos versus many repositories" },
  { id: "py-profile", text: "How do I profile a slow Python script and find the hot function?" },
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
