/** Dashboard tool declarations and result parsing. One tool for now: web_search via
 *  Serper, relayed by the gateway (`/v1/tools/web_search`). Declared on every request
 *  of a chat once the user has entered a key, so the model can use it from turn one. */
export const WEB_SEARCH_TOOLS = [
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web (Google results via Serper). Use it for current events, facts you are unsure about, documentation, prices, versions or anything after your training data. Returns titles, links and snippets.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "The search query, as you would type it into a search engine." },
          num: { type: "integer", minimum: 1, maximum: 10, description: "Number of results (default 5)." },
        },
        required: ["query"],
      },
    },
  },
] as const

export interface WebSearchHit {
  title?: string
  link?: string
  snippet?: string
  date?: string
}

export interface ParsedWebSearch {
  query: string
  results: WebSearchHit[]
  answer?: string
  error?: string
}

function parse(content: string): ParsedWebSearch {
  try {
    const data = JSON.parse(content) as { query?: string; results?: WebSearchHit[]; answer?: Record<string, string>; error?: string }
    const answer = data.answer ? [data.answer.answer, data.answer.snippet].filter(Boolean).join(" ") : undefined
    return { query: data.query ?? "", results: Array.isArray(data.results) ? data.results : [], answer: answer || undefined, error: data.error }
  } catch {
    return { query: "", results: [], error: "unreadable result" }
  }
}

/** The query inside a tool call's raw JSON arguments, for display while it runs. */
parse.query = (args: string): string => {
  try { return String((JSON.parse(args || "{}") as { query?: unknown }).query ?? "") } catch { return args }
}

export const parseWebSearchResult = parse
