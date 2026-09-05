import { useEffect, useRef } from "react"
import { X } from "lucide-react"
import { useLocale } from "./i18n"

/* Credits for the prior work AI-DER is built on.
 *
 * The entries are deliberately NOT in the i18n dictionaries: project names,
 * author handles and licence identifiers are proper nouns that must read the
 * same in every locale, and a mistranslated licence name is worse than an
 * untranslated one. Only the surrounding chrome (link label, heading, intro,
 * close button) is localized.
 *
 * Keep this list in sync with docs/ATTRIBUTION.md at the repository root. */
interface Credit {
  name: string
  by: string
  what: string
  license?: string
}

const CREDITS: Credit[] = [
  {
    name: "colibrì",
    by: "JustVugg",
    what:
      "The engine AI-DER is forked from: the expert-tiering design, the stdin/stdout serve protocol, " +
      "the OpenAI-compatible gateway and this dashboard all originate there. The pinned upstream commit " +
      "is recorded in engine/UPSTREAM.txt.",
    license: "Apache-2.0",
  },
  {
    name: "llama.cpp / ggml",
    by: "ggml-org",
    what:
      "The GGUF container format, the quantization block formats and the reference dequantization " +
      "implementations. The qwen4exp reference implementation was used to validate this engine's output " +
      "token-for-token, and tools/mtmd's qwen3vl_merger path is the reference and parity oracle for the " +
      "vision tower. No mtmd code is vendored. llama-server is also the optional llama.cpp backend " +
      "behind the gateway, run unmodified as a child process.",
    license: "MIT",
  },
  {
    name: "Pillow · NumPy",
    by: "Pillow contributors · NumPy developers",
    what:
      "Image decoding and resampling for the vision path, in the Python gateway only " +
      "(engine/c/tools/qwen38_image.py). No image decoder is compiled into the C engine.",
    license: "HPND · BSD-3-Clause",
  },
  {
    name: "Qwen3.8-Flash-Next",
    by: "Qwen team, Alibaba",
    what: "The model this runner serves.",
  },
  {
    name: "unsloth",
    by: "unsloth",
    what: "The UD-IQ4_XS dynamic quantization of the weights AI-DER runs.",
  },
  {
    name: "Qwen3.8-27B",
    by: "Qwen team, Alibaba · W4A16 AutoRound quantization by dbirks",
    what: "The dense sibling served by the optional vLLM backend.",
    license: "Apache-2.0",
  },
  {
    name: "qwen38-27b-rtx3090",
    by: "syv-ai",
    what:
      "The single-GPU serving stack for Qwen3.8-27B on an RTX 3090 that the vLLM backend runs unmodified: " +
      "the patched vLLM 0.27.1, the head/drafter preparation scripts and the single-user launcher. " +
      "AI-DER contributes only the gateway proxy around it.",
    license: "Apache-2.0",
  },
  {
    name: "MoE-Infinity · SP-MoE · SpecPrefetch",
    by: "research prior art",
    what:
      "Expert-offloading and speculative expert-prefetch research that shaped AI-DER's prefetcher design: " +
      "MoE-Infinity's activation-aware offloading, and the SP-MoE and SpecPrefetch papers on predicting " +
      "expert selection ahead of the routing decision.",
  },
  {
    name: "AI Engram: In Search of Memory Traces in Artificial Intelligence",
    by: "research prior art · arxiv.org/pdf/2606.14997",
    what: "The engram framing behind the runner's name and its view of experts as memory traces.",
  },
  {
    name: "ds4 · DeepGEMM · CUTLASS/CuTe · vLLM",
    by: "antirez · DeepSeek · NVIDIA · vLLM",
    what:
      "Credited by colibrì upstream and still present in this build's GPU tiers and reference tooling. " +
      "Exact scope, copyright lines and licence texts are in engine/THIRD_PARTY_NOTICES.md.",
    license: "MIT · BSD-3-Clause",
  },
]

export function Attribution({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null)

  /* Escape closes, and focus starts on the close button so the dialog is
     reachable without a mouse. */
  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose() }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  const { t } = useLocale()

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal attribution-modal" role="dialog" aria-modal="true" aria-label={t("attribution.title")}
        onClick={(event) => event.stopPropagation()}>
        <header className="modal-head">
          <h2>{t("attribution.title")}</h2>
          {/* A plain button, not <Button>: that component is not a forwardRef,
              so it would silently drop the ref this dialog focuses on open. */}
          <button ref={closeRef} type="button" className="modal-close" aria-label={t("attribution.close")} onClick={onClose}>
            <X className="size-4" />
          </button>
        </header>
        <p className="modal-intro">{t("attribution.intro")}</p>
        <ul className="credit-list">
          {CREDITS.map((credit) => (
            <li key={credit.name}>
              <div className="credit-head">
                <strong>{credit.name}</strong>
                <span className="credit-by">{credit.by}</span>
                {credit.license ? <code>{credit.license}</code> : null}
              </div>
              <p>{credit.what}</p>
            </li>
          ))}
        </ul>
        <footer className="modal-foot">
          <p>{t("attribution.upstream")}</p>
          <p>{t("attribution.notices")}</p>
        </footer>
      </div>
    </div>
  )
}
