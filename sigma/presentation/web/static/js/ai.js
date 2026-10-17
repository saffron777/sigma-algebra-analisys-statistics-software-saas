// Optional AI interpretation of results: the `ai` store (availability, consent, the request) and the helpers that
// show an interpretation and carry it into saved results and exports.

import { api } from "./api.js";
import { renderMarkdown } from "./output.js";

const CONSENT_KEY = "sigma-ai-consent";
const CONSENT_TEXT =
  "Se enviará al proveedor de IA configurado en su servidor un resumen de este resultado (parámetros, cifras y "
  + "tablas; no sus datos completos ni los gráficos) para que lo interprete.\n\nLa respuesta es una ayuda, no "
  + "sustituye la revisión de un analista. ¿Continuar?";

const read = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
const write = (key, value) => { try { localStorage.setItem(key, value); } catch { /* ignore */ } };

export const DISCLAIMER = "Generado por IA: puede contener errores. Verifique las cifras con el análisis antes de usarlo.";

/** The report blocks that carry an interpretation into an export (PDF / Word / Excel). */
export function interpretationBlocks(interpretation) {
  if (!interpretation?.text) return [];
  return [
    { type: "heading", text: "Interpretación (IA)" },
    { type: "markdown", text: interpretation.text },
    { type: "alert", kind: "info", message: `${DISCLAIMER}${interpretation.model ? ` Modelo: ${interpretation.model}.` : ""}` },
  ];
}

/** Draw an interpretation (Markdown) into `element`; sanitised, never interpreted as raw HTML. */
export function drawInterpretation(element, interpretation) {
  if (!element) return;
  if (interpretation?.text) renderMarkdown(element, interpretation.text);
  else element.replaceChildren();
}

export function defineAiStore() {
  Alpine.store("ai", {
    status: { available: false, reason: "", model: "", model_label: "", provider: "", server_id: null, models: [], requests_this_hour: 0, requests_per_hour: 0 },
    loaded: false,
    busy: false,

    async init() {
      await this.refresh();
    },
    async refresh() {
      try {
        this.status = await api("/api/ai/status");
      } catch { /* the rest of Sigma Algebra works without it */ }
      this.loaded = true;
    },
    get available() { return this.status.available; },

    /** Ask the first time only; returns false if the user declines. */
    consent() {
      if (read(CONSENT_KEY) === "1") return true;
      if (!window.confirm(CONSENT_TEXT)) return false;
      write(CONSENT_KEY, "1");
      return true;
    },

    /**
     * @param {{module: string, title?: string, report?: object, focus?: string, dataset_id?: string,
     *          model_id?: string}} request
     * @returns {Promise<object|null>} the interpretation, or null if the user declined the consent question
     */
    async interpret(request) {
      if (!this.consent()) return null;
      this.busy = true;
      try {
        const { interpretation } = await api("/api/ai/interpret", { method: "POST", json: request });
        return interpretation;
      } finally {
        this.busy = false;
        this.refresh();
      }
    },
  });
}
