// Behaviour shared by every analysis page: run an endpoint, render the report, save it, reopen a saved one.
// Usage inside an Alpine component:  Alpine.data("x", () => ({ ...analysisPage(), form: {...}, ... }))

import { api } from "./api.js";
import { renderReport } from "./report.js";

export function analysisPage() {
  return {
    loading: false,
    error: "",
    payload: null,
    savedId: null,

    /** POST `body` to `url` and show the resulting report. Returns the payload (or null on error). */
    async runAnalysis(url, body) {
      this.loading = true;
      this.error = "";
      this.savedId = null;
      try {
        const data = await api(url, { method: "POST", json: body });
        await this.showPayload(data);
        return data;
      } catch (error) {
        this.payload = null;
        this.error = error.message;
        return null;
      } finally {
        this.loading = false;
      }
    },

    /** Render an already-computed payload `{meta, result, report, figures}`. */
    async showPayload(data) {
      this.payload = data;
      this.loading = false;
      await this.$nextTick(); // the result section must be visible before Plotly measures it
      renderReport(this.$refs.out, data.report, data.figures);
    },

    openSave() {
      Alpine.store("save").show(this.payload, (id) => { this.savedId = id; });
    },

    /** Call `callback` whenever the active dataset changes (selections made for another dataset are stale). */
    onDatasetChange(callback) {
      this.$watch("$store.datasets.activeId", () => {
        this.payload = null;
        this.error = "";
        callback();
      });
    },

    /** If the URL carries ?load=<saved result id>, restore its form and report (see `applyParams`). */
    async loadSavedFromUrl() {
      const id = new URLSearchParams(location.search).get("load");
      if (!id) return false;
      try {
        const { result } = await api(`/api/results/${encodeURIComponent(id)}`);
        if (result.dataset_id) Alpine.store("datasets").select(result.dataset_id);
        await this.$nextTick();
        this.applyParams?.(result.params ?? {}, result);
        await this.showPayload({
          meta: { module: result.module, analysis_type: result.analysis_type, title: result.title,
                  dataset_id: result.dataset_id, params: result.params },
          result: result.result, report: result.report, figures: result.figures,
        });
        this.savedId = result.id;
        return true;
      } catch (error) {
        this.error = error.message;
        return false;
      }
    },
  };
}

/** Keep only the columns that still exist in the active dataset. */
export function existing(selection, kinds = []) {
  const names = new Set(Alpine.store("datasets").columnsOfKind(kinds).map((column) => column.name));
  return selection.filter((name) => names.has(name));
}
