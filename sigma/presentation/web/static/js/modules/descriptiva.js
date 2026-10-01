// Descriptive statistics page.

import { api } from "../api.js";
import { analysisPage, existing } from "../analysis-page.js";

const emptyForm = () => ({
  numeric: [], categorical: [], group_by: "", confidence: 0.95, normality: true, correlation: "", fits: [],
});

document.addEventListener("alpine:init", () => {
  Alpine.data("descriptivaPage", () => ({
    ...analysisPage("descriptiva", ["form"]),
    fittable: [],
    form: emptyForm(),

    async init() {
      try {
        this.fittable = (await api("/api/descriptive/options")).fittable;
      } catch { /* optional list */ }
      this.onDatasetChange(() => { this.form = emptyForm(); });
      await this.startWorkspace();
    },

    resetWorkspace() {
      this.form = emptyForm();
    },

    get canRun() {
      return !!Alpine.store("datasets").activeId && (this.form.numeric.length || this.form.categorical.length);
    },

    body() {
      const f = this.form;
      return {
        dataset_id: Alpine.store("datasets").activeId,
        numeric_columns: f.numeric, categorical_columns: f.categorical,
        group_by: f.group_by || null, confidence: Number(f.confidence) || 0.95,
        run_normality: f.normality, fit_distributions: f.fits, correlation_method: f.correlation || null,
      };
    },

    async submit() {
      if (!this.canRun) { this.error = "Elija un dataset y al menos una variable."; return; }
      await this.runAnalysis("/api/descriptive/run", this.body());
    },

    applyParams(p) {
      this.form = {
        numeric: existing(p.numeric_columns ?? []), categorical: existing(p.categorical_columns ?? []),
        group_by: p.group_by ?? "", confidence: p.confidence ?? 0.95, normality: p.run_normality ?? true,
        correlation: p.correlation_method ?? "", fits: p.fit_distributions ?? [],
      };
    },
  }));
});
