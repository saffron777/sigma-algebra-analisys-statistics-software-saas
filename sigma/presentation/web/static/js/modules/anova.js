// ANOVA page: one factor, or two factors with optional interaction.

import { analysisPage } from "../analysis-page.js";

const emptyForm = () => ({
  response: "", factor1: "", factor2: "", alpha: 0.05, interaction: true, ss_type: 2, post_hoc: true,
});

document.addEventListener("alpine:init", () => {
  Alpine.data("anovaPage", () => ({
    ...analysisPage("anova", ["form"]),
    form: emptyForm(),

    async init() {
      this.onDatasetChange(() => { this.form = emptyForm(); });
      await this.startWorkspace();
    },

    resetWorkspace() {
      this.form = emptyForm();
    },

    get factors() { return [this.form.factor1, this.form.factor2].filter(Boolean); },
    get canRun() {
      return !!Alpine.store("datasets").activeId && !!this.form.response && this.factors.length > 0
        && this.form.factor1 !== this.form.response && this.form.factor2 !== this.form.response;
    },

    body() {
      const f = this.form;
      return {
        dataset_id: Alpine.store("datasets").activeId, response: f.response, factors: this.factors,
        alpha: Number(f.alpha) || 0.05, interaction: f.interaction, ss_type: Number(f.ss_type), post_hoc: f.post_hoc,
      };
    },

    async submit() {
      if (!this.canRun) { this.error = "Elija la variable respuesta y al menos un factor distinto de ella."; return; }
      await this.runAnalysis("/api/anova/run", this.body());
    },

    applyParams(p) {
      const [factor1 = "", factor2 = ""] = p.factors ?? [];
      this.form = {
        response: p.response ?? "", factor1, factor2, alpha: p.alpha ?? 0.05,
        interaction: p.interaction ?? true, ss_type: p.ss_type ?? 2, post_hoc: p.post_hoc ?? true,
      };
    },
  }));
});
