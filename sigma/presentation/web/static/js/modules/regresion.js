// Linear regression page (simple when one numeric predictor is chosen, multiple otherwise).

import { analysisPage, existing } from "../analysis-page.js";

const emptyForm = () => ({ response: "", predictors: [], categorical: [], intercept: true, confidence: 0.95 });

document.addEventListener("alpine:init", () => {
  Alpine.data("regresionPage", () => ({
    ...analysisPage("regresion", ["form", "rows"]),
    form: emptyForm(),
    rows: [], // new observations to predict: [{predictor: value}]

    async init() {
      this.onDatasetChange(() => { this.form = emptyForm(); this.rows = []; });
      this.$watch("form.response", (name) => {
        this.form.predictors = this.form.predictors.filter((p) => p !== name);
      });
      this.$watch("form.predictors", (names) => {
        this.form.categorical = this.form.categorical.filter((c) => names.includes(c));
        this.rows = this.rows.map((row) => Object.fromEntries(names.map((n) => [n, row[n] ?? ""])));
      });
      await this.startWorkspace();
    },

    resetWorkspace() {
      this.form = emptyForm();
      this.rows = [];
    },

    get kind() {
      const names = this.form.predictors;
      if (!names.length) return "";
      const numericOnly = names.every((n) => this.kindOf(n) === "numeric" && !this.form.categorical.includes(n));
      return names.length === 1 && numericOnly ? "simple" : "múltiple";
    },
    get canRun() {
      return !!Alpine.store("datasets").activeId && !!this.form.response && this.form.predictors.length > 0;
    },
    get numericPredictors() {
      return this.form.predictors.filter((n) => this.kindOf(n) === "numeric");
    },

    kindOf(name) {
      return Alpine.store("datasets").columns.find((c) => c.name === name)?.kind;
    },
    isCategorical(name) {
      return this.kindOf(name) !== "numeric" || this.form.categorical.includes(name);
    },

    addRow() {
      this.rows.push(Object.fromEntries(this.form.predictors.map((n) => [n, ""])));
    },
    removeRow(index) {
      this.rows.splice(index, 1);
    },

    body() {
      const f = this.form;
      const predict = this.rows
        .filter((row) => this.form.predictors.every((n) => row[n] !== "" && row[n] !== null))
        .map((row) => Object.fromEntries(f.predictors.map((n) => [n, this.isCategorical(n) ? String(row[n]) : Number(row[n])])));
      return {
        dataset_id: Alpine.store("datasets").activeId, response: f.response, predictors: f.predictors,
        categorical: f.categorical, intercept: f.intercept, confidence: Number(f.confidence) || 0.95, predict,
      };
    },

    async submit() {
      if (!this.canRun) { this.error = "Elija la variable respuesta y al menos un predictor."; return; }
      await this.runAnalysis("/api/regression/run", this.body());
    },

    applyParams(p) {
      this.form = {
        response: p.response ?? "", predictors: existing(p.predictors ?? []), categorical: p.categorical ?? [],
        intercept: p.intercept ?? true, confidence: p.confidence ?? 0.95,
      };
      this.rows = (p.predict ?? []).map((row) => ({ ...row }));
    },
  }));
});
