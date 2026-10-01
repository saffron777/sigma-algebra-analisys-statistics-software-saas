// Distribution calculator: pick a distribution, set its parameters, evaluate in one of four modes.

import { api } from "../api.js";
import { analysisPage } from "../analysis-page.js";

const MODES = [
  { key: "point", label: "Evaluar en x" },
  { key: "between", label: "P(a ≤ X ≤ b)" },
  { key: "quantile", label: "Cuantil" },
  { key: "pvalue", label: "Valor p" },
];

const toNumber = (value) => (value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value));

document.addEventListener("alpine:init", () => {
  Alpine.data("distribucionesPage", () => ({
    ...analysisPage(),
    modes: MODES,
    catalog: [],
    key: "normal",
    params: {},
    mode: "point",
    inputs: { x: 0, a: -1, b: 1, q: 0.95, statistic: 1.96 },
    tail: "two-sided",
    alpha: 0.05,
    timer: null,

    async init() {
      const { distributions } = await api("/api/distributions");
      this.catalog = distributions;
      this.selectDistribution("normal");
      if (!(await this.loadSavedFromUrl())) await this.submit();
    },

    get spec() {
      return this.catalog.find((d) => d.key === this.key);
    },

    selectDistribution(key) {
      this.key = key;
      this.params = Object.fromEntries(this.spec.params.map((p) => [p.name, p.default]));
    },

    formIsValid() {
      if (!this.spec) return false;
      if (this.spec.params.some((p) => toNumber(this.params[p.name]) === null)) return false;
      const needed = { point: ["x"], between: ["a", "b"], quantile: ["q"], pvalue: ["statistic"] }[this.mode];
      return needed.every((name) => toNumber(this.inputs[name]) !== null);
    },

    body() {
      return {
        distribution: this.key,
        params: Object.fromEntries(Object.entries(this.params).map(([k, v]) => [k, toNumber(v)])),
        mode: this.mode,
        x: toNumber(this.inputs.x), a: toNumber(this.inputs.a), b: toNumber(this.inputs.b),
        q: toNumber(this.inputs.q), statistic: toNumber(this.inputs.statistic),
        tail: this.tail, alpha: toNumber(this.alpha) ?? 0.05,
      };
    },

    /** Debounced recalculation while the user edits the form (only when every needed field is filled). */
    schedule() {
      clearTimeout(this.timer);
      this.timer = setTimeout(() => { if (this.formIsValid()) this.submit(); }, 450);
    },

    async submit() {
      if (!this.formIsValid()) { this.error = "Complete todos los parámetros con números válidos."; return; }
      await this.runAnalysis("/api/distributions/calculate", this.body());
    },

    applyParams(p) {
      if (!this.catalog.length) return;
      this.key = p.distribution ?? this.key;
      this.params = { ...p.params };
      this.mode = p.mode ?? "point";
      this.inputs = { x: p.x ?? 0, a: p.a ?? -1, b: p.b ?? 1, q: p.q ?? 0.95, statistic: p.statistic ?? 1.96 };
      this.tail = p.tail ?? "two-sided";
      this.alpha = p.alpha ?? 0.05;
    },
  }));
});
