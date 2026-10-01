// Hypothesis-test page: the form is generated from the server's test catalogue.

import { api } from "../api.js";
import { analysisPage } from "../analysis-page.js";

const toNumber = (value) => (value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value));
const TAIL_LABELS = { "two-sided": "Bilateral (≠)", left: "Cola izquierda (<)", right: "Cola derecha (>)" };

document.addEventListener("alpine:init", () => {
  Alpine.data("hipotesisPage", () => ({
    ...analysisPage("hipotesis", ["testId", "mode", "roles", "params", "summary", "tail", "alpha"]),
    tailLabels: TAIL_LABELS,
    tests: [],
    testId: "t_one_sample",
    mode: "data", // data | summary
    roles: {},
    params: {},
    summary: {},
    tail: "two-sided",
    alpha: 0.05,

    async init() {
      this.tests = (await api("/api/hypothesis/catalog")).tests;
      this.selectTest(this.testId);
      this.onDatasetChange(() => { this.roles = {}; });
      await this.startWorkspace();
    },

    resetWorkspace() {
      this.selectTest("t_one_sample");
      this.mode = "data";
      this.tail = "two-sided";
      this.alpha = 0.05;
    },

    /** A stored form: choose the test first (it sets that test's defaults), then overlay what the user had typed. */
    applyWorkspace(state) {
      if (!this.tests.some((t) => t.id === state.testId)) return;
      this.selectTest(state.testId);
      this.mode = this.hasSummary ? (state.mode ?? "data") : "data";
      this.roles = { ...(state.roles ?? {}) };
      this.params = { ...this.params, ...(state.params ?? {}) };
      this.summary = { ...this.summary, ...(state.summary ?? {}) };
      this.tail = state.tail ?? "two-sided";
      this.alpha = state.alpha ?? 0.05;
    },

    get spec() { return this.tests.find((t) => t.id === this.testId); },
    get grouped() {
      const groups = new Map();
      for (const test of this.tests) groups.set(test.group, [...(groups.get(test.group) ?? []), test]);
      return [...groups].map(([group, tests]) => ({ group, tests }));
    },
    get hasSummary() { return (this.spec?.summary_fields.length ?? 0) > 0; },
    get usingSummary() { return this.mode === "summary" && this.hasSummary; },

    kindsFor(role) {
      return { numeric: ["numeric", "boolean"], categorical: ["categorical", "boolean", "text"] }[role.kind] ?? [];
    },

    selectTest(id) {
      this.testId = id;
      const spec = this.spec;
      this.roles = {};
      this.params = Object.fromEntries(spec.params.map((p) => [p.name, p.default ?? ""]));
      this.summary = Object.fromEntries(spec.summary_fields.map((f) => [f.name, ""]));
      if (!this.hasSummary) this.mode = "data";
      if (!spec.tails.includes(this.tail)) this.tail = spec.tails[0];
    },

    get canRun() {
      if (this.usingSummary) return true;
      return !!Alpine.store("datasets").activeId;
    },

    convert(fields, values) {
      const out = {};
      for (const field of fields) {
        const value = values[field.name];
        out[field.name] = field.kind === "number" ? toNumber(value) : value;
      }
      return out;
    },

    body() {
      const spec = this.spec;
      return {
        test_id: this.testId,
        dataset_id: this.usingSummary ? null : Alpine.store("datasets").activeId || null,
        roles: this.usingSummary ? {} : this.roles,
        params: this.convert(spec.params, this.params),
        summary: this.usingSummary ? this.convert(spec.summary_fields, this.summary) : null,
        alpha: toNumber(this.alpha) ?? 0.05,
        tail: this.tail,
      };
    },

    async submit() {
      if (!this.canRun) { this.error = "Elija un dataset o use estadísticos resumidos."; return; }
      await this.runAnalysis("/api/hypothesis/run", this.body());
    },

    applyParams(p) {
      if (!this.tests.length) return;
      this.selectTest(p.test_id);
      this.mode = p.summary ? "summary" : "data";
      this.roles = { ...(p.roles ?? {}) };
      this.params = { ...this.params, ...(p.params ?? {}) };
      this.summary = { ...this.summary, ...(p.summary ?? {}) };
      this.tail = p.tail ?? "two-sided";
      this.alpha = p.alpha ?? 0.05;
    },
  }));
});
