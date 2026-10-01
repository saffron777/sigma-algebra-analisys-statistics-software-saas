// Market-research page: sample size, random sample selection, estimation, cross-tabs and Likert scales.

import { analysisPage, existing } from "../analysis-page.js";

const TABS = [
  { key: "size", label: "Tamaño de muestra" },
  { key: "draw", label: "Selección de muestra" },
  { key: "estimate", label: "Estimación" },
  { key: "crosstab", label: "Tabulación cruzada" },
  { key: "likert", label: "Escala Likert" },
];
const CI_METHODS = [
  { key: "wilson", label: "Wilson" }, { key: "wald", label: "Wald (normal)" },
  { key: "agresti_coull", label: "Agresti-Coull" }, { key: "clopper_pearson", label: "Clopper-Pearson" },
];

const num = (value) => (value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value));
const text = (value) => (value ? value : null);

const defaults = () => ({
  size: { kind: "proportion", confidence: 0.95, margin: 5, proportion: 50, std_dev: "", population: "", deff: 1,
          strata_column: "", allocation: "proportional", sd_column: "" },
  draw: { design: "srs", n: "", seed: "", strata_column: "", allocation: "proportional", sd_column: "", cluster_column: "",
          n_clusters: "", save_as: "" },
  estimate: { kind: "proportion", variable: "", confidence: 0.95, methods: ["wilson"], population: "", weight: "",
              strata: "", cluster: "", fpc: true },
  crosstab: { row: "", col: "" },
  likert: { items: [] },
});

document.addEventListener("alpine:init", () => {
  Alpine.data("mercadosPage", () => {
    const base = analysisPage("mercados", ["tab", "f", "suggestedN"]);
    return {
    ...base,
    tabs: TABS,
    ciMethods: CI_METHODS,
    tab: "size",
    f: defaults(),
    suggestedN: null,
    results: {}, // the last result of each tab, so switching tabs does not lose any of them

    async init() {
      this.onDatasetChange(() => {
        const keep = this.f.size;
        this.f = { ...defaults(), size: keep };
        this.results = this.results.size ? { size: this.results.size } : {}; // the others belonged to the old dataset
        if (this.tab === "size" && this.results.size) this.showPayload(this.results.size);
      });
      await this.startWorkspace();
    },

    async showPayload(data) {
      this.results[this.tab] = data;
      await base.showPayload.call(this, data);
    },

    async selectTab(key) {
      this.tab = key;
      this.error = "";
      const kept = this.results[key] ?? null;
      if (kept) await base.showPayload.call(this, kept);
      else this.payload = null;
    },

    // ------------------------------------------------------------------ workspace
    workspacePayload() {
      return Object.keys(this.results).length ? { results: this.results } : null;
    },
    async restorePayload(stored) {
      this.results = stored.results ?? {};
      const current = this.results[this.tab];
      if (current) await base.showPayload.call(this, current);
    },
    resetWorkspace() {
      this.f = defaults();
      this.tab = "size";
      this.results = {};
      this.suggestedN = null;
    },

    get hasDataset() { return !!Alpine.store("datasets").activeId; },
    get needsDataset() { return this.tab !== "size"; },
    get canRun() {
      const f = this.f;
      if (this.needsDataset && !this.hasDataset) return false;
      return {
        size: true,
        draw: !!f.draw.design,
        estimate: !!f.estimate.variable && (f.estimate.kind === "mean" || f.estimate.methods.length > 0),
        crosstab: !!f.crosstab.row && !!f.crosstab.col && f.crosstab.row !== f.crosstab.col,
        likert: f.likert.items.length > 0,
      }[this.tab];
    },

    // ------------------------------------------------------------------ request bodies
    sizeBody() {
      const s = this.f.size;
      const dataset = this.hasDataset && s.strata_column;
      return {
        kind: s.kind, confidence: Number(s.confidence),
        margin_of_error: s.kind === "proportion" ? (num(s.margin) ?? 5) / 100 : num(s.margin),
        proportion: (num(s.proportion) ?? 50) / 100, std_dev: num(s.std_dev), population_size: num(s.population),
        design_effect: num(s.deff) ?? 1,
        dataset_id: dataset ? Alpine.store("datasets").activeId : null, strata_column: dataset ? s.strata_column : null,
        allocation: s.allocation, allocation_sd_column: text(s.sd_column),
      };
    },
    drawBody() {
      const d = this.f.draw;
      return {
        dataset_id: Alpine.store("datasets").activeId, design: d.design, sample_size: num(d.n), seed: num(d.seed),
        strata_column: text(d.strata_column), allocation: d.allocation, sd_column: text(d.sd_column),
        cluster_column: text(d.cluster_column), n_clusters: num(d.n_clusters), save_as: text(d.save_as),
      };
    },
    estimateBody() {
      const e = this.f.estimate;
      return {
        dataset_id: Alpine.store("datasets").activeId, variable: e.variable, kind: e.kind, confidence: Number(e.confidence),
        methods: e.methods.length ? e.methods : ["wilson"], population_size: num(e.population),
        weight_column: text(e.weight), strata_column: text(e.strata), cluster_column: text(e.cluster), apply_fpc: e.fpc,
      };
    },

    async submit() {
      if (!this.canRun) { this.error = this.needsDataset && !this.hasDataset ? "Elija un dataset." : "Complete los campos obligatorios."; return; }
      const store = Alpine.store("datasets");
      delete this.results[this.tab]; // a failed run must not leave the previous result behind
      let data = null;
      if (this.tab === "size") {
        data = await this.runAnalysis("/api/markets/sample-size", this.sizeBody());
        this.suggestedN = data?.result.n_final ?? this.suggestedN;
      } else if (this.tab === "draw") {
        data = await this.runAnalysis("/api/markets/draw", this.drawBody());
        if (data?.saved_dataset) await store.refresh();
      } else if (this.tab === "estimate") {
        await this.runAnalysis("/api/markets/estimate", this.estimateBody());
      } else if (this.tab === "crosstab") {
        await this.runAnalysis("/api/markets/crosstab", { dataset_id: store.activeId, row_variable: this.f.crosstab.row, col_variable: this.f.crosstab.col });
      } else {
        await this.runAnalysis("/api/markets/likert", { dataset_id: store.activeId, items: this.f.likert.items });
      }
    },

    useSuggestedSize() {
      this.f.draw.n = this.suggestedN;
    },

    // ------------------------------------------------------------------ reopening a saved result
    applyParams(p, saved) {
      const type = saved?.analysis_type ?? "";
      this.f = defaults();
      if (type.startsWith("sample_size")) {
        this.tab = "size";
        Object.assign(this.f.size, {
          kind: p.kind, confidence: p.confidence, margin: p.kind === "proportion" ? p.margin_of_error * 100 : p.margin_of_error,
          proportion: (p.proportion ?? 0.5) * 100, std_dev: p.std_dev ?? "", population: p.population_size ?? "",
          deff: p.design_effect ?? 1, strata_column: p.strata_column ?? "", allocation: p.allocation ?? "proportional",
          sd_column: p.allocation_sd_column ?? "",
        });
      } else if (type.startsWith("draw")) {
        this.tab = "draw";
        Object.assign(this.f.draw, {
          design: p.design, n: p.sample_size ?? "", seed: p.seed ?? "", strata_column: p.strata_column ?? "",
          allocation: p.allocation ?? "proportional", sd_column: p.sd_column ?? "", cluster_column: p.cluster_column ?? "",
          n_clusters: p.n_clusters ?? "", save_as: "",
        });
      } else if (type.startsWith("estimate")) {
        this.tab = "estimate";
        Object.assign(this.f.estimate, {
          kind: p.kind, variable: p.variable, confidence: p.confidence, methods: p.methods ?? ["wilson"],
          population: p.population_size ?? "", weight: p.weight_column ?? "", strata: p.strata_column ?? "",
          cluster: p.cluster_column ?? "", fpc: p.apply_fpc ?? true,
        });
      } else if (type === "crosstab") {
        this.tab = "crosstab";
        Object.assign(this.f.crosstab, { row: p.row_variable, col: p.col_variable });
      } else if (type === "likert") {
        this.tab = "likert";
        this.f.likert.items = existing(p.items ?? []);
      }
    },
  };
  });
});
