// Generic page of the analysis catalogue: the form is built from what the server declares for the module
// (/api/analysis/<module>/catalog), so adding an analysis on the server adds it here.

import { api } from "../api.js";
import { analysisPage } from "../analysis-page.js";

const KINDS = {
  numeric: ["numeric", "boolean"],
  categorical: ["categorical", "boolean", "text"],
  datetime: ["datetime"],
  any: [],
};
const toNumber = (value) => (value === "" || value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value));
const clone = (value) => JSON.parse(JSON.stringify(value));

document.addEventListener("alpine:init", () => {
  Alpine.data("catalogoPage", (module) => ({
    ...analysisPage(module, ["analysisId", "roles", "params", "saveAs", "replace"]),
    info: null,
    analyses: [],
    analysisId: "",
    roles: {},
    params: {},
    saveAs: {},
    replace: false,
    created: [], // datasets the last run left behind
    code: "",
    codeOpen: false,
    codeError: "",

    async init() {
      const data = await api(`/api/analysis/${module}/catalog`);
      this.info = data.module;
      this.analyses = data.analyses;
      if (this.analyses.length) this.selectAnalysis(this.analyses[0].id);
      this.onDatasetChange(() => { this.roles = this.emptyRoles(); });
      await this.startWorkspace();
    },

    // ------------------------------------------------------------------ the catalogue
    get spec() { return this.analyses.find((a) => a.id === this.analysisId) ?? null; },
    get grouped() {
      const groups = new Map();
      for (const analysis of this.analyses) groups.set(analysis.group, [...(groups.get(analysis.group) ?? []), analysis]);
      return [...groups].map(([group, items]) => ({ group, items }));
    },
    /** The parameters that apply with the current values (some only matter for one option of another). */
    get visibleParams() {
      return (this.spec?.params ?? []).filter((f) => !f.visible_if || f.visible_if.values.includes(String(this.params[f.visible_if.param])));
    },
    columnsFor(role) {
      return Alpine.store("datasets").columnsOfKind(KINDS[role.kind] ?? []);
    },
    emptyRoles() {
      return Object.fromEntries((this.spec?.roles ?? []).map((r) => [r.name, r.multiple ? [] : ""]));
    },

    selectAnalysis(id) {
      this.analysisId = id;
      const spec = this.spec;
      this.roles = this.emptyRoles();
      this.params = Object.fromEntries(spec.params.map((p) => [p.name, p.kind === "multiselect" ? [...(p.default ?? [])] : p.default ?? (p.kind === "boolean" ? false : "")]));
      this.saveAs = Object.fromEntries(spec.outputs.map((o) => [o.key, ""]));
      this.replace = false;
      this.created = [];
      this.code = "";
      this.codeError = "";
    },

    resetWorkspace() {
      if (this.analyses.length) this.selectAnalysis(this.analyses[0].id);
      this.codeOpen = false;
    },

    /** A stored form: choose the analysis first (it sets that analysis' defaults), then overlay what was typed. */
    applyWorkspace(state) {
      if (!this.analyses.some((a) => a.id === state.analysisId)) return;
      this.selectAnalysis(state.analysisId);
      this.applyParams({ analysis_id: state.analysisId, roles: state.roles, params: state.params, save_as: state.saveAs, replace: state.replace });
    },

    /** A saved result reopened (``?load=``) or a restored workspace. */
    applyParams(p) {
      if (!this.analyses.length) return;
      if (p.analysis_id && p.analysis_id !== this.analysisId && this.analyses.some((a) => a.id === p.analysis_id)) this.selectAnalysis(p.analysis_id);
      this.roles = { ...this.emptyRoles(), ...clone(p.roles ?? {}) };
      this.params = { ...this.params, ...clone(p.params ?? {}) };
      this.saveAs = { ...this.saveAs, ...(p.save_as ?? {}) };
      this.replace = !!p.replace;
    },

    // ------------------------------------------------------------------ running
    convertParams() {
      const out = {};
      for (const field of this.spec.params) {
        const value = this.params[field.name];
        out[field.name] = field.kind === "number" || field.kind === "integer" ? toNumber(value) : value;
      }
      return out;
    },
    cleanRoles() {
      const out = {};
      for (const role of this.spec.roles) {
        const value = this.roles[role.name];
        if (role.multiple ? value?.length : value) out[role.name] = value;
      }
      return out;
    },
    body() {
      return {
        analysis_id: this.analysisId,
        dataset_id: this.spec.needs_dataset ? Alpine.store("datasets").activeId || null : null,
        roles: this.cleanRoles(),
        params: this.convertParams(),
        save_as: Object.fromEntries(Object.entries(this.saveAs).filter(([, name]) => (name ?? "").trim())),
        replace: this.replace,
      };
    },

    get canRun() {
      return !!this.spec && (!this.spec.needs_dataset || !!Alpine.store("datasets").activeId);
    },

    async submit() {
      if (!this.canRun) { this.error = "Elija un dataset."; return; }
      const data = await this.runAnalysis(`/api/analysis/${module}/run`, this.body());
      this.created = data?.datasets ?? [];
      if (this.created.length) await Alpine.store("datasets").refresh();
    },

    // ------------------------------------------------------------------ the equivalent Python
    async toggleCode() {
      this.codeOpen = !this.codeOpen;
      if (this.codeOpen) await this.loadCode();
    },
    async loadCode() {
      this.codeError = "";
      try {
        const { analysis_id, dataset_id, roles, params } = this.body();
        this.code = (await api(`/api/analysis/${module}/code`, { method: "POST", json: { analysis_id, dataset_id, roles, params } })).code;
      } catch (error) {
        this.code = "";
        this.codeError = error.message;
      }
    },
    async copyCode() {
      try {
        await navigator.clipboard.writeText(this.code);
        Alpine.store("toast").show("Código copiado.", "success", 2200);
      } catch { Alpine.store("toast").show("No se pudo copiar al portapapeles.", "error"); }
    },
  }));
});
