// Survey projects: design a questionnaire, capture responses and run the market-research analyses.

import { api } from "../api.js";
import { drawInterpretation, interpretationBlocks } from "../ai.js";
import { printPage } from "../export.js";
import { renderReport } from "../report.js";

const KIND_LABELS = { numeric: "Numérica", categorical: "Categórica", boolean: "Lógica", datetime: "Fecha", text: "Texto" };
const INDICATORS = [
  { key: "sample_size", label: "Tamaño de muestra" },
  { key: "estimate", label: "Estimación (proporción/media)" },
  { key: "crosstab", label: "Tabulación cruzada" },
  { key: "likert", label: "Escala Likert" },
];

let manualSheet = null;
let manualSheetData = null;

function defaultQuestion() {
  return { name: "", kind: "numeric", numericType: "decimal", decimals: 2, categories: "" };
}

function columnDef(question, index) {
  const title = (question.name || "").trim() || `P${index + 1}`;
  const base = { title, width: 150 };
  switch (question.kind) {
    case "numeric": return { ...base, type: "numeric", decimal: "." };
    case "datetime": return { ...base, type: "calendar", options: { format: "YYYY-MM-DD" } };
    case "boolean": return { ...base, type: "dropdown", source: ["Sí", "No"] };
    case "categorical": {
      const cats = (question.categories || "").split(",").map((s) => s.trim()).filter(Boolean);
      return cats.length ? { ...base, type: "dropdown", source: cats } : { ...base, type: "text" };
    }
    default: return { ...base, type: "text" };
  }
}

function questionNames() {
  return (Alpine.store("encuestas")?.project?.design || []).map((q, i) => (q.name || "").trim() || `P${i + 1}`);
}

// stored (typed) value -> the string shown in the grid
function gridCell(kind, value) {
  if (value === null || value === undefined) return "";
  switch (kind) {
    case "boolean": return value ? "Sí" : "No";
    case "datetime": return String(value).slice(0, 10);
    case "numeric": return value === "" ? "" : String(value);
    default: return String(value);
  }
}

document.addEventListener("alpine:init", () => {
  Alpine.store("encuestas", {
    list: [],
    loaded: false,
    project: null, // the full selected project
    busy: false,

    async refresh() {
      this.loaded = false;
      const { projects } = await api("/api/projects");
      this.list = projects;
      this.loaded = true;
    },

    async open(id) {
      this.busy = true;
      try {
        const data = await api(`/api/projects/${id}`);
        this.project = data.project;
        manualSheet = null;
        manualSheetData = null;
        if (data.data) {
          const names = data.data.columns.map((c) => c.name);
          const kinds = data.data.columns.map((c) => c.kind);
          manualSheetData = data.data.rows.map((row) => Object.fromEntries(names.map((name, j) => [name, gridCell(kinds[j], row[j])])));
        }
      } finally {
        this.busy = false;
      }
    },

    select(id) { this.project = { id, design: [], indicators: [], results: null, dataset_id: null, dataset_name: null, n_responses: 0 }; },
  });

  Alpine.data("encuestasPage", () => ({
    kindLabels: KIND_LABELS,
    indicators: INDICATORS,
    tab: "diseno",
    error: "",
    sheetReady: false,

    async init() {
      await Alpine.store("encuestas").refresh();
    },

    get store() { return Alpine.store("encuestas"); },
    get project() { return this.store.project; },

    formatDate(iso) {
      return iso ? new Date(iso).toLocaleString("es", { dateStyle: "medium", timeStyle: "short" }) : "";
    },

    // ------------------------------------------------------------------ project list
    async create() {
      const name = window.prompt("Nombre del proyecto:");
      if (!name || !name.trim()) return;
      try {
        const { project } = await api("/api/projects", { method: "POST", json: { name: name.trim() } });
        await this.store.refresh();
        await this.store.open(project.id);
        this.tab = "diseno";
      } catch (error) {
        this.error = error.message;
      }
    },

    async remove(project) {
      if (!window.confirm(`¿Eliminar el proyecto «${project.name}»?`)) return;
      try {
        await api(`/api/projects/${project.id}`, { method: "DELETE" });
        if (this.project?.id === project.id) this.project = null;
        await this.store.refresh();
        Alpine.store("toast").show("Proyecto eliminado.", "success");
      } catch (error) {
        this.error = error.message;
      }
    },

    async save() {
      const p = this.project;
      if (!p) return;
      this.error = "";
      try {
        const body = {
          name: p.name, population_size: p.population_size, sample_size: p.sample_size,
          indicators: p.indicators || [], design: p.design || [], results: p.results,
        };
        if (manualSheet) body.data = this.commitData(); // capture the responses when the grid is on screen
        const { project } = await api(`/api/projects/${p.id}`, { method: "PUT", json: body });
        this.store.project = project;
        await this.store.refresh();
        Alpine.store("toast").show("Proyecto guardado.", "success");
      } catch (error) {
        this.error = error.message;
      }
    },

    // ------------------------------------------------------------------ design
    addQuestion() {
      this.project.design.push(defaultQuestion());
    },
    removeQuestion(i) {
      this.project.design.splice(i, 1);
      if (manualSheet) this._buildSheet();
    },
    async selectTab(tab) {
      this.tab = tab;
      this.error = "";
      if (tab === "datos") { this.$nextTick(() => this._buildSheet()); }
      if (tab === "resultados") { this.$nextTick(() => this.renderResults()); }
    },

    // ------------------------------------------------------------------ data grid
    _snapshotSheet() {
      if (!manualSheet || !manualSheet[0]) return;
      const names = questionNames();
      const data = manualSheet[0].getData(false, true);
      manualSheetData = data.map((row) => Object.fromEntries(names.map((name, j) => [name, row[j] ?? ""])));
    },
    _destroySheet() {
      if (manualSheet) { try { jspreadsheet.destroy(this.$refs.manualSheet); } catch (error) { /* ignore */ } manualSheet = null; }
    },
    _onSheetKeydown(event) {
      const sheet = manualSheet?.[0];
      if (!sheet) return;
      const [col] = sheet.getHighlighted()[0] ?? [-1];
      const question = this.project?.design?.[col];
      if (!question || question.kind !== "numeric") return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const key = event.key;
      if (key.length > 1) return;
      if (/[0-9]/.test(key)) return;
      if (key === "-") return;
      if (question.numericType === "decimal" && key === ".") return;
      event.preventDefault();
      event.stopPropagation();
    },
    _onSheetClick(event) {
      const sheet = manualSheet?.[0];
      if (!sheet) return;
      const td = event.target.closest("td");
      if (!td) return;
      const [col] = sheet.getHighlighted()[0] ?? [-1];
      const question = this.project?.design?.[col];
      if (!question) return;
      const type = columnDef(question, col).type;
      if (type === "dropdown" || type === "calendar") sheet.openEditor(td, false, event);
    },
    _buildSheet() {
      this._snapshotSheet();
      this._destroySheet();
      const design = this.project?.design ?? [];
      const names = design.map((q, i) => (q.name || "").trim() || `P${i + 1}`);
      const columns = design.map((q, i) => columnDef(q, i));
      const data = (manualSheetData ?? [{}]).map((obj) => names.map((name) => obj[name] ?? ""));
      const height = this.$refs.manualSheet.clientHeight || 560;
      manualSheet = jspreadsheet(this.$refs.manualSheet, {
        worksheets: [{
          data,
          columns,
          minDimensions: [Math.max(1, columns.length), Math.max(data.length + 1, 200)],
          minSpareRows: 30,
          minSpareCols: 0,
          tableOverflow: true,
          tableHeight: height,
          tableWidth: "100%",
          allowInsertRow: true,
          allowDeleteRow: true,
          allowInsertColumn: false,
          allowDeleteColumn: false,
          allowManualInsertRow: true,
          allowManualInsertColumn: false,
          columnSorting: false,
        }],
      });
      const keydown = (event) => this._onSheetKeydown(event);
      const click = (event) => this._onSheetClick(event);
      this.$refs.manualSheet.addEventListener("keydown", keydown, true);
      this.$refs.manualSheet.addEventListener("click", click, false);
      this.sheetReady = true;
    },
    commitData() {
      this._snapshotSheet();
      const names = questionNames();
      return (manualSheetData ?? [])
        .map((obj) => Object.fromEntries(names.map((n) => [n, obj[n] ?? ""])))
        .filter((row) => names.some((n) => String(row[n] ?? "").trim() !== ""));
    },

    // ------------------------------------------------------------------ analysis
    indicatorEnabled(key) {
      return (this.project?.indicators ?? []).includes(key);
    },
    async runAnalysis(key) {
      const p = this.project;
      if (!p) return;
      this.error = "";
      try {
        const params = this._analysisParams(key);
        const payload = await api(`/api/projects/${p.id}/analysis`, { method: "POST", json: { analysis_type: key, params } });
        p.results = { ...(p.results || {}), [key]: payload };
        await this.save();
        this.$nextTick(() => this.renderResults());
      } catch (error) {
        this.error = error.message;
      }
    },
    renderResults() {
      for (const ind of this.indicators) {
        const el = document.getElementById(`project-result-${ind.key}`);
        const payload = this.project?.results?.[ind.key];
        if (el) renderReport(el, payload?.report ?? { title: "", blocks: [] }, payload?.figures ?? []);
      }
    },
    exportResults() {
      const p = this.project;
      if (!p) return;
      const sections = this.indicators
        .filter((ind) => p.results?.[ind.key])
        .map((ind) => {
          const payload = p.results[ind.key];
          return {
            title: ind.label,
            subtitle: payload.report?.subtitle ?? null,
            meta: [`Proyecto: ${p.name}`],
            blocks: payload.report?.blocks ?? [],
            figures: payload.figures ?? [],
          };
        });
      if (!sections.length) { this.error = "No hay resultados que exportar. Calcule primero un indicador."; return; }
      if (this.ai.interpretation) {
        sections.push({
          title: "Interpretación (IA)",
          subtitle: null,
          meta: [`Proyecto: ${p.name}`],
          blocks: interpretationBlocks(this.ai.interpretation),
          figures: [],
        });
      }
      Alpine.store("export").show({
        title: `Informe de encuesta — ${p.name}`,
        heading: "Exportar informe del proyecto",
        sections: () => Promise.resolve(sections),
      });
    },
    // ------------------------------------------------------------------ AI interpretation (all results together)
    ai: { interpretation: null, error: "" },
    aiAvailable() { return Alpine.store("ai").available; },
    async askAi() {
      const p = this.project;
      if (!p) return;
      this.ai.error = "";
      const blocks = [];
      for (const ind of this.indicators) {
        const payload = p.results?.[ind.key];
        if (payload?.report) {
          blocks.push({ type: "heading", text: ind.label });
          blocks.push(...(payload.report.blocks ?? []));
        }
      }
      if (!blocks.length) { this.ai.error = "Calcule primero algún indicador para poder interpretarlo."; return; }
      try {
        const interpretation = await Alpine.store("ai").interpret({
          module: "encuestas", title: `Encuesta: ${p.name}`, report: { title: p.name, blocks },
        });
        if (interpretation) this.ai.interpretation = interpretation;
      } catch (error) {
        this.ai.error = error.message;
      }
    },
    drawAi(element) {
      drawInterpretation(element, this.ai.interpretation);
    },
    _analysisParams(key) {
      const p = this.project;
      if (key === "sample_size") {
        return { kind: "proportion", confidence: 0.95, margin_of_error: 0.05, proportion: 0.5 };
      }
      if (key === "estimate") {
        const numeric = p.design.find((q) => q.kind === "numeric");
        const categorical = p.design.find((q) => q.kind === "categorical" || q.kind === "boolean");
        const variable = categorical ? categorical.name : (numeric ? numeric.name : "");
        return { variable, kind: categorical ? "proportion" : "mean", confidence: 0.95, methods: ["wilson"] };
      }
      if (key === "crosstab") {
        const cats = p.design.filter((q) => q.kind === "categorical" || q.kind === "boolean");
        return { row_variable: cats[0]?.name ?? "", col_variable: cats[1]?.name ?? "" };
      }
      if (key === "likert") {
        return { items: p.design.filter((q) => q.kind === "numeric").map((q) => q.name) };
      }
      return {};
    },

    printProject() { printPage(); },
  }));
});
