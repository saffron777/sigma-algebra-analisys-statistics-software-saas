// Data page: import from CSV / Excel / text / SQL Server / JSON API, review, save, and manage datasets.

import { api } from "../api.js";
import { printPage } from "../export.js";
import { workspace } from "../workspace.js";

const KIND_LABELS = { numeric: "numérica", categorical: "categórica", boolean: "lógica", datetime: "fecha", text: "texto" };
const SOURCE_LABELS = { csv: "CSV", excel: "Excel", text: "Texto", sqlserver: "SQL Server", json_api: "API JSON", sample: "Muestra" };
const FILE_SOURCES = new Set(["csv", "excel", "text"]);
const PAGE_SIZE = 50;

const defaultOptions = () => ({
  csv: { delimiter: "auto", encoding: "auto", decimal: "auto", header: true, skip_rows: 0 },
  excel: { sheet: "", header_row: 1 },
  text: { layout: "delimited", delimiter: "auto", encoding: "auto", decimal: "auto", header: true, widths: "", skip_rows: 0 },
  sqlserver: { server: "", database: "", auth: "windows", username: "", password: "", trust_server_certificate: false,
               mode: "query", query: "", table: "", schema: "", max_rows: 1000000 },
  json_api: { url: "", method: "GET", headers: "", params: "", body: "", records_path: "", timeout: 30 },
});

// what must never be stored in the workspace: passwords and anything that may carry a token
const SECRET_FIELDS = { sqlserver: ["password"], json_api: ["headers", "body", "params"] };

document.addEventListener("alpine:init", () => {
  Alpine.data("datosPage", () => ({
    ...workspace("datos", ["source", "saveName", "kinds"], { dataset: false }),
    kindLabels: KIND_LABELS,
    sourceLabels: SOURCE_LABELS,
    sources: [
      { key: "csv", label: "CSV" }, { key: "excel", label: "Excel" }, { key: "text", label: "Texto" },
      { key: "sqlserver", label: "SQL Server" }, { key: "json_api", label: "API JSON" },
    ],
    source: "csv",
    opts: defaultOptions(),
    drafts: {}, // pending imports on the server, by source type
    sheets: [],
    busy: false,
    error: "",
    preview: null,
    saveName: "",
    kinds: {},

    viewer: { dataset: null, columns: [], rows: [], offset: 0, total: 0, kinds: {}, busy: false },

    async init() {
      await this.startWorkspace();
      await this.loadDrafts();
      // another window uploaded, previewed, saved or removed a pending import
      Alpine.store("sync").on("drafts", () => this.loadDrafts());
    },

    // ------------------------------------------------------------------ workspace (the form; secrets are never kept)
    workspaceState() {
      const opts = JSON.parse(JSON.stringify(this.opts));
      const empty = defaultOptions();
      for (const [source, names] of Object.entries(SECRET_FIELDS)) for (const name of names) opts[source][name] = empty[source][name];
      opts.json_api.url = String(opts.json_api.url ?? "").split(/[?#]/)[0]; // a token often travels in the query string
      return {
        source: this.source, opts, saveName: this.saveName, kinds: { ...this.kinds },
        viewerId: this.viewer.dataset?.id ?? null, viewerOffset: this.viewer.offset,
      };
    },
    async applyWorkspace(state) {
      this.source = state.source in this.sourceLabels ? state.source : "csv";
      const defaults = defaultOptions();
      this.opts = Object.fromEntries(Object.keys(defaults).map((key) => [key, { ...defaults[key], ...(state.opts?.[key] ?? {}) }]));
      this.saveName = state.saveName ?? "";
      this.kinds = { ...(state.kinds ?? {}) };
      const dataset = Alpine.store("datasets").list.find((d) => d.id === state.viewerId);
      if (dataset) await this.view(dataset, state.viewerOffset ?? 0, false);
      await this.loadDrafts();
    },
    /** «Limpiar» (the server also discards the pending imports, which belong to this module's workspace). */
    async resetWorkspace() {
      this.source = "csv";
      this.opts = defaultOptions();
      this.drafts = {};
      this.sheets = [];
      this.preview = null;
      this.saveName = "";
      this.kinds = {};
      this.error = "";
      this.viewer.dataset = null;
      if (this.$refs.file) this.$refs.file.value = "";
    },

    // ------------------------------------------------------------------ pending imports (kept on the server)
    get isFileSource() { return FILE_SOURCES.has(this.source); },
    get draft() { return this.drafts[this.source] ?? null; },
    get fileAccept() {
      return { csv: ".csv,.tsv,text/csv", excel: ".xlsx,.xlsm,.xls", text: ".txt,.tsv,.dat,.prn,text/plain" }[this.source];
    },

    /** Read the pending imports and show the one of the selected source again (file card and, if any, its preview). */
    async loadDrafts() {
      try {
        const { drafts } = await api("/api/datasets/import/drafts");
        this.drafts = Object.fromEntries(drafts.map((draft) => [draft.source_type, draft]));
      } catch { return; }
      await this.showDraft();
    },

    async showDraft() {
      const draft = this.draft;
      if (!draft) { this.preview = null; this.sheets = []; return; }
      if (this.source === "excel" && !this.sheets.length) await this.loadSheets();
      if (!draft.has_preview) { if (this.preview?.token === draft.id) this.preview = null; return; }
      if (this.preview?.token === draft.id && this.preview.updated === draft.updated_at) return; // already on screen
      try {
        this.preview = { ...(await api(`/api/datasets/import/drafts/${draft.id}`)), updated: draft.updated_at };
        if (!this.saveName.trim()) this.saveName = this.preview.suggested_name;
      } catch { this.preview = null; }
    },

    async selectSource(key) {
      this.source = key;
      this.sheets = [];
      this.error = "";
      this.preview = null;
      if (this.$refs.file) this.$refs.file.value = "";
      await this.showDraft();
    },

    async onFile(event) {
      const file = event.target.files[0] ?? null;
      if (this.$refs.file) this.$refs.file.value = ""; // the same file can be chosen again later
      if (!file) return;
      this.preview = null;
      this.error = "";
      this.sheets = [];
      this.busy = true;
      try {
        const form = new FormData();
        form.append("source_type", this.source);
        form.append("file", file);
        const { draft } = await api("/api/datasets/import/draft", { method: "POST", form });
        this.drafts = { ...this.drafts, [draft.source_type]: draft };
        if (this.source === "excel") await this.loadSheets();
      } catch (error) {
        this.error = error.message;
      } finally {
        this.busy = false;
      }
    },

    /** «Quitar»: forget the pending import of this source (file and preview). */
    async removeDraft() {
      const draft = this.draft;
      if (!draft) return;
      try {
        await api(`/api/datasets/import/drafts/${draft.id}`, { method: "DELETE" });
      } catch (error) {
        this.error = error.message;
        return;
      }
      const { [this.source]: _removed, ...rest } = this.drafts;
      this.drafts = rest;
      this.preview = null;
      this.sheets = [];
      this.kinds = {};
    },

    buildOptions() {
      const options = { ...this.opts[this.source] };
      if (this.source === "sqlserver") {
        if (options.mode === "query") { delete options.table; delete options.schema; } else { delete options.query; }
        delete options.mode;
      }
      for (const key of Object.keys(options)) if (options[key] === "") delete options[key];
      return options;
    },

    /** The request that reads this source: a file source names its stored file (no new upload). */
    requestInit() {
      const json = { source_type: this.source, options: this.buildOptions() };
      if (this.isFileSource) json.draft_id = this.draft?.id;
      return { method: "POST", json };
    },

    async loadSheets() {
      if (!this.draft) return;
      try {
        const { sheets } = await api("/api/datasets/import/inspect", this.requestInit());
        this.sheets = sheets;
        if (!sheets.includes(this.opts.excel.sheet)) this.opts.excel.sheet = sheets[0] ?? "";
      } catch (error) {
        this.error = error.message;
      }
    },

    async previewImport() {
      if (this.isFileSource && !this.draft) { this.error = "Seleccione un archivo."; return; }
      this.busy = true;
      this.error = "";
      try {
        const preview = await api("/api/datasets/import/preview", this.requestInit());
        this.preview = preview;
        this.saveName = preview.suggested_name;
        this.kinds = {};
        const { drafts } = await api("/api/datasets/import/drafts");
        this.drafts = Object.fromEntries(drafts.map((draft) => [draft.source_type, draft]));
        this.preview.updated = this.draft?.updated_at;
        await this.$nextTick();
        this.$refs.previewCard?.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        this.preview = null;
        this.error = error.message;
      } finally {
        this.busy = false;
      }
    },

    async commitImport() {
      this.busy = true;
      this.error = "";
      try {
        const kinds = Object.fromEntries(Object.entries(this.kinds).filter(([, kind]) => kind));
        const { dataset } = await api("/api/datasets/import/commit", {
          method: "POST", json: { token: this.preview.token, name: this.saveName, column_kinds: kinds },
        });
        Alpine.store("toast").show(`Dataset «${dataset.name}» guardado.`, "success");
        const { [this.source]: _saved, ...rest } = this.drafts; // the server removed the pending import
        this.drafts = rest;
        this.preview = null;
        this.sheets = [];
        this.kinds = {};
        this.saveName = "";
        await Alpine.store("datasets").refresh();
        Alpine.store("datasets").select(dataset.id);
      } catch (error) {
        this.error = error.message;
      } finally {
        this.busy = false;
      }
    },

    /** «Descartar» the preview: a file stays available to be analysed again with other options. */
    async cancelPreview() {
      const draft = this.draft;
      this.preview = null;
      this.error = "";
      if (!draft) return;
      try {
        await api(`/api/datasets/import/drafts/${draft.id}${this.isFileSource ? "?preview_only=1" : ""}`, { method: "DELETE" });
      } catch { /* already gone */ }
      await this.loadDrafts();
    },

    formatSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      return bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;
    },

    async loadSamples() {
      this.busy = true;
      try {
        const { created, skipped } = await api("/api/datasets/samples", { method: "POST" });
        await Alpine.store("datasets").refresh();
        const message = created.length
          ? `Se cargaron ${created.length} dataset(s) de ejemplo.`
          : `Los datos de ejemplo ya estaban cargados (${skipped.length}).`;
        Alpine.store("toast").show(message, "success");
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      } finally {
        this.busy = false;
      }
    },

    // ------------------------------------------------------------------ dataset management
    sourceLabel(type) { return SOURCE_LABELS[type] ?? type; },
    kindLabel(kind) { return KIND_LABELS[kind] ?? kind; },
    formatCell(value) {
      if (value === null || value === undefined) return "";
      return typeof value === "number" ? String(Math.round(value * 1e6) / 1e6) : String(value);
    },
    formatDate(iso) {
      return iso ? new Date(iso).toLocaleString("es", { dateStyle: "medium", timeStyle: "short" }) : "";
    },

    async view(dataset, offset = 0, scroll = true) {
      const viewer = this.viewer;
      viewer.busy = true;
      try {
        const page = await api(`/api/datasets/${dataset.id}/preview?limit=${PAGE_SIZE}&offset=${offset}`);
        Object.assign(viewer, { dataset: page.dataset, columns: page.columns, rows: page.rows, offset, total: page.total });
        if (offset === 0) viewer.kinds = {};
        await this.$nextTick();
        if (scroll) this.$refs.viewerCard?.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      } finally {
        viewer.busy = false;
      }
    },

    page(delta) {
      const next = this.viewer.offset + delta * PAGE_SIZE;
      if (next >= 0 && next < this.viewer.total) this.view(this.viewer.dataset, next);
    },

    closeViewer() { this.viewer.dataset = null; },

    printViewer() { printPage(); },

    /** «Exportar ficha»: summary, columns, statistics and first rows (Excel also gets every row). */
    exportFicha(dataset = this.viewer.dataset) {
      Alpine.store("export").show({
        title: `Dataset ${dataset.name}`, heading: "Exportar la ficha del dataset", dataset: dataset.id,
      });
    },

    async applyTypes() {
      const kinds = Object.fromEntries(Object.entries(this.viewer.kinds).filter(([, kind]) => kind));
      if (!Object.keys(kinds).length) { Alpine.store("toast").show("Elija el nuevo tipo de al menos una columna.", "info"); return; }
      try {
        const { dataset, failed_conversions: failed } = await api(`/api/datasets/${this.viewer.dataset.id}/column-types`, {
          method: "POST", json: { column_kinds: kinds },
        });
        const lost = Object.entries(failed).filter(([, count]) => count > 0);
        const detail = lost.length ? ` Valores no convertibles (quedaron vacíos): ${lost.map(([c, n]) => `${c}: ${n}`).join(", ")}.` : "";
        Alpine.store("toast").show(`Tipos actualizados.${detail}`, lost.length ? "info" : "success", lost.length ? 7000 : 3600);
        await Alpine.store("datasets").refresh();
        await this.view(dataset, this.viewer.offset);
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
    },

    async rename(dataset) {
      const name = window.prompt("Nuevo nombre del dataset:", dataset.name);
      if (!name || name.trim() === dataset.name) return;
      try {
        await api(`/api/datasets/${dataset.id}`, { method: "PATCH", json: { name } });
        await Alpine.store("datasets").refresh();
        if (this.viewer.dataset?.id === dataset.id) this.viewer.dataset.name = name.trim();
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
    },

    async remove(dataset) {
      if (!window.confirm(`¿Eliminar el dataset «${dataset.name}»? Los resultados guardados se conservan.`)) return;
      try {
        await api(`/api/datasets/${dataset.id}`, { method: "DELETE" });
        if (this.viewer.dataset?.id === dataset.id) this.closeViewer();
        await Alpine.store("datasets").refresh();
        Alpine.store("toast").show("Dataset eliminado.", "success");
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
    },
  }));
});
