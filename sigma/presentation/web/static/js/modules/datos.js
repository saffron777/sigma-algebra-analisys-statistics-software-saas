// Data page: import from CSV / Excel / text / SQL Server / JSON API, review, save, and manage datasets.

import { api } from "../api.js";

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

document.addEventListener("alpine:init", () => {
  Alpine.data("datosPage", () => ({
    kindLabels: KIND_LABELS,
    sourceLabels: SOURCE_LABELS,
    sources: [
      { key: "csv", label: "CSV" }, { key: "excel", label: "Excel" }, { key: "text", label: "Texto" },
      { key: "sqlserver", label: "SQL Server" }, { key: "json_api", label: "API JSON" },
    ],
    source: "csv",
    opts: defaultOptions(),
    file: null,
    sheets: [],
    busy: false,
    error: "",
    preview: null,
    saveName: "",
    kinds: {},

    viewer: { dataset: null, columns: [], rows: [], offset: 0, total: 0, kinds: {}, busy: false },

    // ------------------------------------------------------------------ import flow
    get isFileSource() { return FILE_SOURCES.has(this.source); },
    get fileAccept() {
      return { csv: ".csv,.tsv,text/csv", excel: ".xlsx,.xlsm,.xls", text: ".txt,.tsv,.dat,.prn,text/plain" }[this.source];
    },

    selectSource(key) {
      this.source = key;
      this.file = null;
      this.sheets = [];
      this.error = "";
      this.preview = null;
      if (this.$refs.file) this.$refs.file.value = "";
    },

    async onFile(event) {
      this.file = event.target.files[0] ?? null;
      this.preview = null;
      this.error = "";
      this.sheets = [];
      if (this.file && this.source === "excel") await this.loadSheets();
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

    requestInit() {
      const options = this.buildOptions();
      if (this.isFileSource) {
        const form = new FormData();
        form.append("source_type", this.source);
        form.append("options", JSON.stringify(options));
        if (this.file) form.append("file", this.file);
        return { method: "POST", form };
      }
      return { method: "POST", json: { source_type: this.source, options } };
    },

    async loadSheets() {
      try {
        const { sheets } = await api("/api/datasets/import/inspect", this.requestInit());
        this.sheets = sheets;
        this.opts.excel.sheet = sheets[0] ?? "";
      } catch (error) {
        this.error = error.message;
      }
    },

    async previewImport() {
      if (this.isFileSource && !this.file) { this.error = "Seleccione un archivo."; return; }
      this.busy = true;
      this.error = "";
      try {
        this.preview = await api("/api/datasets/import/preview", this.requestInit());
        this.saveName = this.preview.suggested_name;
        this.kinds = {};
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
        this.preview = null;
        this.file = null;
        if (this.$refs.file) this.$refs.file.value = "";
        await Alpine.store("datasets").refresh();
        Alpine.store("datasets").select(dataset.id);
      } catch (error) {
        this.error = error.message;
      } finally {
        this.busy = false;
      }
    },

    cancelPreview() {
      this.preview = null;
      this.error = "";
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

    async view(dataset, offset = 0) {
      const viewer = this.viewer;
      viewer.busy = true;
      try {
        const page = await api(`/api/datasets/${dataset.id}/preview?limit=${PAGE_SIZE}&offset=${offset}`);
        Object.assign(viewer, { dataset: page.dataset, columns: page.columns, rows: page.rows, offset, total: page.total });
        if (offset === 0) viewer.kinds = {};
        await this.$nextTick();
        this.$refs.viewerCard?.scrollIntoView({ behavior: "smooth", block: "start" });
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
