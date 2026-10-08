// Data page: import from CSV / Excel / text / external databases / JSON API, review, save, and manage datasets.

import { api } from "../api.js";
import { drawInterpretation } from "../ai.js";
import { printPage } from "../export.js";
import { workspace } from "../workspace.js";

const KIND_LABELS = { numeric: "numérica", categorical: "categórica", boolean: "lógica", datetime: "fecha", text: "texto" };
const SOURCE_LABELS = {
  csv: "CSV", excel: "Excel", text: "Texto", sqlserver: "SQL Server", mysql: "MySQL / MariaDB", postgresql: "PostgreSQL",
  oracle: "Oracle", mongodb: "MongoDB", json_api: "API JSON", sample: "Muestra", manual: "Manual",
};
// the «Base de datos» tab reads from one of these; each engine is its own source type on the server
const DB_ENGINES = ["sqlserver", "mysql", "postgresql", "oracle", "mongodb"];
const TABS = new Set(["csv", "excel", "text", "database", "json_api"]);
const FILE_SOURCES = new Set(["csv", "excel", "text"]);
const PAGE_SIZE = 50;

const databaseOptions = (port) => ({
  host: "", port, database: "", username: "", password: "", mode: "query", query: "", table: "", schema: "", max_rows: 1000000,
});

const defaultOptions = () => ({
  csv: { delimiter: "auto", encoding: "auto", decimal: "auto", header: true, skip_rows: 0 },
  excel: { sheet: "", header_row: 1 },
  text: { layout: "delimited", delimiter: "auto", encoding: "auto", decimal: "auto", header: true, widths: "", skip_rows: 0 },
  database: { engine: "sqlserver" },
  sqlserver: { server: "", database: "", auth: "windows", username: "", password: "", trust_server_certificate: false,
               mode: "query", query: "", table: "", schema: "", max_rows: 1000000 },
  mysql: { ...databaseOptions(3306), ssl_disabled: false },
  postgresql: { ...databaseOptions(5432), sslmode: "" },
  oracle: databaseOptions(1521),
  mongodb: { host: "", port: 27017, database: "", username: "", password: "", auth_source: "", tls: false,
             collection: "", filter: "", fields: "", max_rows: 1000000 },
  json_api: { url: "", method: "GET", headers: "", params: "", body: "", records_path: "", timeout: 30 },
});

// what must never be stored in the workspace: passwords and anything that may carry a token
const SECRET_FIELDS = {
  sqlserver: ["password"], mysql: ["password"], postgresql: ["password"], oracle: ["password"], mongodb: ["password"],
  json_api: ["headers", "body", "params"],
};

// the Jspreadsheet instance lives outside Alpine so its Proxy does not wrap (and break) the third-party object
let manualSheet = null;
let manualSheetData = null; // last grid data (array of arrays), kept across column rebuilds
let manualResizeTimer = null;
let manualResizeHandler = null;
let manualKeydownHandler = null;
let manualClickHandler = null;

// Sigma variable kind -> Jspreadsheet v5 column definition
function manualColumn(variable, index) {
  const title = (variable.name || "").trim() || `V${index + 1}`;
  const base = { title, width: 140 };
  switch (variable.kind) {
    case "numeric": return { ...base, type: "numeric", decimal: "." };
    case "datetime": return { ...base, type: "calendar", options: { format: "YYYY-MM-DD" } };
    case "boolean": return { ...base, type: "dropdown", source: ["Sí", "No"] };
    case "categorical": {
      const categories = (variable.categories || "").split(",").map((s) => s.trim()).filter(Boolean);
      return categories.length ? { ...base, type: "dropdown", source: categories } : { ...base, type: "text" };
    }
    default: return { ...base, type: "text" }; // text is free entry
  }
}

function defaultVariable() {
  return { name: "", kind: "numeric", numericType: "decimal", decimals: 2, categories: "" };
}

// stored (typed) value -> the string shown in the editor grid
function gridCell(kind, value) {
  if (value === null || value === undefined) return "";
  switch (kind) {
    case "boolean": return value ? "Sí" : "No";
    case "datetime": return String(value).slice(0, 10); // "YYYY-MM-DD"
    case "numeric": return value === "" ? "" : String(value);
    default: return String(value);
  }
}

document.addEventListener("alpine:init", () => {
  Alpine.data("datosPage", () => ({
    ...workspace("datos", ["source", "saveName", "kinds"], { dataset: false }),
    kindLabels: KIND_LABELS,
    sourceLabels: SOURCE_LABELS,
    sources: [
      { key: "csv", label: "CSV" }, { key: "excel", label: "Excel" }, { key: "text", label: "Texto" },
      { key: "database", label: "Base de datos" }, { key: "json_api", label: "API JSON" },
    ],
    source: "csv",
    // the engines this server can read from, with their driver status (refreshed from the server)
    dbEngines: DB_ENGINES.map((key) => ({ source_type: key, label: SOURCE_LABELS[key], driver_available: true })),
    catalog: null, // what «Ver tablas» found: {tables, columns, schema, schemas}
    catalogBusy: false,
    catalogError: "",
    opts: defaultOptions(),
    drafts: {}, // pending imports on the server, by source type
    sheets: [],
    busy: false,
    error: "",
    preview: null,
    saveName: "",
    kinds: {},

    viewer: { dataset: null, columns: [], rows: [], offset: 0, total: 0, kinds: {}, busy: false, mode: "data" },

    /** Whether this user can ask the AI right now (the store explains why not when it cannot). */
    aiAvailable() {
      return Alpine.store("ai").available;
    },

    async init() {
      await this.startWorkspace();
      await this.loadDatabases();
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
      const defaults = defaultOptions();
      this.opts = Object.fromEntries(Object.keys(defaults).map((key) => [key, { ...defaults[key], ...(state.opts?.[key] ?? {}) }]));
      if (DB_ENGINES.includes(state.source)) { // saved by a version where SQL Server had a tab of its own
        this.source = "database";
        this.opts.database.engine = state.source;
      } else {
        this.source = TABS.has(state.source) ? state.source : "csv";
      }
      if (!DB_ENGINES.includes(this.opts.database.engine)) this.opts.database.engine = "sqlserver";
      this.catalog = null;
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
      this.catalog = null;
      this.catalogError = "";
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
    get isDatabase() { return this.source === "database"; },
    /** The source type the server knows: the engine chosen in the «Base de datos» tab, or the tab itself. */
    get apiSource() { return this.isDatabase ? this.opts.database.engine : this.source; },
    /** The option block of the source being edited. */
    get cur() { return this.opts[this.apiSource]; },
    get engineInfo() { return this.dbEngines.find((e) => e.source_type === this.opts.database.engine) ?? null; },
    get draft() { return this.drafts[this.apiSource] ?? null; },
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
      this.catalog = null;
      this.catalogError = "";
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
      const { [this.apiSource]: _removed, ...rest } = this.drafts;
      this.drafts = rest;
      this.preview = null;
      this.sheets = [];
      this.kinds = {};
    },

    buildOptions() {
      const options = { ...this.opts[this.apiSource] };
      if (this.isDatabase && this.apiSource !== "mongodb") { // SQL engines: either a query or a whole table
        if (options.mode === "query") { delete options.table; delete options.schema; } else { delete options.query; }
        delete options.mode;
      }
      for (const key of Object.keys(options)) if (options[key] === "") delete options[key];
      return options;
    },

    /** The request that reads this source: a file source names its stored file (no new upload). */
    requestInit() {
      const json = { source_type: this.apiSource, options: this.buildOptions() };
      if (this.isFileSource) json.draft_id = this.draft?.id;
      return { method: "POST", json };
    },

    // ------------------------------------------------------------------ external databases
    async loadDatabases() {
      try {
        this.dbEngines = (await api("/api/datasets/import/databases")).databases;
      } catch { /* the form still works; engines just are not greyed out */ }
    },

    changeEngine() {
      this.catalog = null;
      this.catalogError = "";
      this.error = "";
      this.preview = null;
      this.showDraft();
    },

    /** «Ver tablas» / «Ver colecciones»: what the server offers (and the columns of ``name`` when given). */
    async loadCatalog(name = "") {
      this.catalogBusy = true;
      this.catalogError = "";
      try {
        const options = this.buildOptions();
        delete options.query;
        delete options.max_rows;
        delete options.filter;
        delete options.fields;
        const key = this.apiSource === "mongodb" ? "collection" : "table";
        if (name) options[key] = name; else delete options[key];
        const found = await api("/api/datasets/import/inspect", { method: "POST", json: { source_type: this.apiSource, options } });
        this.catalog = { ...found, selected: name };
      } catch (error) {
        this.catalogError = error.message;
      } finally {
        this.catalogBusy = false;
      }
    },

    /** Choosing from the list fills the form in; the columns are fetched so the user sees what is in there. */
    async pickTable(table) {
      const block = this.cur;
      if (this.apiSource === "mongodb") {
        block.collection = table.name;
      } else {
        block.table = table.name;
        block.mode = "table";
        if ("schema" in block && this.catalog?.schema) block.schema = this.catalog.schema;
      }
      await this.loadCatalog(table.name);
    },

    async pickSchema(schema) {
      this.cur.schema = schema;
      await this.loadCatalog();
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
        const { [this.apiSource]: _saved, ...rest } = this.drafts; // the server removed the pending import
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

    // ------------------------------------------------------------------ AI interpretation of the dataset (data quality)
    ficha: { interpretation: null, error: "", for: "" },
    async askAiDataset() {
      const dataset = this.viewer.dataset;
      if (!dataset) return;
      this.ficha.error = "";
      try {
        const interpretation = await Alpine.store("ai").interpret({ module: "dataset", dataset_id: dataset.id });
        if (interpretation) { this.ficha.interpretation = interpretation; this.ficha.for = dataset.id; }
      } catch (error) {
        this.ficha.error = error.message;
      }
    },
    drawFichaInterpretation(element) {
      drawInterpretation(element, this.ficha.for === this.viewer.dataset?.id ? this.ficha.interpretation : null);
    },
    get fichaInterpretation() {
      return this.ficha.for === this.viewer.dataset?.id ? this.ficha.interpretation : null;
    },

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

    // ------------------------------------------------------------------ manual dataset editor
    editor: { open: false, tab: "variables", name: "", variables: [defaultVariable()], busy: false, error: "", mode: "create", datasetId: null },

    _attachResize() {
      manualResizeHandler = () => {
        clearTimeout(manualResizeTimer);
        manualResizeTimer = setTimeout(() => { if (manualSheet) this._buildSheet(); }, 200);
      };
      window.addEventListener("resize", manualResizeHandler);
    },

    openEditor() {
      manualSheet = null;
      manualSheetData = null;
      this.editor = { open: true, tab: "variables", name: "", variables: [defaultVariable()], busy: false, error: "", mode: "create", datasetId: null };
      this._attachResize();
    },

    async openEdit(dataset) {
      manualSheet = null;
      manualSheetData = null;
      const variables = (dataset.columns ?? []).map((c) => ({
        name: c.name, kind: c.kind, numericType: "decimal", decimals: 2, categories: "",
      }));
      this.editor = { open: true, tab: "datos", name: dataset.name, variables, busy: false, error: "", mode: "edit", datasetId: dataset.id };
      this._attachResize();
      try {
        const res = await api(`/api/datasets/${dataset.id}/preview?limit=${dataset.n_rows || 100000}&offset=0`);
        const kinds = (res.dataset?.columns ?? []).map((c) => c.kind);
        const names = variables.map((v) => v.name);
        manualSheetData = (res.rows ?? []).map((row) => Object.fromEntries(names.map((name, j) => [name, gridCell(kinds[j], row[j])])));
        this.$nextTick(() => this._buildSheet());
      } catch (error) {
        this.editor.error = error.message;
      }
    },

    closeEditor() {
      this._destroySheet();
      if (manualResizeHandler) { window.removeEventListener("resize", manualResizeHandler); manualResizeHandler = null; }
      this.editor.open = false;
    },

    setEditorTab(tab) {
      this.editor.tab = tab;
      if (tab === "datos") this.$nextTick(() => this._buildSheet());
    },

    addVariable() {
      this.editor.variables.push(defaultVariable());
      if (manualSheet) this._buildSheet();
    },

    removeVariable(index) {
      if (this.editor.variables.length <= 1) { Alpine.store("toast").show("El dataset necesita al menos una variable.", "info"); return; }
      this.editor.variables.splice(index, 1);
      if (manualSheet) this._buildSheet();
    },

    _sheetNames() {
      return this.editor.variables.map((v, i) => v.name.trim() || `V${i + 1}`);
    },

    _snapshotSheet() {
      if (!manualSheet || !manualSheet[0]) return;
      const names = this._sheetNames();
      const data = manualSheet[0].getData(false, true);
      manualSheetData = data.map((row) => Object.fromEntries(names.map((name, j) => [name, row[j] ?? ""])));
    },

    _destroySheet() {
      if (manualSheet) { try { jspreadsheet.destroy(this.$refs.manualSheet); } catch (error) { /* already destroyed */ } manualSheet = null; }
      if (manualKeydownHandler) { this.$refs.manualSheet.removeEventListener("keydown", manualKeydownHandler, true); manualKeydownHandler = null; }
      if (manualClickHandler) { this.$refs.manualSheet.removeEventListener("click", manualClickHandler, false); manualClickHandler = null; }
    },

    /** Numeric cells only accept digits (and `.` for decimals, `-` for negatives). */
    _onSheetKeydown(event) {
      const sheet = manualSheet?.[0];
      if (!sheet) return;
      const [col] = sheet.getHighlighted()[0] ?? [-1];
      const variable = this.editor.variables[col];
      if (!variable || variable.kind !== "numeric") return;
      if (event.ctrlKey || event.metaKey || event.altKey) return; // shortcuts
      const key = event.key;
      if (key.length > 1) return; // navigation/control keys (Backspace, arrows, Tab, Enter, Delete…)
      if (/[0-9]/.test(key)) return;
      if (key === "-") return; // negative sign
      if (variable.numericType === "decimal" && key === ".") return;
      event.preventDefault();
      event.stopPropagation();
    },

    /** Open dropdown/calendar editors with a single click (text/numeric keep double-click). */
    _onSheetClick(event) {
      const sheet = manualSheet?.[0];
      if (!sheet) return;
      const td = event.target.closest("td");
      if (!td) return;
      const [col] = sheet.getHighlighted()[0] ?? [-1];
      const variable = this.editor.variables[col];
      if (!variable) return;
      const columnType = manualColumn(variable, col).type;
      if (columnType === "dropdown" || columnType === "calendar") sheet.openEditor(td, false, event);
    },

    /** Toolbar actions that map onto the Jspreadsheet instance. */
    async sheetAction(action) {
      const sheet = manualSheet?.[0];
      if (!sheet) return;
      try {
        switch (action) {
          case "undo": sheet.undo(); break;
          case "redo": sheet.redo(); break;
          case "copy": sheet.copy(false); break;
          case "paste": {
            const [x1, y1] = sheet.getHighlighted()[0] ?? [0, 0];
            const text = await navigator.clipboard.readText();
            sheet.paste(x1, y1, text);
            break;
          }
          case "insertRow": sheet.insertRow(); break;
          case "deleteRow": sheet.deleteRow(); break;
          case "insertColumn": sheet.insertColumn(); break;
          case "deleteColumn": sheet.deleteColumn(); break;
        }
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
    },

    _buildSheet() {
      this._snapshotSheet();
      this._destroySheet();
      const names = this._sheetNames();
      const columns = this.editor.variables.map((v, i) => manualColumn(v, i));
      const data = (manualSheetData ?? [{}]).map((obj) => names.map((name) => obj[name] ?? ""));
      const height = this.$refs.manualSheet.clientHeight || 560;
      manualSheet = jspreadsheet(this.$refs.manualSheet, {
        // Round/truncate numeric values on commit (no visible mask while typing).
        onbeforechange: (instance, cell, colIndex, rowIndex, newValue) => {
          const variable = this.editor.variables[Number(colIndex)];
          if (!variable || variable.kind !== "numeric") return undefined;
          if (this.editor.mode === "edit") return undefined; // editing preserves existing precision
          if (newValue === null || newValue === undefined || newValue === "") return undefined;
          const num = Number(String(newValue).replace(",", "."));
          if (Number.isNaN(num)) return undefined; // the numeric type already rejects letters
          if (variable.numericType === "integer") return Math.round(num);
          const decimals = Math.max(0, Math.min(10, Number(variable.decimals) || 0));
          const factor = Math.pow(10, decimals);
          return Math.round(num * factor) / factor;
        },
        worksheets: [
          {
            data,
            columns,
            // v5 order is [columns, rows]
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
          },
        ],
      });
      manualKeydownHandler = (event) => this._onSheetKeydown(event);
      manualClickHandler = (event) => this._onSheetClick(event);
      this.$refs.manualSheet.addEventListener("keydown", manualKeydownHandler, true);
      this.$refs.manualSheet.addEventListener("click", manualClickHandler, false);
    },

    async saveManual() {
      const isEdit = this.editor.mode === "edit";
      const name = (this.editor.name || "").trim();
      if (!isEdit && !name) { this.editor.error = "Escriba un nombre para el dataset."; return; }
      const columns = this.editor.variables.filter((v) => v.name.trim()).map((v) => ({ name: v.name.trim(), kind: v.kind }));
      if (!columns.length) { this.editor.error = "Defina al menos una variable con nombre."; return; }
      const names = columns.map((c) => c.name);
      if (new Set(names.map((n) => n.toLowerCase())).size !== names.length) {
        this.editor.error = "Los nombres de variable deben ser únicos."; return;
      }
      this._snapshotSheet();
      const rows = (manualSheetData ?? [])
        .map((obj) => Object.fromEntries(names.map((n) => [n, obj[n] ?? ""])))
        .filter((row) => names.some((n) => String(row[n] ?? "").trim() !== ""));
      this.editor.busy = true;
      this.editor.error = "";
      try {
        if (isEdit) {
          await api(`/api/datasets/${this.editor.datasetId}/data`, { method: "PUT", json: { columns, rows } });
          Alpine.store("toast").show(`Datos de «${this.editor.name}» guardados.`, "success");
        } else {
          const { dataset } = await api("/api/datasets/manual", { method: "POST", json: { name, columns, rows } });
          Alpine.store("datasets").select(dataset.id);
          Alpine.store("toast").show(`Dataset «${dataset.name}» creado.`, "success");
        }
        await Alpine.store("datasets").refresh();
        this.closeEditor();
        if (isEdit && this.viewer.dataset) await this.view(this.viewer.dataset, this.viewer.offset);
      } catch (error) {
        this.editor.error = error.message;
      } finally {
        this.editor.busy = false;
      }
    },
  }));
});
