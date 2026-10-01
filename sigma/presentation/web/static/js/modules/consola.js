// Python console: notebook editor (code + markdown cells) backed by one Jupyter kernel per notebook.

import { api, downloadUrl } from "../api.js";
import { renderMarkdown, renderOutputs } from "../output.js";

const LAST_NOTEBOOK_KEY = "sigma-notebook";
const AUTOSAVE_MS = 1500;
const STATUS_POLL_MS = 4000;

const newId = () => Math.random().toString(16).slice(2, 10);
const newCell = (type = "code", source = "") => ({
  id: newId(), cell_type: type, source, metadata: {}, outputs: [], execution_count: null, running: false, editing: type === "markdown",
});

const prepareCells = (cells) =>
  cells.map((cell) => ({ ...cell, running: false, editing: false, outputs: cell.outputs ?? [], source: String(cell.source ?? "") }));

const forSaving = (cells) =>
  cells.map(({ running, editing, ...cell }) => (cell.cell_type === "code" ? cell : { ...cell, outputs: undefined, execution_count: undefined }));

const read = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
const write = (key, value) => { try { localStorage.setItem(key, value); } catch { /* ignore */ } };

document.addEventListener("alpine:init", () => {
  Alpine.data("consolaPage", () => ({
    notebooks: [],
    current: null,
    error: "",
    busy: false,
    kernel: { running: false, busy: false, starting: false },
    dirty: false,
    saving: false,
    savedAt: "",
    runningAll: false,
    helpOpen: false,
    saveTimer: null,
    pollTimer: null,

    // ------------------------------------------------------------------ lifecycle
    async init() {
      window.addEventListener("beforeunload", (event) => {
        if (this.dirty) { event.preventDefault(); event.returnValue = ""; }
      });
      await this.refreshList();
      const remembered = read(LAST_NOTEBOOK_KEY);
      const target = this.notebooks.find((n) => n.id === remembered) ?? this.notebooks[0];
      if (target) await this.open(target.id);
      this.pollTimer = setInterval(() => this.pollKernel(), STATUS_POLL_MS);
    },

    async refreshList() {
      try {
        this.notebooks = (await api("/api/notebooks")).notebooks;
      } catch (error) {
        this.error = error.message;
      }
    },

    async open(id) {
      if (this.dirty) await this.save();
      this.error = "";
      try {
        const { notebook } = await api(`/api/notebooks/${id}`);
        this.current = { id: notebook.id, name: notebook.name, metadata: notebook.metadata, cells: prepareCells(notebook.cells) };
        this.dirty = false;
        this.savedAt = "";
        write(LAST_NOTEBOOK_KEY, id);
        await this.$nextTick();
        this.startKernel();
      } catch (error) {
        this.error = error.message;
      }
    },

    // ------------------------------------------------------------------ notebook management
    async createNotebook() {
      const name = window.prompt("Nombre del notebook:", "Análisis nuevo");
      if (name === null) return;
      await this.guard(async () => {
        const { notebook } = await api("/api/notebooks", { method: "POST", json: { name } });
        await this.refreshList();
        await this.open(notebook.id);
      });
    },

    async importNotebook(event) {
      const file = event.target.files[0];
      event.target.value = "";
      if (!file) return;
      await this.guard(async () => {
        const form = new FormData();
        form.append("file", file);
        const { notebook } = await api("/api/notebooks/import", { method: "POST", form });
        await this.refreshList();
        await this.open(notebook.id);
        Alpine.store("toast").show(`Notebook «${notebook.name}» importado (${notebook.cells.length} celdas).`, "success");
      });
    },

    async renameNotebook() {
      const name = window.prompt("Nuevo nombre:", this.current.name);
      if (!name || name.trim() === this.current.name) return;
      await this.guard(async () => {
        const { notebook } = await api(`/api/notebooks/${this.current.id}`, { method: "PATCH", json: { name } });
        this.current.name = notebook.name;
        await this.refreshList();
      });
    },

    async deleteNotebook() {
      if (!window.confirm(`¿Eliminar el notebook «${this.current.name}»? Esta acción no se puede deshacer.`)) return;
      await this.guard(async () => {
        await api(`/api/notebooks/${this.current.id}`, { method: "DELETE" });
        this.current = null;
        this.dirty = false;
        await this.refreshList();
        if (this.notebooks[0]) await this.open(this.notebooks[0].id);
      });
    },

    async exportNotebook() {
      if (this.dirty) await this.save();
      downloadUrl(`/api/notebooks/${this.current.id}/export`);
    },

    async guard(action) {
      this.busy = true;
      this.error = "";
      try { await action(); } catch (error) { this.error = error.message; } finally { this.busy = false; }
    },

    // ------------------------------------------------------------------ saving
    touch() {
      this.dirty = true;
      clearTimeout(this.saveTimer);
      this.saveTimer = setTimeout(() => this.save(), AUTOSAVE_MS);
    },

    async save() {
      if (!this.current || !this.dirty || this.saving) return;
      clearTimeout(this.saveTimer);
      this.saving = true;
      try {
        await api(`/api/notebooks/${this.current.id}`, {
          method: "PUT", json: { cells: forSaving(this.current.cells), metadata: this.current.metadata },
        });
        this.dirty = false;
        this.savedAt = new Date().toLocaleTimeString("es", { hour: "2-digit", minute: "2-digit" });
      } catch (error) {
        this.error = `No se pudo guardar el notebook: ${error.message}`;
      } finally {
        this.saving = false;
      }
    },

    // ------------------------------------------------------------------ kernel
    async startKernel() {
      if (!this.current) return;
      this.kernel.starting = true;
      try {
        this.kernel = { ...(await api(`/api/notebooks/${this.current.id}/kernel/start`, { method: "POST" })).kernel, starting: false };
      } catch (error) {
        this.kernel = { running: false, busy: false, starting: false };
        this.error = error.message;
      }
    },

    async pollKernel() {
      if (!this.current || this.kernel.starting || document.hidden) return;
      try {
        const { kernel } = await api(`/api/notebooks/${this.current.id}/kernel`);
        this.kernel = { ...kernel, starting: false };
      } catch { /* transient */ }
    },

    async interrupt() {
      try { await api(`/api/notebooks/${this.current.id}/kernel/interrupt`, { method: "POST" }); } catch (error) { this.error = error.message; }
    },

    async restartKernel() {
      if (!window.confirm("¿Reiniciar el kernel? Se perderán las variables en memoria (el código y las salidas se conservan).")) return;
      this.kernel.starting = true;
      this.runningAll = false;
      await this.guard(async () => {
        await api(`/api/notebooks/${this.current.id}/kernel/restart`, { method: "POST" });
        await this.startKernel();
        Alpine.store("toast").show("Kernel reiniciado.", "success");
      });
      this.kernel.starting = false;
    },

    get kernelLabel() {
      if (this.kernel.starting) return "Iniciando kernel…";
      if (!this.kernel.running) return "Kernel detenido";
      return this.kernel.busy ? "Kernel ocupado" : "Kernel listo";
    },
    get anyRunning() { return !!this.current?.cells.some((c) => c.running); },

    // ------------------------------------------------------------------ cells
    index(cell) { return this.current.cells.findIndex((c) => c.id === cell.id); },

    addCell(type = "code", after = null) {
      const cell = newCell(type);
      const cells = this.current.cells;
      cells.splice(after === null ? cells.length : this.index(after) + 1, 0, cell);
      this.touch();
      this.$nextTick(() => document.getElementById(`cell-${cell.id}`)?.focus());
      return cell;
    },
    deleteCell(cell) {
      this.current.cells.splice(this.index(cell), 1);
      if (!this.current.cells.length) this.addCell("code");
      this.touch();
    },
    moveCell(cell, delta) {
      const cells = this.current.cells;
      const from = this.index(cell);
      const to = from + delta;
      if (to < 0 || to >= cells.length) return;
      [cells[from], cells[to]] = [cells[to], cells[from]];
      this.touch();
    },
    toggleType(cell) {
      cell.cell_type = cell.cell_type === "code" ? "markdown" : "code";
      cell.outputs = [];
      cell.execution_count = null;
      cell.editing = cell.cell_type === "markdown";
      this.touch();
    },
    clearOutputs() {
      this.current.cells.forEach((cell) => { cell.outputs = []; cell.execution_count = null; });
      this.touch();
    },

    async runCell(cell, advance = false) {
      if (cell.cell_type === "markdown") {
        cell.editing = false;
      } else if (!cell.running) {
        await this.execute(cell);
      }
      if (advance) {
        const next = this.current.cells[this.index(cell) + 1];
        if (next) document.getElementById(`cell-${next.id}`)?.focus();
        else this.addCell("code");
      }
    },

    /** Runs a code cell on the server; returns false when it failed (so "run all" can stop). */
    async execute(cell) {
      cell.running = true;
      cell.outputs = [];
      try {
        const result = await api(`/api/notebooks/${this.current.id}/execute`, { method: "POST", json: { source: cell.source } });
        cell.outputs = result.outputs;
        cell.execution_count = result.execution_count ?? cell.execution_count;
        this.kernel.running = true;
        if (result.kernel_restarted) Alpine.store("toast").show("El kernel se reinició; las variables anteriores ya no existen.", "info", 6000);
        return result.status === "ok";
      } catch (error) {
        cell.outputs = [{ output_type: "error", ename: "Error", evalue: error.message, traceback: [error.message] }];
        return false;
      } finally {
        cell.running = false;
        this.touch();
      }
    },

    async runAll() {
      this.runningAll = true;
      for (const cell of [...this.current.cells]) {
        if (!this.runningAll) break;
        if (cell.cell_type === "code" && cell.source.trim()) {
          if (!(await this.execute(cell))) break;
        } else if (cell.cell_type === "markdown") {
          cell.editing = false;
        }
      }
      this.runningAll = false;
    },

    // ------------------------------------------------------------------ editor helpers
    autosize(textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.max(textarea.scrollHeight, 44)}px`;
    },
    indent(textarea) {
      const { selectionStart: start, selectionEnd: end, value } = textarea;
      textarea.value = `${value.slice(0, start)}    ${value.slice(end)}`;
      textarea.selectionStart = textarea.selectionEnd = start + 4;
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    },
    drawOutputs(element, cell) { renderOutputs(element, cell.outputs); },
    drawMarkdown(element, cell) {
      if (cell.editing) return;
      if (cell.source.trim()) renderMarkdown(element, cell.source);
      else element.replaceChildren(Object.assign(document.createElement("p"), { className: "text-sm text-slate-400", textContent: "Celda de texto vacía (haga doble clic para editar)" }));
    },
  }));
});
