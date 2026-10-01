// Saved results: browse, filter, reopen (tables + charts re-rendered from the stored report), edit, export, delete.

import { api, downloadUrl } from "../api.js";
import { renderReport } from "../report.js";

const MODULE_PAGES = {
  descriptive: "/descriptiva", distributions: "/distribuciones", hypothesis: "/hipotesis",
  regression: "/regresion", anova: "/anova", markets: "/mercados",
};
const PAGE_SIZE = 20;

document.addEventListener("alpine:init", () => {
  Alpine.data("resultadosPage", () => ({
    items: [],
    total: 0,
    modules: {},
    filters: { module: "", search: "" },
    loading: false,
    error: "",
    detail: null,
    editing: null,
    _timer: null,

    async init() {
      await this.load(true);
      const open = new URLSearchParams(location.search).get("open");
      if (open) await this.open(open);
    },

    async load(reset = true) {
      this.loading = true;
      this.error = "";
      try {
        const offset = reset ? 0 : this.items.length;
        const query = new URLSearchParams({ limit: PAGE_SIZE, offset });
        if (this.filters.module) query.set("module", this.filters.module);
        if (this.filters.search.trim()) query.set("search", this.filters.search.trim());
        const data = await api(`/api/results?${query}`);
        this.items = reset ? data.results : [...this.items, ...data.results];
        this.total = data.total;
        this.modules = data.modules;
      } catch (error) {
        this.error = error.message;
      } finally {
        this.loading = false;
      }
    },

    debouncedSearch() {
      clearTimeout(this._timer);
      this._timer = setTimeout(() => this.load(true), 300);
    },

    get hasMore() { return this.items.length < this.total; },
    moduleLabel(key) { return this.modules[key] ?? key; },
    pageFor(item) { return `${MODULE_PAGES[item.module] ?? "/"}?load=${item.id}`; },
    formatDate(iso) { return new Date(iso).toLocaleString("es", { dateStyle: "medium", timeStyle: "short" }); },

    async open(id) {
      this.error = "";
      try {
        const { result } = await api(`/api/results/${encodeURIComponent(id)}`);
        this.detail = result;
        await this.$nextTick();
        renderReport(this.$refs.out, result.report, result.figures);
        this.$refs.detailCard?.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        this.error = error.message;
      }
    },

    closeDetail() { this.detail = null; },
    exportJson(item) { downloadUrl(`/api/results/${item.id}/export`); },

    edit(item) { this.editing = { id: item.id, title: item.title, notes: item.notes ?? "", error: "" }; },
    async saveEdit() {
      const edit = this.editing;
      try {
        const { result } = await api(`/api/results/${edit.id}`, { method: "PATCH", json: { title: edit.title, notes: edit.notes } });
        const index = this.items.findIndex((item) => item.id === edit.id);
        if (index >= 0) this.items[index] = result;
        if (this.detail?.id === edit.id) Object.assign(this.detail, { title: result.title, notes: result.notes });
        this.editing = null;
        Alpine.store("toast").show("Cambios guardados.", "success");
      } catch (error) {
        edit.error = error.message;
      }
    },

    async remove(item) {
      if (!window.confirm(`¿Eliminar el resultado «${item.title}»? Esta acción no se puede deshacer.`)) return;
      try {
        await api(`/api/results/${item.id}`, { method: "DELETE" });
        this.items = this.items.filter((i) => i.id !== item.id);
        this.total -= 1;
        if (this.detail?.id === item.id) this.closeDetail();
        Alpine.store("toast").show("Resultado eliminado.", "success");
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
    },
  }));
});
