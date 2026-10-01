import { api } from "../api.js";

document.addEventListener("alpine:init", () => {
  Alpine.data("homePage", () => ({
    recent: [],
    total: 0,
    modules: {},

    async init() {
      await this.refresh();
      Alpine.store("sync").on("results", () => this.refresh());
    },

    async refresh() {
      try {
        const data = await api("/api/results?limit=5");
        this.recent = data.results;
        this.total = data.total;
        this.modules = data.modules;
      } catch {
        /* the dashboard still works without the recent-results list */
      }
    },

    cleaning: false,
    /** «Limpiar todos los módulos»: every workspace (forms and results) is emptied; saved results are untouched. */
    async clearAll() {
      if (!window.confirm("¿Limpiar el espacio de trabajo de todos los módulos? Los resultados guardados se conservan.")) return;
      this.cleaning = true;
      try {
        await api("/api/workspace", { method: "DELETE" });
        Alpine.store("toast").show("Todos los módulos quedaron limpios.", "success");
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      } finally {
        this.cleaning = false;
      }
    },

    formatDate(iso) {
      return new Date(iso).toLocaleDateString("es", { day: "numeric", month: "short" });
    },
  }));
});
