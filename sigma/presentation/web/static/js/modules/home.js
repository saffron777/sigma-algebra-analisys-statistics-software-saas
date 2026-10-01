import { api } from "../api.js";

document.addEventListener("alpine:init", () => {
  Alpine.data("homePage", () => ({
    recent: [],
    total: 0,
    modules: {},

    async init() {
      try {
        const data = await api("/api/results?limit=5");
        this.recent = data.results;
        this.total = data.total;
        this.modules = data.modules;
      } catch {
        /* the dashboard still works without the recent-results list */
      }
    },

    formatDate(iso) {
      return new Date(iso).toLocaleDateString("es", { day: "numeric", month: "short" });
    },
  }));
});
