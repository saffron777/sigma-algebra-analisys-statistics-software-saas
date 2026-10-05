// Asistente guiado: recomienda el análisis según la pregunta y los tipos de variable.

import { api } from "../api.js";

document.addEventListener("alpine:init", () => {
  Alpine.data("assistantPage", () => ({
    goals: [],
    kinds: [],
    goal: "",
    primary: "",
    secondary: "",
    recommendation: null,
    loading: false,

    async init() {
      try {
        const data = await api("/api/assistant/catalog");
        this.goals = data.goals;
        this.kinds = data.kinds;
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
    },

    async recommend() {
      if (!this.goal || !this.primary) {
        this.recommendation = null;
        return;
      }
      this.loading = true;
      try {
        const query = new URLSearchParams({ goal: this.goal, primary: this.primary });
        if (this.secondary) query.set("secondary", this.secondary);
        this.recommendation = await api(`/api/assistant/recommend?${query}`);
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      } finally {
        this.loading = false;
      }
    },
  }));
});
