// Database administration page (administrators only): what is in use and how to switch.

import { api } from "../api.js";
import "../database-chooser.js";

document.addEventListener("alpine:init", () => {
  Alpine.data("databaseOverview", () => ({
    status: null,
    loading: false,
    error: "",

    async init() {
      await this.load();
    },
    async load() {
      this.loading = true;
      this.error = "";
      try {
        this.status = await api("/api/admin/database");
      } catch (error) {
        this.error = error.message;
      } finally {
        this.loading = false;
      }
    },
  }));
});
