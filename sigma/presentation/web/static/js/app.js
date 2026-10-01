// Global Alpine registrations shared by every page: shell (layout), stores (datasets, save dialog, toasts).
// Loaded as a module *before* Alpine starts, so the `alpine:init` listener is always in time.

import { api } from "./api.js";
import { rethemeAll } from "./plot.js";

const ACTIVE_DATASET_KEY = "sigma-active-dataset";
const THEME_KEY = "sigma-theme";

const storage = {
  get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* storage unavailable: ignore */ }
  },
};

document.addEventListener("alpine:init", () => {
  // ------------------------------------------------------------------ layout shell
  Alpine.data("shell", () => ({
    menuOpen: false,
    dark: document.documentElement.classList.contains("dark"),
    toggleTheme() {
      this.dark = !this.dark;
      document.documentElement.classList.toggle("dark", this.dark);
      storage.set(THEME_KEY, this.dark ? "dark" : "light");
      rethemeAll();
    },
  }));

  // ------------------------------------------------------------------ toasts
  Alpine.store("toast", {
    items: [],
    show(message, kind = "info", milliseconds = 3600) {
      const id = Date.now() + Math.random();
      this.items.push({ id, message, kind });
      setTimeout(() => { this.items = this.items.filter((item) => item.id !== id); }, milliseconds);
    },
  });

  // ------------------------------------------------------------------ datasets
  Alpine.store("datasets", {
    list: [],
    loaded: false,
    activeId: storage.get(ACTIVE_DATASET_KEY) ?? "",

    async init() {
      await this.refresh();
    },
    async refresh() {
      try {
        const { datasets } = await api("/api/datasets");
        this.list = datasets;
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
      this.loaded = true;
      if (!this.list.some((dataset) => dataset.id === this.activeId)) this.activeId = this.list[0]?.id ?? "";
      this.persist();
    },
    select(id) {
      this.activeId = id;
      this.persist();
    },
    persist() {
      storage.set(ACTIVE_DATASET_KEY, this.activeId);
    },
    get active() {
      return this.list.find((dataset) => dataset.id === this.activeId) ?? null;
    },
    get columns() {
      return this.active?.columns ?? [];
    },
    columnsOfKind(kinds = []) {
      return kinds.length ? this.columns.filter((column) => kinds.includes(column.kind)) : this.columns;
    },
  });

  // ------------------------------------------------------------------ "save result" dialog
  Alpine.store("save", {
    open: false,
    title: "",
    notes: "",
    saving: false,
    error: "",
    payload: null,
    onSaved: null,

    show(payload, onSaved) {
      this.payload = payload;
      this.onSaved = onSaved ?? null;
      this.title = payload.meta.title;
      this.notes = "";
      this.error = "";
      this.open = true;
      setTimeout(() => document.getElementById("save-title-input")?.select(), 50);
    },
    close() {
      this.open = false;
    },
    async submit() {
      this.saving = true;
      this.error = "";
      try {
        const { meta, result, report, figures } = this.payload;
        const { result: saved } = await api("/api/results", {
          method: "POST",
          json: { ...meta, title: this.title, notes: this.notes, result, report, figures },
        });
        this.open = false;
        Alpine.store("toast").show("Resultado guardado en la base de datos local.", "success");
        this.onSaved?.(saved.id);
      } catch (error) {
        this.error = error.message;
      } finally {
        this.saving = false;
      }
    },
  });
});
