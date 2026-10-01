// Account administration (administrators only).

import { api } from "../api.js";

const emptyForm = () => ({ username: "", display_name: "", password: "", role: "user", can_use_console: false, can_use_ai: false });
const toast = (message, kind = "info") => Alpine.store("toast").show(message, kind);

document.addEventListener("alpine:init", () => {
  Alpine.data("usuariosPage", () => ({
    users: [],
    me: null,
    creating: false,
    form: emptyForm(),
    error: "",
    busy: false,

    // ------------------------------------------------------------------ AI settings (server-wide)
    ai: null, // {enabled, configured, key_source, key_hint, model, requests_per_hour, usage: [...]}
    aiForm: { key: "", model: "" },
    aiBusy: false,
    aiMessage: "",
    aiError: "",

    async init() {
      this.me = (await api("/api/auth/me")).user.id;
      await Promise.all([this.load(), this.loadAi()]);
    },

    async loadAi() {
      try {
        this.ai = await api("/api/admin/ai");
        this.aiForm.model = this.ai.model;
      } catch (error) {
        this.aiError = error.message;
      }
    },
    /** One call to the settings endpoint; shows the outcome and refreshes what the page knows. */
    async saveAi(changes, message) {
      this.aiBusy = true;
      this.aiError = "";
      this.aiMessage = "";
      try {
        this.ai = await api("/api/admin/ai", { method: "PUT", json: changes });
        this.aiForm = { key: "", model: this.ai.model };
        this.aiMessage = message;
        Alpine.store("ai").refresh();
      } catch (error) {
        this.aiError = error.message;
      } finally {
        this.aiBusy = false;
      }
    },
    saveKey() { return this.saveAi({ key: this.aiForm.key }, "Clave guardada."); },
    clearKey() {
      if (window.confirm("¿Quitar la clave de API guardada por Sigma? (No afecta a la variable de entorno ANTHROPIC_API_KEY.)")) {
        return this.saveAi({ clear_key: true }, "Clave eliminada.");
      }
    },
    saveModel() { return this.saveAi({ model: this.aiForm.model }, "Modelo guardado."); },
    toggleAi() { return this.saveAi({ enabled: !this.ai.enabled }, this.ai.enabled ? "IA desactivada." : "IA activada."); },
    async testAi() {
      this.aiBusy = true;
      this.aiError = "";
      this.aiMessage = "";
      try {
        const result = await api("/api/admin/ai/test", { method: "POST" });
        this.aiMessage = `Conexión correcta con ${result.model}${result.fallback_used ? " (respondió el modelo de reserva)" : ""}.`;
      } catch (error) {
        this.aiError = error.message;
      } finally {
        this.aiBusy = false;
      }
    },
    formatTokens(n) { return (n ?? 0).toLocaleString("es"); },

    async load() {
      this.users = (await api("/api/users")).users;
    },

    formatDate(iso) {
      return iso ? new Date(iso).toLocaleString("es", { dateStyle: "medium", timeStyle: "short" }) : "Nunca";
    },

    openCreate() {
      this.form = emptyForm();
      this.error = "";
      this.creating = true;
    },

    async create() {
      this.busy = true;
      this.error = "";
      try {
        await api("/api/users", { method: "POST", json: this.form });
        this.creating = false;
        await this.load();
        toast("Usuario creado.", "success");
      } catch (error) {
        this.error = error.message;
      } finally {
        this.busy = false;
      }
    },

    async update(user, changes) {
      try {
        const { user: saved } = await api(`/api/users/${user.id}`, { method: "PATCH", json: changes });
        Object.assign(user, saved);
      } catch (error) {
        toast(error.message, "error");
      }
      await this.load(); // also reverts a checkbox/select the server refused to change
    },

    async resetPassword(user) {
      const password = window.prompt(`Nueva contraseña para ${user.username} (mínimo 10 caracteres):`);
      if (!password) return;
      try {
        await api(`/api/users/${user.id}/reset-password`, { method: "POST", json: { new_password: password } });
        toast("Contraseña restablecida. Sus sesiones abiertas se cerraron.", "success");
      } catch (error) {
        toast(error.message, "error");
      }
    },

    async remove(user) {
      if (!window.confirm(`¿Eliminar a ${user.username} y TODOS sus datos (datasets, resultados, notebooks)? Esta acción no se puede deshacer.`)) return;
      try {
        await api(`/api/users/${user.id}`, { method: "DELETE" });
        await this.load();
        toast("Usuario eliminado.", "success");
      } catch (error) {
        toast(error.message, "error");
      }
    },
  }));
});
