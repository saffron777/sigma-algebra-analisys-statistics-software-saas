// Account administration (administrators only).

import { api } from "../api.js";

const emptyForm = () => ({ username: "", display_name: "", password: "", role: "user", can_use_console: false, can_use_ai: false });
const toast = (message, kind = "info") => Alpine.store("toast").show(message, kind);

const emptyAiForm = () => ({
  provider: "",
  label: "",
  model: "",
  base_url: "",
  key: "",
  note: "",
  key_hint: "",
  key_required: true,
  suggested_models: [],
  suggestions: [],
  custom_model: false, // «Otro modelo…»: a free-text field instead of the provider's list
  default_base_url: "",
  path: "",
  placeholder: "",
});

document.addEventListener("alpine:init", () => {
  Alpine.data("usuariosPage", () => ({
    users: [],
    me: null,
    creating: false,
    form: emptyForm(),
    error: "",
    busy: false,

    // ------------------------------------------------------------------ AI models (server-wide)
    ai: null, // {enabled, configured, models:[…], providers:[…], default_server_id, requests_per_hour, usage:[…]}
    aiForm: emptyAiForm(),
    keyDraft: {}, // {model id: new key} while it is being typed
    aiBusy: false,
    aiMessage: "",
    aiError: "",
    formError: "",

    async init() {
      this.me = (await api("/api/auth/me")).user.id;
      await Promise.all([this.load(), this.loadAi()]);
    },

    async loadAi() {
      try {
        this.ai = await api("/api/admin/ai");
        this.keyDraft = {};
        const current = (this.ai.providers ?? []).find((p) => p.key === this.aiForm.provider);
        const provider = current ?? (this.ai.providers ?? [])[0];
        if (provider) this.applyProvider(provider);
      } catch (error) {
        this.aiError = error.message;
      }
    },


    /** Prefill the fields a provider already knows (address, its models, whether it needs a key). */
    applyProvider(provider) {
      this.aiForm.provider = provider.key;
      this.aiForm.note = provider.note ?? "";
      this.aiForm.key_hint = provider.key_hint ?? "";
      this.aiForm.key_required = provider.key_required;
      this.aiForm.suggested_models = provider.suggested_models ?? [];
      this.aiForm.suggestions = provider.suggested_models ?? [];
      this.aiForm.default_base_url = provider.default_base_url ?? "";
      this.aiForm.path = provider.path ?? "";
      this.aiForm.placeholder = provider.placeholder || "";
      this.aiForm.base_url = provider.default_base_url ?? "";
      // with a list, the first model comes selected; without one (local services), a text field is shown
      this.aiForm.custom_model = this.aiForm.suggestions.length === 0;
      this.aiForm.model = this.aiForm.suggestions[0] ?? "";
    },

    providerChanged() {
      const provider = (this.ai?.providers ?? []).find((p) => p.key === this.aiForm.provider);
      if (provider) this.applyProvider(provider);
    },

    providerLabel() {
      return (this.ai?.providers ?? []).find((p) => p.key === this.aiForm.provider)?.label ?? "";
    },

    /** The models the list currently offers (always an array: the template reads it before the first load too). */
    suggestions() {
      return this.aiForm.suggestions ?? [];
    },

    /** The «Otro modelo…» entry of the list swaps the select for a text field (and «Lista» comes back). */
    pickModel() {
      if (this.aiForm.model === "__other__") this.customModel(true);
    },

    customModel(enabled) {
      this.aiForm.custom_model = enabled;
      if (enabled && this.aiForm.model === "__other__") this.aiForm.model = "";
      if (!enabled) this.aiForm.model = this.aiForm.suggestions[0] ?? "";
    },

    /** One call to a model endpoint; shows the outcome and refreshes what the page knows. */
    async saveAi(url, method, changes, message) {
      this.aiBusy = true;
      this.aiError = "";
      this.formError = "";
      this.aiMessage = "";
      try {
        const updated = await api(url, { method, json: changes });
        this.ai = updated;
        this.keyDraft = {};
        this.aiMessage = message;
        Alpine.store("ai").refresh();
        return true;
      } catch (error) {
        this.aiError = error.message;
        this.formError = error.message;
        return false;
      } finally {
        this.aiBusy = false;
      }
    },

    addServer() {
      const { provider, model, label, base_url, key } = this.aiForm;
      return this.saveAi(
        "/api/admin/ai/servers",
        "POST",
        { provider, model: model.trim(), label, base_url: base_url.trim(), key },
        "Modelo añadido.",
      ).then((ok) => {
        if (ok) {
          const chosen = this.aiForm.provider;
          const provider = (this.ai?.providers ?? []).find((p) => p.key === chosen);
          this.aiForm = { ...emptyAiForm(), provider: chosen };
          if (provider) this.applyProvider(provider);
        }
        return ok;
      });
    },
    updateServer(model, changes) { return this.saveAi(`/api/admin/ai/servers/${model.id}`, "PATCH", changes, "Modelo actualizado."); },
    saveKey(model) { return this.updateServer(model, { key: this.keyDraft[model.id] }); },
    clearKey(model) {
      if (window.confirm(`¿Quitar la clave de API de «${model.label}»? El modelo quedará sin usar hasta que ponga otra.`)) {
        return this.updateServer(model, { clear_key: true });
      }
    },
    makeDefault(model) { return this.updateServer(model, { make_default: true }); },
    removeServer(model) {
      if (window.confirm(`¿Eliminar el modelo «${model.label}»?`)) {
        return this.saveAi(`/api/admin/ai/servers/${model.id}`, "DELETE", undefined, "Modelo eliminado.");
      }
    },
    async testServer(model) {
      this.aiBusy = true;
      this.aiError = "";
      this.aiMessage = "";
      try {
        const result = await api(`/api/admin/ai/servers/${model.id}/test`, { method: "POST", json: {} });
        this.aiMessage = `Conexión correcta con ${result.model} (${result.server_label})${result.fallback_used ? ": respondió el modelo de reserva" : ""}.`;
      } catch (error) {
        this.aiError = error.message;
      } finally {
        this.aiBusy = false;
      }
    },
    toggleAi() { return this.saveAi("/api/admin/ai", "PUT", { enabled: !this.ai.enabled }, this.ai.enabled ? "IA desactivada." : "IA activada."); },
    async testAi() {
      this.aiBusy = true;
      this.aiError = "";
      this.aiMessage = "";
      try {
        const result = await api("/api/admin/ai/test", { method: "POST", json: {} });
        this.aiMessage = `Conexión correcta con ${result.model} (${result.server_label})${result.fallback_used ? ": respondió el modelo de reserva" : ""}.`;
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
