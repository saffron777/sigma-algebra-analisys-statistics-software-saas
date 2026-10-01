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

    async init() {
      this.me = (await api("/api/auth/me")).user.id;
      await this.load();
    },

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
