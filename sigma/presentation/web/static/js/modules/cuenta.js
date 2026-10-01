import { api } from "../api.js";

document.addEventListener("alpine:init", () => {
  Alpine.data("cuentaPage", () => ({
    displayName: document.body.dataset.displayName ?? "",
    current: "",
    next: "",
    repeat: "",
    error: "",
    busyProfile: false,
    busyPassword: false,

    async init() {
      try {
        this.displayName = (await api("/api/auth/me")).user.display_name;
      } catch { /* the page still works without the prefilled name */ }
    },

    async saveProfile() {
      this.busyProfile = true;
      try {
        const { user } = await api("/api/auth/profile", { method: "PATCH", json: { display_name: this.displayName } });
        this.displayName = user.display_name;
        Alpine.store("toast").show("Nombre actualizado.", "success");
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      } finally {
        this.busyProfile = false;
      }
    },

    async changePassword() {
      this.error = "";
      if (this.next !== this.repeat) {
        this.error = "Las contraseñas nuevas no coinciden.";
        return;
      }
      this.busyPassword = true;
      try {
        await api("/api/auth/change-password", { method: "POST", json: { current_password: this.current, new_password: this.next } });
        this.current = this.next = this.repeat = "";
        Alpine.store("toast").show("Contraseña cambiada. Las demás sesiones se cerraron.", "success");
      } catch (error) {
        this.error = error.message;
      } finally {
        this.busyPassword = false;
      }
    },
  }));
});
