import { api } from "../api.js";
import "../database-chooser.js";

document.addEventListener("alpine:init", () => {
  Alpine.data("setupPage", () => ({
    username: "",
    displayName: "",
    password: "",
    confirm: "",
    error: "",
    busy: false,

    async submit() {
      this.error = "";
      if (this.password !== this.confirm) {
        this.error = "Las contraseñas no coinciden.";
        return;
      }
      this.busy = true;
      try {
        await api("/api/auth/setup", {
          method: "POST",
          json: { username: this.username, display_name: this.displayName, password: this.password },
        });
        location.assign("/");
      } catch (error) {
        this.error = error.message;
        this.busy = false;
      }
    },
  }));
});
