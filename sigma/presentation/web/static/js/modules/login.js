import { api } from "../api.js";

document.addEventListener("alpine:init", () => {
  Alpine.data("loginPage", () => ({
    username: "",
    password: "",
    error: "",
    busy: false,

    async submit() {
      this.busy = true;
      this.error = "";
      try {
        await api("/api/auth/login", { method: "POST", json: { username: this.username, password: this.password } });
        location.assign(this.$root.dataset.next || "/");
      } catch (error) {
        this.error = error.message;
        this.password = "";
        this.busy = false;
      }
    },
  }));
});
