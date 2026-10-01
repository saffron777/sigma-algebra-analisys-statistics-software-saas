// Report templates: list, edit (colours, texts, page set-up, logo) and choose the default.

import { api } from "../api.js";

const emptyForm = () => ({
  name: "", organization: "", primary_color: "#4f46e5", secondary_color: "#0f172a", footer_text: "", cover_text: "",
  page_size: "A4", orientation: "portrait", show_cover: true, show_toc: true, make_default: false,
});

document.addEventListener("alpine:init", () => {
  Alpine.data("plantillasPage", () => ({
    templates: [],
    loaded: false,
    editing: null, // the template being edited ({id}) or {id: null} for a new one
    form: emptyForm(),
    error: "",
    saving: false,
    pendingLogo: null, // a logo chosen before the template exists: uploaded right after the first save
    logoPreview: "",
    logoVersion: 0,

    async init() {
      await this.load();
      Alpine.store("sync").on("templates", () => this.load());
    },

    async load() {
      try {
        this.templates = (await api("/api/templates")).templates;
      } catch (error) {
        this.error = error.message;
      }
      this.loaded = true;
    },

    logoUrl(template) {
      return `/api/templates/${template.id}/logo?v=${encodeURIComponent(template.updated_at)}`;
    },

    newTemplate() {
      this.editing = { id: null };
      this.form = emptyForm();
      this.pendingLogo = null;
      this.logoPreview = "";
      this.error = "";
      if (this.$refs.logo) this.$refs.logo.value = "";
    },

    edit(template) {
      this.editing = { id: template.id };
      const { name, organization, primary_color, secondary_color, footer_text, cover_text, page_size, orientation, show_cover, show_toc } = template;
      this.form = { name, organization, primary_color, secondary_color, footer_text, cover_text, page_size, orientation, show_cover, show_toc, make_default: false };
      this.pendingLogo = null;
      this.logoPreview = template.has_logo ? this.logoUrl(template) : "";
      this.error = "";
      if (this.$refs.logo) this.$refs.logo.value = "";
    },

    /** Pick a logo: shown at once; uploaded now for a saved template, or after the first save for a new one. */
    async chooseLogo(event) {
      const file = event.target.files[0];
      if (!file) return;
      this.error = "";
      if (this.editing?.id) {
        await this.uploadLogo(this.editing.id, file);
      } else {
        this.pendingLogo = file;
        this.logoPreview = URL.createObjectURL(file);
      }
    },

    async uploadLogo(id, file) {
      const form = new FormData();
      form.append("file", file);
      try {
        const { template } = await api(`/api/templates/${id}/logo`, { method: "POST", form });
        this.logoPreview = this.logoUrl(template);
        await this.load();
      } catch (error) {
        this.error = error.message;
        this.logoPreview = "";
      } finally {
        if (this.$refs.logo) this.$refs.logo.value = "";
      }
    },

    async removeLogo() {
      if (!this.editing?.id) {
        this.pendingLogo = null;
        this.logoPreview = "";
        if (this.$refs.logo) this.$refs.logo.value = "";
        return;
      }
      try {
        await api(`/api/templates/${this.editing.id}/logo`, { method: "DELETE" });
        this.logoPreview = "";
        await this.load();
      } catch (error) {
        this.error = error.message;
      }
    },

    async save() {
      this.saving = true;
      this.error = "";
      try {
        const isNew = !this.editing.id;
        const { template } = await api(isNew ? "/api/templates" : `/api/templates/${this.editing.id}`, {
          method: isNew ? "POST" : "PUT", json: this.form,
        });
        if (isNew && this.pendingLogo) {
          await this.uploadLogo(template.id, this.pendingLogo);
          this.pendingLogo = null;
        }
        await this.load();
        this.editing = { id: template.id };
        this.form.make_default = false;
        Alpine.store("toast").show(`Plantilla «${template.name}» guardada.`, "success");
      } catch (error) {
        this.error = error.message;
      } finally {
        this.saving = false;
      }
    },

    async makeDefault(template) {
      try {
        await api(`/api/templates/${template.id}/default`, { method: "POST", json: { default: true } });
        await this.load();
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
    },

    async remove() {
      const id = this.editing?.id;
      if (!id || !window.confirm(`¿Eliminar la plantilla «${this.form.name}»? Los informes ya generados no cambian.`)) return;
      try {
        await api(`/api/templates/${id}`, { method: "DELETE" });
        this.editing = null;
        await this.load();
        Alpine.store("toast").show("Plantilla eliminada.", "success");
      } catch (error) {
        this.error = error.message;
      }
    },
  }));
});
