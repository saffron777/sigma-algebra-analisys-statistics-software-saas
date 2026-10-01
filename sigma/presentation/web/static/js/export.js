// Exporting reports (PDF · Word · Excel): the store that drives the export dialog, and helpers that turn a result
// into the section the server needs. Charts are drawn here (light theme) and travel to the server as PNG files.

import { api, fetchFile, saveBlob } from "./api.js";
import { interpretationBlocks } from "./ai.js";
import { figureToPng, freezeFiguresForPrint } from "./plot.js";

const AUTHOR_KEY = "sigma-author";
const FORMAT_KEY = "sigma-export-format";
const FORMATS = [
  { key: "pdf", label: "PDF", hint: "Para entregar o imprimir" },
  { key: "docx", label: "Word", hint: "Para seguir editando" },
  { key: "xlsx", label: "Excel", hint: "Tablas como hojas de cálculo" },
];

const read = (key) => { try { return localStorage.getItem(key) ?? ""; } catch { return ""; } };
const write = (key, value) => { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } };

/** The section of a report for one analysis payload `{meta, report, figures}`. */
export function resultSection(payload, { datasetName, extraMeta = [] } = {}) {
  const meta = [...extraMeta];
  const name = datasetName ?? Alpine.store("datasets").list.find((d) => d.id === payload.meta?.dataset_id)?.name;
  if (name) meta.unshift(`Dataset: ${name}`);
  meta.push(`Fecha: ${new Date().toLocaleDateString("es")}`);
  return {
    title: payload.report.title, subtitle: payload.report.subtitle ?? null, meta,
    blocks: [...(payload.report.blocks ?? []), ...interpretationBlocks(payload.interpretation)],
    figures: payload.figures ?? [],
  };
}

const queryOf = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v !== null && v !== undefined));

export function defineExportStore() {
  // the section builder is a function: keep it off the reactive store (Alpine would wrap it in a proxy)
  let provider = null;

  Alpine.store("export", {
    open: false,
    formats: FORMATS,
    kind: "sections", // sections (a result or several) | dataset | notebook
    heading: "Exportar informe",
    format: read(FORMAT_KEY) || "pdf",
    title: "",
    subtitle: "",
    author: read(AUTHOR_KEY),
    templateId: "",
    cover: "", // "" = as the template says, "1" yes, "0" no
    toc: "",
    pageSize: "",
    orientation: "",
    templates: [],
    datasetId: "",
    notebookId: "",
    content: { text: true, code: true, outputs: true, numbers: false },
    busy: false,
    progress: "",
    error: "",

    init() {
      this.loadTemplates();
      Alpine.store("sync").on("templates", () => this.loadTemplates());
    },

    async loadTemplates() {
      try {
        this.templates = (await api("/api/templates")).templates;
        if (this.templateId && !this.templates.some((t) => t.id === this.templateId)) this.templateId = "";
      } catch { /* the dialog still works with the built-in look */ }
    },

    /**
     * Open the dialog.
     * @param {{title: string, subtitle?: string, sections?: () => Promise<object[]>|object[], dataset?: string, notebook?: string,
     *          heading?: string, formats?: string[]}} options
     */
    show({ title, subtitle = "", sections, dataset = "", notebook = "", heading = "Exportar informe" }) {
      provider = sections ?? null;
      this.kind = dataset ? "dataset" : notebook ? "notebook" : "sections";
      this.datasetId = dataset;
      this.notebookId = notebook;
      this.heading = heading;
      this.title = title;
      this.subtitle = subtitle;
      this.error = "";
      this.progress = "";
      this.open = true;
    },
    close() {
      if (!this.busy) this.open = false;
    },

    get hasTemplates() { return this.templates.length > 0; },
    get defaultTemplateName() { return this.templates.find((t) => t.is_default)?.name ?? ""; },

    options() {
      return {
        template: this.templateId, title: this.title.trim(), subtitle: this.subtitle.trim(), author: this.author.trim(),
        cover: this.cover, toc: this.toc, page_size: this.pageSize, orientation: this.orientation,
      };
    },

    async submit() {
      if (this.busy) return;
      this.busy = true;
      this.error = "";
      write(AUTHOR_KEY, this.author.trim());
      write(FORMAT_KEY, this.format);
      try {
        const { blob, filename } = await this.build();
        saveBlob(blob, filename);
        Alpine.store("toast").show(`Informe «${filename}» descargado.`, "success", 4500);
        this.open = false;
      } catch (error) {
        this.error = error.message;
      } finally {
        this.busy = false;
        this.progress = "";
      }
    },

    async build() {
      const options = this.options();
      if (this.kind === "dataset") {
        this.progress = "Generando la ficha…";
        return fetchFile(`/api/datasets/${this.datasetId}/report/${this.format}?${queryOf(options)}`);
      }
      if (this.kind === "notebook") {
        this.progress = "Generando el informe del notebook…";
        const content = { text: +this.content.text, code: +this.content.code, outputs: +this.content.outputs, numbers: +this.content.numbers };
        return fetchFile(`/api/notebooks/${this.notebookId}/report/${this.format}?${queryOf({ ...options, ...content })}`);
      }
      return this.buildFromSections(options);
    },

    async buildFromSections(options) {
      this.progress = "Reuniendo los resultados…";
      const sections = await provider();
      if (!sections.length) throw new Error("No hay nada que exportar.");
      const total = sections.reduce((n, s) => n + s.figures.length, 0);
      const form = new FormData();
      let done = 0;
      const payloadSections = [];
      for (const [position, section] of sections.entries()) {
        // a figure that fails to draw is simply left out: the report says «Gráfico no disponible»
        for (const [index, figure] of section.figures.entries()) {
          this.progress = total ? `Preparando gráficos ${++done}/${total}` : "";
          try {
            form.append(`img-${position}-${index}`, await figureToPng(figure), `grafico-${position}-${index}.png`);
          } catch { /* skipped */ }
        }
        payloadSections.push({ title: section.title, subtitle: section.subtitle, meta: section.meta, blocks: section.blocks });
      }
      this.progress = "Generando el archivo…";
      form.append("payload", JSON.stringify({
        title: options.title || null, subtitle: options.subtitle || null, author: options.author,
        template_id: options.template || null,
        cover: options.cover === "" ? null : options.cover === "1",
        toc: options.toc === "" ? null : options.toc === "1",
        page_size: options.page_size || null, orientation: options.orientation || null,
        sections: payloadSections,
      }));
      return fetchFile(`/api/export/${this.format}`, { method: "POST", form });
    },
  });
}

/** «Imprimir»: freeze the charts as white images, open the browser's print dialog, then restore the screen. */
export async function printPage() {
  const restore = await freezeFiguresForPrint();
  const finish = () => { window.removeEventListener("afterprint", finish); restore(); };
  window.addEventListener("afterprint", finish);
  window.print();
}
