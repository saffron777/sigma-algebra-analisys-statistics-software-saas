// Printing a notebook with page control: page size and orientation, margins, what to include, page breaks before
// headings / before marked cells, short cells kept in one piece, and a header and footer with «Página X de Y».
//
// The notebook is laid out in a hidden print-only element and the browser's print dialog does the rest. Page numbers
// use CSS page margin boxes (Chrome / Edge 131+); the server's PDF export is the way to get them everywhere.

import { renderMarkdown, renderOutputs } from "./output.js";

export const PRINT_DEFAULTS = {
  pageSize: "A4", orientation: "portrait", margin: 18,
  text: true, code: true, outputs: true, numbers: false,
  breakH1: false, breakH2: false, keepCells: true,
  header: true, footer: true, footerText: "", templateId: "",
};

const ROOT_ID = "nb-print-root";
const STYLE_ID = "nb-print-style";
const SHORT_CELL_LINES = 30;

const cssString = (text) => `"${String(text ?? "").replace(/[\\"]/g, "\\$&").replace(/[\r\n]+/g, " ")}"`;

function pageCss(options, title, organization) {
  const box = "font: 8pt 'Segoe UI', Arial, sans-serif; color: #64748b;";
  const rules = [`size: ${options.pageSize} ${options.orientation}`, `margin: ${Number(options.margin) || 18}mm`];
  if (options.header) {
    rules.push(`@top-left { content: ${cssString(title)}; ${box} }`);
    if (organization) rules.push(`@top-right { content: ${cssString(organization)}; ${box} }`);
  }
  if (options.footer) {
    if (options.footerText) rules.push(`@bottom-left { content: ${cssString(options.footerText)}; ${box} }`);
    rules.push(`@bottom-right { content: "Página " counter(page) " de " counter(pages); ${box} }`);
  }
  return `
@page { ${rules.join("; ")}; }
@media print {
  html, body { background: #fff !important; color: #0f172a !important; }
  body > *:not(#${ROOT_ID}) { display: none !important; }
  #${ROOT_ID} { display: block !important; }
}`;
}

const lines = (text) => String(text ?? "").split("\n").length;

function buildRoot(notebook, options, template) {
  const root = document.createElement("div");
  root.id = ROOT_ID;
  root.className = "nb-print";

  const head = document.createElement("header");
  head.className = "nb-print-head";
  if (template?.has_logo) {
    const logo = document.createElement("img");
    logo.src = `/api/templates/${template.id}/logo`;
    logo.alt = "";
    logo.className = "nb-print-logo";
    head.appendChild(logo);
  }
  const title = document.createElement("h1");
  title.className = "nb-print-title";
  title.textContent = notebook.name;
  head.appendChild(title);
  const meta = document.createElement("p");
  meta.className = "nb-print-meta";
  meta.textContent = [template?.organization, new Date().toLocaleDateString("es")].filter(Boolean).join(" · ");
  head.appendChild(meta);
  root.appendChild(head);

  for (const cell of notebook.cells) {
    const source = Array.isArray(cell.source) ? cell.source.join("") : String(cell.source ?? "");
    const section = document.createElement("section");
    section.className = "nb-cell";
    if (cell.metadata?.sigma?.page_break_before) section.classList.add("nb-break");

    if (cell.cell_type === "markdown") {
      if (!options.text || !source.trim()) continue;
      const body = document.createElement("div");
      body.className = "nb-md";
      renderMarkdown(body, source);
      if (options.breakH1) body.querySelectorAll("h1").forEach((h) => h.classList.add("nb-break"));
      if (options.breakH2) body.querySelectorAll("h2").forEach((h) => h.classList.add("nb-break"));
      section.appendChild(body);
    } else if (cell.cell_type === "code") {
      const hasOutputs = options.outputs && (cell.outputs ?? []).length > 0;
      if (!(options.code && source.trim()) && !hasOutputs) continue;
      if (options.code && source.trim()) {
        if (options.numbers) {
          const label = document.createElement("div");
          label.className = "nb-label";
          label.textContent = `[${cell.execution_count ?? " "}]`;
          section.appendChild(label);
        }
        const code = document.createElement("pre");
        code.className = "nb-code";
        code.textContent = source;
        section.appendChild(code);
      }
      if (hasOutputs) {
        const out = document.createElement("div");
        out.className = "nb-out";
        renderOutputs(out, cell.outputs);
        section.appendChild(out);
      }
      const short = lines(source) <= SHORT_CELL_LINES && (cell.outputs ?? []).reduce((n, o) => n + lines(o.text ?? ""), 0) <= SHORT_CELL_LINES;
      if (options.keepCells && short) section.classList.add("nb-keep");
    } else {
      continue;
    }
    root.appendChild(section);
  }
  return root;
}

/**
 * Lay the notebook out for printing and open the print dialog.
 * @param {{name: string, cells: object[]}} notebook
 * @param {typeof PRINT_DEFAULTS} options
 * @param {object|null} template  the report template whose logo and organization head the printout
 */
export function printNotebook(notebook, options, template = null) {
  cleanup();
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = pageCss(options, notebook.name, template?.organization ?? "");
  document.head.appendChild(style);
  document.body.appendChild(buildRoot(notebook, options, template));
  window.addEventListener("afterprint", cleanup, { once: true });
  waitForImages(document.getElementById(ROOT_ID)).then(() => window.print());
}

/** Resolve when every image of `root` has loaded (or failed), or after `timeout` ms: a stuck image must not block printing. */
function waitForImages(root, timeout = 2500) {
  const pending = [...root.querySelectorAll("img")].filter((image) => !image.complete);
  if (!pending.length) return Promise.resolve();
  const settled = pending.map((image) => new Promise((resolve) => {
    image.addEventListener("load", resolve, { once: true });
    image.addEventListener("error", resolve, { once: true });
  }));
  return Promise.race([Promise.all(settled), new Promise((resolve) => setTimeout(resolve, timeout))]);
}

function cleanup() {
  document.getElementById(ROOT_ID)?.remove();
  document.getElementById(STYLE_ID)?.remove();
}
