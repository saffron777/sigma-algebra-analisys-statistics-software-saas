// Renders the generic report blocks produced by the server (heading, tiles, table, text, alert, figure).
// Everything goes through textContent: results and imported data are never interpreted as HTML.

import { purgeFigures, renderFigure } from "./plot.js";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

const TONES = { strong: "tone-strong", good: "tone-good", bad: "tone-bad", muted: "tone-muted" };

function fillCell(td, value) {
  if (value !== null && typeof value === "object") {
    td.textContent = value.text ?? "";
    if (TONES[value.tone]) td.className = TONES[value.tone];
  } else {
    td.textContent = value === null || value === undefined ? "—" : String(value);
  }
}

// Numbers, intervals ("[1, 2]"), percentages and dashes are right-aligned; everything else stays left.
const NUMERIC_TEXT = /^[−+\-<>±≈ ]*[\d.,]+(e[+-]?\d+)?( ?%)?$|^[—∞−]+$|^\[.+\]$/;

export function buildTable(columns, rows) {
  const wrap = el("div", "table-wrap");
  const table = el("table", "tbl");
  const head = el("thead");
  const headRow = el("tr");
  columns.forEach((name) => headRow.appendChild(el("th", null, name)));
  head.appendChild(headRow);
  const body = el("tbody");
  rows.forEach((row) => {
    const tr = el("tr");
    row.forEach((value, index) => {
      const td = el("td");
      fillCell(td, value);
      if (index > 0 && NUMERIC_TEXT.test(td.textContent.trim())) td.classList.add("text-right");
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  table.append(head, body);
  wrap.appendChild(table);
  return wrap;
}

const RENDERERS = {
  heading: (block) => el("h3", "report-heading", block.text),

  tiles(block) {
    const grid = el("div", "grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4");
    block.items.forEach((item) => {
      const tile = el("div", "tile");
      tile.append(el("p", "tile-label", item.label), el("p", "tile-value", item.value));
      if (item.hint) tile.appendChild(el("p", "tile-hint", item.hint));
      grid.appendChild(tile);
    });
    return grid;
  },

  table(block) {
    const section = el("div", "space-y-1.5");
    if (block.title) section.appendChild(el("h4", "text-sm font-semibold text-slate-700 dark:text-slate-300", block.title));
    section.appendChild(buildTable(block.columns, block.rows));
    if (block.note) section.appendChild(el("p", "hint", block.note));
    return section;
  },

  text(block) {
    const section = el("div", "space-y-2 text-sm text-slate-700 dark:text-slate-300");
    block.paragraphs.forEach((paragraph) => section.appendChild(el("p", null, paragraph)));
    return section;
  },

  alert(block) {
    const node = el("div", `alert alert-${block.kind}`, block.message);
    node.setAttribute("role", "note");
    return node;
  },

  figure(block, figures) {
    const card = el("figure", "plot-card");
    const holder = el("div", "w-full");
    card.appendChild(holder);
    if (block.caption) card.appendChild(el("figcaption", "caption", block.caption));
    const figure = figures[block.index];
    // A microtask runs once the whole report is attached to the DOM, and (unlike requestAnimationFrame)
    // is not paused while the tab is in the background.
    if (figure) queueMicrotask(() => renderFigure(holder, figure));
    return card;
  },
};

/** Replace the contents of `container` with the rendered report. */
export function renderReport(container, report, figures = []) {
  purgeFigures(container);
  container.replaceChildren();
  for (const block of report.blocks ?? []) {
    const render = RENDERERS[block.type];
    if (render) container.appendChild(render(block, figures));
  }
}
