// Plotly rendering with the active light/dark theme applied at draw time.

const palette = (dark) =>
  dark
    ? { text: "#e2e8f0", grid: "rgba(148,163,184,0.20)", line: "rgba(148,163,184,0.35)" }
    : { text: "#334155", grid: "rgba(100,116,139,0.20)", line: "rgba(100,116,139,0.35)" };

const isDark = () => document.documentElement.classList.contains("dark");

function themed(layout) {
  const colors = palette(isDark());
  const result = plain(layout ?? {});
  result.autosize = true;
  result.font = { ...(result.font ?? {}), color: colors.text };
  if (result.title && typeof result.title === "object") {
    result.title.font = { ...(result.title.font ?? {}), color: colors.text };
  }
  for (const key of Object.keys(result)) {
    if (/^(x|y)axis\d*$/.test(key) && typeof result[key] === "object") {
      result[key] = { ...result[key], gridcolor: colors.grid, zerolinecolor: colors.grid, linecolor: colors.line };
    }
  }
  return result;
}

/** A deep copy that also works on Alpine's reactive proxies (``structuredClone`` throws on those). */
function plain(value) {
  try {
    return structuredClone(value);
  } catch {
    return JSON.parse(JSON.stringify(value));
  }
}

const CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  toImageButtonOptions: { format: "png", scale: 2, filename: "sigma-grafico" },
};

/** Draw (or redraw) a figure `{data, layout}` into `element`. */
export function renderFigure(element, figure) {
  element.dataset.plot = "1";
  element._figure = figure;
  Plotly.react(element, plain(figure.data ?? []), themed(figure.layout), CONFIG);
  requestAnimationFrame(() => {
    // the element may have been hidden or removed (e.g. another result replaced this one) by the next frame
    if (element.isConnected && element.offsetParent !== null) Plotly.Plots.resize(element).catch(() => {});
  });
}

/** Re-apply the theme to every figure on the page (called when the theme toggles); resolves once all are drawn. */
export function rethemeAll() {
  const drawing = [];
  document.querySelectorAll("[data-plot]").forEach((element) => {
    if (element._figure) drawing.push(Plotly.react(element, plain(element._figure.data ?? []), themed(element._figure.layout), CONFIG));
  });
  return Promise.all(drawing).catch(() => {});
}

const dataUrlToBlob = (url) => {
  const [head, data] = url.split(",");
  const bytes = Uint8Array.from(atob(data), (char) => char.charCodeAt(0));
  return new Blob([bytes], { type: /data:([^;]+)/.exec(head)?.[1] ?? "image/png" });
};

/**
 * Draw a figure off-screen with the LIGHT theme (reports are printed on white) and return it as a PNG Blob.
 * Independent of the on-screen theme and of whether the figure is currently visible.
 */
export async function figureToPng(figure, { width = 980, scale = 2 } = {}) {
  const height = Math.round(Math.min(Math.max(figure.layout?.height ?? 460, 260), 900));
  const holder = document.createElement("div");
  holder.style.cssText = `position:fixed;left:-20000px;top:0;width:${width}px;height:${height}px`;
  document.body.appendChild(holder);
  try {
    const colors = palette(false);
    const layout = {
      ...plain(figure.layout ?? {}), autosize: false, width, height,
      paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", font: { ...(figure.layout?.font ?? {}), color: colors.text },
    };
    await Plotly.newPlot(holder, plain(figure.data ?? []), layout, { staticPlot: true, displayModeBar: false });
    const url = await Plotly.toImage(holder, { format: "png", width, height, scale });
    return dataUrlToBlob(url);
  } finally {
    Plotly.purge(holder);
    holder.remove();
  }
}

/**
 * Before printing: draw every visible figure as a static image on white (browsers print live SVG charts at the
 * screen width and in the dark theme). Returns a function that undoes it.
 */
export async function freezeFiguresForPrint() {
  const dark = isDark();
  if (dark) {
    document.documentElement.classList.remove("dark");
    await rethemeAll();
  }
  const images = [];
  for (const element of document.querySelectorAll("[data-plot]")) {
    if (!element.isConnected || element.offsetParent === null) continue;
    try {
      const width = Math.round(element.clientWidth) || 800;
      const height = Math.round(element.clientHeight) || 420;
      const url = await Plotly.toImage(element, { format: "png", width, height, scale: 2 });
      const image = Object.assign(document.createElement("img"), { src: url, alt: "Gráfico", className: "print-figure" });
      element.after(image);
      element.dataset.plotFrozen = "1";
      images.push([element, image]);
    } catch { /* a figure that cannot be frozen prints as it is */ }
  }
  return async () => {
    for (const [element, image] of images) {
      image.remove();
      delete element.dataset.plotFrozen;
    }
    if (dark) {
      document.documentElement.classList.add("dark");
      await rethemeAll();
    }
  };
}

export function purgeFigures(container) {
  container.querySelectorAll("[data-plot]").forEach((element) => Plotly.purge(element));
}
