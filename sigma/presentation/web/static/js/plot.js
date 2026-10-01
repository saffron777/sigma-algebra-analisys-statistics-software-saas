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

/** Re-apply the theme to every figure on the page (called when the theme toggles). */
export function rethemeAll() {
  document.querySelectorAll("[data-plot]").forEach((element) => {
    if (element._figure) renderFigure(element, element._figure);
  });
}

export function purgeFigures(container) {
  container.querySelectorAll("[data-plot]").forEach((element) => Plotly.purge(element));
}
