// Rendering of Jupyter outputs and markdown. Anything that may contain HTML (imported notebooks are untrusted)
// goes through DOMPurify first; plain text always goes through textContent.

const ANSI = /\u001b\[[0-9;?]*[A-Za-z]/g;

const PURIFY_OPTIONS = {
  FORBID_TAGS: ["style", "form", "input", "button", "textarea", "select", "iframe", "object", "embed", "link", "meta"],
  FORBID_ATTR: ["style", "srcset"],
  ALLOW_DATA_ATTR: false,
};

export const stripAnsi = (text) => String(text ?? "").replace(ANSI, "");
const joined = (value) => (Array.isArray(value) ? value.join("") : String(value ?? ""));

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function safeHtml(html) {
  const holder = element("div", "rendered-html");
  holder.innerHTML = DOMPurify.sanitize(joined(html), PURIFY_OPTIONS);
  holder.querySelectorAll("a[href]").forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
  return holder;
}

export function renderMarkdown(container, source) {
  const html = marked.parse(joined(source), { gfm: true, breaks: false });
  container.replaceChildren(safeHtml(html));
}

function renderBundle(data) {
  for (const type of ["image/png", "image/jpeg"]) {
    if (data[type]) {
      const image = element("img", "max-w-full rounded bg-white");
      image.src = `data:${type};base64,${joined(data[type]).replace(/\s/g, "")}`;
      image.alt = "Gráfico generado por el código";
      return image;
    }
  }
  if (data["image/svg+xml"]) {
    const image = element("img", "max-w-full rounded bg-white");
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(joined(data["image/svg+xml"]))}`; // as <img>: scripts never run
    image.alt = "Gráfico vectorial generado por el código";
    return image;
  }
  if (data["text/html"]) return safeHtml(data["text/html"]);
  if (data["text/markdown"]) {
    const holder = element("div");
    renderMarkdown(holder, data["text/markdown"]);
    return holder;
  }
  if (data["application/json"]) return element("pre", "cell-output", JSON.stringify(data["application/json"], null, 2));
  return element("pre", "cell-output", stripAnsi(joined(data["text/plain"])));
}

/** Render a list of nbformat outputs into `container`. */
export function renderOutputs(container, outputs) {
  container.replaceChildren();
  for (const output of outputs ?? []) {
    let node;
    if (output.output_type === "stream") {
      node = element("pre", `cell-output ${output.name === "stderr" ? "text-amber-700 dark:text-amber-400" : ""}`, stripAnsi(joined(output.text)));
    } else if (output.output_type === "error") {
      const trace = (output.traceback ?? []).map(stripAnsi).join("\n") || `${output.ename}: ${output.evalue}`;
      node = element("pre", "cell-output tone-bad", trace);
    } else if (output.data) {
      node = renderBundle(output.data);
      node.classList.add("px-3", "py-2");
    }
    if (node) container.appendChild(node);
  }
}
