// Thin fetch wrapper: sends the CSRF token, parses JSON and turns error payloads into ApiError.

export class ApiError extends Error {
  constructor(message, { status = 0, code = "error", details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

const csrfToken = () => document.querySelector('meta[name="csrf-token"]')?.content ?? "";

/**
 * @param {string} url
 * @param {{method?: string, json?: unknown, form?: FormData, signal?: AbortSignal, keepalive?: boolean}} [options]
 * ``keepalive`` lets a request outlive the page (used when a tab is closed); browsers cap such bodies at 64 KB.
 */
export async function api(url, { method = "GET", json, form, signal, keepalive = false } = {}) {
  const headers = { Accept: "application/json", "X-CSRF-Token": csrfToken() };
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form; // the browser sets the multipart boundary
  }

  let response;
  try {
    response = await fetch(url, { method, headers, body, signal, keepalive, credentials: "same-origin" });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("No se pudo conectar con el servidor de Sigma Algebra. ¿Sigue en ejecución?");
  }

  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON body */
  }
  if (response.status === 401 && !url.startsWith("/api/auth/")) {
    // the session expired (or the account was disabled): go back to the login page and return here afterwards
    location.assign(`/login?next=${encodeURIComponent(location.pathname + location.search)}`);
  }
  if (!response.ok) {
    const info = data?.error;
    throw new ApiError(info?.message ?? `Error ${response.status} del servidor.`, {
      status: response.status,
      code: info?.code,
      details: info?.details,
    });
  }
  return data;
}

/** Download helper for endpoints that answer with a file. */
export function downloadUrl(url) {
  const link = document.createElement("a");
  link.href = url;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/** The file name a download response suggests (RFC 5987 ``filename*`` first, then ``filename``). */
function filenameOf(response, fallback) {
  const header = response.headers.get("Content-Disposition") ?? "";
  const star = /filename\*=UTF-8''([^;]+)/i.exec(header);
  if (star) {
    try { return decodeURIComponent(star[1]); } catch { /* fall through */ }
  }
  return /filename="?([^";]+)"?/i.exec(header)?.[1] ?? fallback;
}

/**
 * Fetch a file (report) and return `{blob, filename}`; error payloads become ApiError like in `api()`.
 * @param {string} url
 * @param {{method?: string, json?: unknown, form?: FormData, signal?: AbortSignal}} [options]
 */
export async function fetchFile(url, { method = "GET", json, form, signal } = {}) {
  const headers = { "X-CSRF-Token": csrfToken() };
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form;
  }
  let response;
  try {
    response = await fetch(url, { method, headers, body, signal, credentials: "same-origin" });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("No se pudo conectar con el servidor de Sigma Algebra. ¿Sigue en ejecución?");
  }
  if (response.status === 401) location.assign(`/login?next=${encodeURIComponent(location.pathname + location.search)}`);
  if (!response.ok) {
    let info = null;
    try { info = (await response.json())?.error; } catch { /* not JSON */ }
    throw new ApiError(info?.message ?? `Error ${response.status} del servidor.`, { status: response.status, code: info?.code });
  }
  return { blob: await response.blob(), filename: filenameOf(response, "informe") };
}

/** Hand a Blob to the browser as a download. */
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
