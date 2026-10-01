// Per-module workspace: the form and the last result stay on the server until the user presses «Limpiar», and
// stay in step across every window of the same account.
//
// Mixin for an Alpine page component. The page names the module and the properties that make up its form:
//   Alpine.data("x", () => ({ ...workspace("regresion", ["form", "rows"]), form: {...}, ... }))
// and may override the hooks:  workspaceState() · applyWorkspace(state) · resetWorkspace() · workspacePayload()
// · restorePayload(payload).  `analysisPage()` already includes this mixin.

import { api } from "./api.js";

const SAVE_DELAY_MS = 700;
const clone = (value) => (value === undefined ? undefined : JSON.parse(JSON.stringify(value)));

/**
 * @param {string} module   workspace key (see WORKSPACE_MODULES on the server)
 * @param {string[]} fields component properties that hold the form
 * @param {{dataset?: boolean}} [options] dataset: also remember (and restore) the selected dataset
 */
export function workspace(module, fields = [], { dataset = true } = {}) {
  // timers and in-flight bookkeeping live in a closure, not on the reactive component
  const ws = { revision: 0, lastJson: "", timer: null, saving: false, again: false, payloadToo: false, ready: false, remote: 0 };

  return {
    wsConflict: false,
    wsRestoring: false,
    wsReady: false,

    // ------------------------------------------------------------------ hooks (override in the page)
    workspaceFields: fields,
    /** The form as plain JSON. */
    workspaceState() {
      return Object.fromEntries(this.workspaceFields.map((name) => [name, clone(this[name])]));
    },
    /** Put a stored form back (the dataset has already been selected). May be async. */
    applyWorkspace(state) {
      for (const name of this.workspaceFields) if (name in state) this[name] = clone(state[name]);
    },
    /** Back to an empty form. Pages with a non-trivial default override this (may be async). */
    resetWorkspace() {},
    /** The last result, as stored on the server. */
    workspacePayload() {
      return this.payload ?? null;
    },
    /** Show a stored result. */
    async restorePayload(payload) {
      await this.showPayload(payload);
    },

    // ------------------------------------------------------------------ snapshot
    wsSnapshot() {
      const state = this.workspaceState();
      if (dataset) state.dataset_id = Alpine.store("datasets").activeId || null;
      state.savedId = this.savedId ?? null;
      return state;
    },
    wsJson() {
      return JSON.stringify(this.wsSnapshot());
    },
    wsDirty() {
      return ws.timer !== null || ws.saving || this.wsJson() !== ws.lastJson;
    },

    // ------------------------------------------------------------------ start-up
    /** Call at the end of `init()`: restores (or opens ?load=) and begins autosaving. */
    async startWorkspace() {
      await Alpine.store("datasets").whenLoaded();
      let opened = false;
      if (this.loadSavedFromUrl) {
        opened = await this.loadSavedFromUrl();
        if (opened) history.replaceState(null, "", location.pathname); // a refresh must not reopen it over the workspace
      }
      if (opened) {
        await this.wsPersist(true);
      } else {
        await this.wsRestore();
      }
      this.wsReady = ws.ready = true;
      this.$watch(() => this.wsJson(), () => this.wsScheduleSave());
      Alpine.store("sync").on(`workspace:${module}`, (revision) => this.wsRemoteChanged(revision));
      window.addEventListener("pagehide", () => this.wsFlush(true));
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "hidden") this.wsFlush(true);
      });
    },

    async wsRestore() {
      try {
        const { workspace: stored } = await api(`/api/workspace/${module}`);
        ws.revision = stored.revision;
        await this.wsApply(stored);
      } catch (error) {
        this.error = error.message;
      }
      ws.lastJson = this.wsJson();
    },

    /** Replace the page's content with a stored workspace (an empty one resets the page). */
    async wsApply(stored) {
      this.wsRestoring = true; // selecting the stored dataset must not trigger the «dataset changed» reset
      try {
        const { state, payload } = stored;
        this.payload = null;
        this.error = "";
        if (!state && !payload) {
          this.savedId = null;
          await this.resetWorkspace();
          return;
        }
        const store = Alpine.store("datasets");
        let missingDataset = false;
        if (dataset && state?.dataset_id) {
          if (store.list.some((d) => d.id === state.dataset_id)) {
            if (store.activeId !== state.dataset_id) store.select(state.dataset_id);
          } else {
            missingDataset = true;
          }
        }
        await this.$nextTick();
        if (state && !missingDataset) await this.applyWorkspace(state);
        this.savedId = state?.savedId ?? null;
        if (missingDataset) {
          Alpine.store("toast").show("El dataset que usaba este módulo ya no existe; se conservó el resultado.", "info", 6000);
        }
        await this.$nextTick();
        if (payload) await this.restorePayload(payload);
      } finally {
        this.wsRestoring = false;
      }
    },

    // ------------------------------------------------------------------ saving
    wsScheduleSave() {
      if (!ws.ready || this.wsRestoring || this.wsConflict) return;
      if (this.wsJson() === ws.lastJson) return;
      clearTimeout(ws.timer);
      ws.timer = setTimeout(() => { ws.timer = null; this.wsFlush(); }, SAVE_DELAY_MS);
    },

    /** Save the form now (``leaving``: the page is closing, so the request must outlive it). */
    async wsFlush(leaving = false) {
      // while another window's version is waiting for the user's decision nothing is written over it
      if (!ws.ready || this.wsRestoring || this.wsConflict) return;
      clearTimeout(ws.timer);
      ws.timer = null;
      const json = this.wsJson();
      if (json === ws.lastJson) return;
      if (ws.saving) { ws.again = true; return; }
      await this.wsSend({ state: JSON.parse(json) }, json, leaving);
    },

    /** Save the form together with the current result (after every run, which is when the result changes). */
    async wsPersist(force = false) {
      if ((!ws.ready && !force) || this.wsRestoring || this.wsConflict) return; // restoring/clearing must not write what it just removed
      clearTimeout(ws.timer);
      ws.timer = null;
      if (ws.saving) { ws.again = true; ws.payloadToo = true; return; }
      const json = this.wsJson();
      await this.wsSend({ state: JSON.parse(json), payload: this.workspacePayload() }, json, false);
    },

    /** The result changed (or disappeared): store it with the form. Debounced into the same queue as the form. */
    wsPayloadChanged() {
      if (ws.ready) this.wsPersist();
    },

    async wsSend(body, json, leaving) {
      ws.saving = true;
      try {
        const { revision } = await api(`/api/workspace/${module}`, { method: "PUT", json: body, keepalive: leaving && !body.payload });
        ws.revision = revision;
        ws.lastJson = json;
      } catch (error) {
        if (!leaving) Alpine.store("toast").show(`No se pudo guardar el espacio de trabajo: ${error.message}`, "error", 5000);
      } finally {
        ws.saving = false;
        if (ws.again && !leaving) {
          const withPayload = ws.payloadToo;
          ws.again = false;
          ws.payloadToo = false;
          if (withPayload) await this.wsPersist(); else await this.wsFlush();
        }
      }
    },

    // ------------------------------------------------------------------ other windows
    wsRemoteChanged(revision) {
      if (revision <= ws.revision || ws.saving) return; // our own write
      ws.remote = revision;
      if (this.wsDirty()) { this.wsConflict = true; return; } // unsaved local edits: let the user choose
      this.wsLoadRemote();
    },

    async wsLoadRemote() {
      try {
        const { workspace: stored } = await api(`/api/workspace/${module}`);
        ws.revision = stored.revision;
        this.wsConflict = false;
        await this.wsApply(stored);
        ws.lastJson = this.wsJson();
      } catch (error) {
        this.error = error.message;
      }
    },

    /** «Cargar la versión nueva» */
    wsAcceptRemote() {
      return this.wsLoadRemote();
    },

    /** «Mantener la mía»: the next save overwrites the other window's version. */
    async wsKeepMine() {
      this.wsConflict = false;
      ws.revision = Math.max(ws.revision, ws.remote);
      ws.lastJson = "";
      await this.wsPersist();
    },

    // ------------------------------------------------------------------ «Limpiar»
    async clearWorkspace() {
      clearTimeout(ws.timer);
      ws.timer = null;
      this.wsRestoring = true;
      try {
        this.payload = null;
        this.error = "";
        this.savedId = null;
        await this.resetWorkspace();
        await this.$nextTick();
      } finally {
        this.wsRestoring = false;
      }
      try {
        const { revision } = await api(`/api/workspace/${module}`, { method: "DELETE" });
        ws.revision = revision;
        this.wsConflict = false;
      } catch (error) {
        Alpine.store("toast").show(error.message, "error");
      }
      ws.lastJson = this.wsJson();
    },
  };
}
