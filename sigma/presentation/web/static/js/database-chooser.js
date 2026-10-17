// «Base de datos» chooser shared by the first-run wizard and the administration page.
// It talks to /api/setup/database (before the first account exists) or /api/admin/database (administrators).

import { api } from "./api.js";

const blankForm = () => ({
  engine: "sqlite",
  host: "",
  port: "",
  database: "",
  schema: "",
  username: "",
  password: "",
  windows: false, // SQL Server: Windows authentication
  trustCertificate: false, // SQL Server: accept the server's TLS certificate
  sslmode: "", // PostgreSQL
  noTls: false, // MySQL / MariaDB
});

const SCHEMA_ENGINES = ["mssql", "postgresql"];

document.addEventListener("alpine:init", () => {
  Alpine.data("databaseChooser", (base, mode = "admin") => ({
    base,
    mode, // "setup" (wizard, collapsible) | "admin"
    loading: true,
    loadError: "",
    status: null,
    engines: [],
    form: blankForm(),
    open: mode === "admin",
    report: null,
    testing: false,
    applying: false,
    error: "",
    applied: null,
    migrate: true, // copy the live data when the chosen database is empty
    migrating: false,
    migrated: false,
    job: null, // progress of the background migration

    async init() {
      try {
        this.status = await api(this.base);
        this.engines = this.status.engines;
        this.fromCurrent(this.status.current);
      } catch (error) {
        this.loadError = error.message;
      } finally {
        this.loading = false;
      }
    },

    fromCurrent(current) {
      const options = current.options || {};
      this.form = {
        ...blankForm(),
        engine: current.engine,
        host: current.host || "",
        port: current.port ?? "",
        database: current.database || "",
        schema: current.schema || "",
        username: current.username || "",
        windows: options.trusted_connection === "yes",
        trustCertificate: options.TrustServerCertificate === "yes",
        sslmode: options.sslmode || "",
        noTls: options.ssl_disabled === "true",
      };
      this.report = null;
    },

    // ------------------------------------------------------------------ derived
    get meta() {
      return this.engines.find((e) => e.engine === this.form.engine) || {};
    },
    get isSqlite() {
      return this.form.engine === "sqlite";
    },
    get editable() {
      return !!this.status?.editable;
    },
    get databaseLabel() {
      return this.form.engine === "oracle" ? "Nombre del servicio" : "Base de datos";
    },
    get supportsSchema() {
      return SCHEMA_ENGINES.includes(this.form.engine);
    },
    get usesLogin() {
      return !(this.form.engine === "mssql" && this.form.windows);
    },
    /** The stored password is kept when the target (engine, server, port, user) is unchanged and the field stays empty. */
    get keepsPassword() {
      const c = this.status?.current;
      return !!(c?.has_password && c.engine === this.form.engine && (c.host || "") === this.form.host.trim()
        && String(c.port ?? "") === String(this.form.port ?? "") && (c.username || "") === this.form.username.trim());
    },
    get summary() {
      const c = this.status?.current;
      if (!c) return "";
      if (c.engine === "sqlite") return `${c.label} · ${c.path}`;
      return `${c.label} · ${c.host}${c.port ? ":" + c.port : ""} · ${c.database}${c.schema ? " (esquema " + c.schema + ")" : ""}`;
    },
    get isCurrent() {
      const c = this.status?.current;
      if (!c || c.engine !== this.form.engine) return false;
      if (this.isSqlite) return true;
      return (c.host || "") === this.form.host.trim() && String(c.port ?? "") === String(this.form.port ?? "")
        && (c.database || "") === this.form.database.trim() && (c.schema || "") === this.form.schema.trim();
    },
    /** A migration is offered when the target is empty and the database in use has accounts to carry over. */
    get canMigrate() {
      return this.mode === "admin" && this.report?.state === "empty" && (this.status?.live?.counts?.users ?? 0) > 0;
    },
    get willMigrate() {
      return this.canMigrate && this.migrate;
    },
    get progress() {
      const steps = this.job?.steps || [];
      const done = steps.filter((step) => step.state === "done").length;
      return steps.length ? Math.round((done / steps.length) * 100) : 0;
    },
    get canApply() {
      return this.editable && !this.isCurrent && !!this.report?.ok && !this.applying && !this.testing;
    },

    // ------------------------------------------------------------------ editing
    changeEngine() {
      const known = this.engines.map((e) => String(e.default_port ?? ""));
      if (this.form.port === "" || known.includes(String(this.form.port))) {
        this.form.port = this.meta.default_port ?? "";
      }
      this.edited();
      if (this.isSqlite) this.test(); // nothing to fill in: check the local file straight away
    },
    edited() {
      this.report = null;
      this.error = "";
    },
    payload() {
      const f = this.form;
      if (this.isSqlite) return { engine: "sqlite" };
      const options = {};
      if (f.engine === "mssql") {
        if (f.windows) options.trusted_connection = "yes";
        if (f.trustCertificate) options.TrustServerCertificate = "yes";
      }
      if (f.engine === "postgresql" && f.sslmode) options.sslmode = f.sslmode;
      if (f.engine === "mysql" && f.noTls) options.ssl_disabled = "true";
      return {
        engine: f.engine,
        host: f.host,
        port: f.port === "" ? null : Number(f.port),
        database: f.database,
        schema: this.supportsSchema ? f.schema : "",
        username: this.usesLogin ? f.username : "",
        password: this.usesLogin ? f.password : "",
        options,
        migrate: this.willMigrate,
      };
    },

    // ------------------------------------------------------------------ actions
    async test() {
      this.testing = true;
      this.error = "";
      this.report = null;
      try {
        this.report = await api(`${this.base}/test`, { method: "POST", json: this.payload() });
      } catch (error) {
        this.error = error.message;
      } finally {
        this.testing = false;
      }
    },

    async apply() {
      if (!this.canApply) return;
      const target = this.meta.label;
      if (this.mode === "admin" && !window.confirm(this.confirmText(target))) return;
      this.applying = true;
      this.error = "";
      try {
        const result = await api(`${this.base}/apply`, { method: "POST", json: this.payload() });
        this.form.password = "";
        if (result.migrating) {
          this.job = result.job;
          this.migrating = true;
          this.poll();
          return; // `applying` stays on until the migration ends
        }
        if (this.mode === "admin") {
          location.assign(result.setup_required ? "/setup" : "/login");
          return;
        }
        if (!result.setup_required) {
          location.assign("/login"); // that database already has accounts
          return;
        }
        this.applied = target;
        this.open = false;
        this.status = await api(this.base);
        this.fromCurrent(this.status.current);
      } catch (error) {
        this.error = error.message;
      } finally {
        if (!this.migrating) this.applying = false;
      }
    },

    confirmText(target) {
      if (this.willMigrate) {
        return `Se copiarán a «${target}» todos los datos de Sigma Algebra (cuentas, conjuntos de datos, resultados, notebooks…).\n\n`
          + "• Mientras dure la copia Sigma Algebra no estará disponible para nadie (minutos, según el volumen).\n"
          + "• Se cerrarán las consolas de Python en ejecución; las sesiones se conservan.\n"
          + "• La base actual no se modifica ni se borra: queda como copia de seguridad.\n\n¿Continuar?";
      }
      return `Sigma Algebra pasará a guardar todo en «${target}».\n\n`
        + "• Los datos de la base actual NO se copian: se usa lo que ya haya en la nueva base (si está vacía, deberá crear de nuevo la cuenta de administrador).\n"
        + "• Se cerrarán las sesiones y las consolas de Python en ejecución.\n"
        + "• La base actual no se borra: puede volver a ella cuando quiera.\n\n¿Continuar?";
    },

    /** Follows the background migration until it ends (the page keeps working while Sigma Algebra is in maintenance). */
    async poll() {
      while (this.migrating) {
        await new Promise((resolve) => setTimeout(resolve, 800));
        try {
          this.job = await api(`${this.base}/migration`);
        } catch {
          continue; // the server is busy switching: ask again
        }
        if (this.job.state === "done") {
          this.migrating = false;
          this.migrated = true;
          this.applying = false;
          setTimeout(() => location.assign("/basedatos"), 2500);
        } else if (this.job.state === "failed") {
          this.migrating = false;
          this.applying = false;
          this.error = this.job.message || "La migración falló.";
        }
      }
    },
  }));
});
