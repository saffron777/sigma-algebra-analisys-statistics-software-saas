"""Generates the demo datasets in ``sample_data/`` (deterministic: same files on every run).

python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent.parent / "sample_data"


def sales(rng: np.random.Generator) -> pd.DataFrame:
    """300 monthly sales records: numeric drivers, two factors and a binary outcome."""
    n = 300
    price = rng.normal(50, 8, n).round(2)
    advertising = rng.gamma(6, 3.5, n).round(2)
    competition = rng.integers(2, 15, n)
    region = rng.choice(["Norte", "Sur", "Centro", "Oeste"], n, p=[0.3, 0.25, 0.3, 0.15])
    channel = rng.choice(["Online", "Tienda"], n, p=[0.45, 0.55])
    region_effect = pd.Series(region).map({"Norte": 6, "Sur": -4, "Centro": 0, "Oeste": 10}).to_numpy()
    channel_effect = np.where(channel == "Online", 5, 0)
    units = (
        120 - 1.6 * price + 2.4 * advertising - 1.8 * competition + region_effect + channel_effect + rng.normal(0, 9, n)
    )
    return pd.DataFrame(
        {
            "mes": rng.integers(1, 13, n),
            "region": region,
            "canal": channel,
            "precio": price,
            "publicidad": advertising,
            "competidores": competition,
            "unidades": units.round(1),
            "cliente_nuevo": rng.choice(["Sí", "No"], n, p=[0.35, 0.65]),
        }
    )


def sampling_frame(rng: np.random.Generator) -> pd.DataFrame:
    """1 000 households grouped in districts (strata) and blocks (clusters)."""
    districts = np.repeat(["Norte", "Sur", "Este", "Oeste"], [400, 300, 200, 100])
    block = np.arange(1000) // 20 + 1
    income = np.where(districts == "Norte", rng.normal(1500, 300, 1000), rng.normal(1100, 250, 1000)).round(0)
    return pd.DataFrame(
        {
            "id_hogar": np.arange(1, 1001),
            "distrito": districts,
            "manzana": block,
            "miembros": rng.integers(1, 8, 1000),
            "ingreso": income,
            "vivienda": rng.choice(["Propia", "Alquilada", "Familiar"], 1000, p=[0.55, 0.3, 0.15]),
        }
    )


def survey(rng: np.random.Generator) -> pd.DataFrame:
    """500 respondents of a satisfaction survey with a 5-point Likert block."""
    n = 500
    base = rng.normal(3.4, 1.0, n)
    likert = {f"p{i}": np.clip(np.round(base + rng.normal(0, 0.7, n)), 1, 5).astype(int) for i in range(1, 6)}
    age = rng.integers(18, 70, n)
    gender = rng.choice(["Femenino", "Masculino", "Otro"], n, p=[0.52, 0.45, 0.03])
    recommend = np.where(base + rng.normal(0, 0.6, n) > 3.4, "Sí", "No")
    return pd.DataFrame(
        {
            "edad": age,
            "genero": gender,
            "region": rng.choice(["Norte", "Sur", "Centro", "Oeste"], n),
            "ingreso": rng.lognormal(7.1, 0.4, n).round(0),
            **likert,
            "recomienda": recommend,
        }
    )


def demo_notebook() -> None:
    """A small notebook that exercises the console: sigma.load, pandas, matplotlib, statsmodels and sigma.save."""
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    notebook = new_notebook()
    notebook.metadata["title"] = "Análisis de ventas (demo)"
    notebook.cells = [
        new_markdown_cell(
            "# Análisis de ventas (demo)\n\n"
            "Este notebook usa el dataset **ventas** de Sigma Algebra (cárguelo con `python scripts/load_sample_data.py`).\n"
            "Ejecute las celdas con **Shift+Enter**."
        ),
        new_code_cell('ventas = sigma.load("ventas")\nventas.head()'),
        new_code_cell('ventas.groupby("region")["unidades"].agg(["count", "mean", "std"]).round(2)'),
        new_markdown_cell("## Relación entre precio y unidades vendidas"),
        new_code_cell(
            "fig, ax = plt.subplots()\n"
            'ax.scatter(ventas["precio"], ventas["unidades"], alpha=0.5, s=18)\n'
            'slope, intercept = np.polyfit(ventas["precio"], ventas["unidades"], 1)\n'
            'xs = np.linspace(ventas["precio"].min(), ventas["precio"].max(), 50)\n'
            'ax.plot(xs, intercept + slope * xs, color="crimson")\n'
            'ax.set_xlabel("Precio"); ax.set_ylabel("Unidades"); ax.set_title(f"Pendiente = {slope:.2f}")\n'
            "plt.show()"
        ),
        new_markdown_cell("## Regresión múltiple con statsmodels"),
        new_code_cell(
            "import statsmodels.api as sm\n"
            'X = sm.add_constant(ventas[["precio", "publicidad", "competidores"]])\n'
            'modelo = sm.OLS(ventas["unidades"], X).fit()\n'
            "print(modelo.summary())"
        ),
        new_code_cell(
            'online = ventas.loc[ventas["canal"] == "Online", "unidades"]\n'
            'tienda = ventas.loc[ventas["canal"] == "Tienda", "unidades"]\n'
            "stats.ttest_ind(online, tienda, equal_var=False)"
        ),
        new_markdown_cell(
            "## Guardar un resultado en Sigma Algebra\n\nEl DataFrame resultante aparece en la página **Datos**."
        ),
        new_code_cell(
            'resumen = ventas.groupby("region")["unidades"].mean().round(2).reset_index(name="unidades_promedio")\n'
            'sigma.save(resumen, "resumen_regiones", replace=True)'
        ),
    ]
    nbformat.validate(notebook)
    path = OUTPUT / "analisis_demo.ipynb"
    path.write_text(nbformat.writes(notebook), encoding="utf-8")
    print(f"{path.name}: {len(notebook.cells)} celdas")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(2024)
    for name, frame in {"ventas": sales(rng), "marco_muestral": sampling_frame(rng), "encuesta": survey(rng)}.items():
        path = OUTPUT / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        print(f"{path.name}: {len(frame)} filas x {frame.shape[1]} columnas")
    demo_notebook()


if __name__ == "__main__":
    main()
