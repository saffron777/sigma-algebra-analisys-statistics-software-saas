"""Rows and structure of a dataset (SPSS *Data*, Minitab *Data*): filter, sort, duplicates, samples, partitions, reshape."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sigma.domain.analysis import registry
from sigma.domain.analysis.expressions import REFERENCE, evaluate
from sigma.domain.analysis.output import Output
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import (
    MAX_NAME,
    MAX_NEW_COLUMNS,
    RESULT,
    RESULT_KEY,
    cell,
    column_name,
    load_line,
    preview,
    transformed,
)
from sigma.domain.datasets import classify_column
from sigma.domain.errors import InvalidInputError

ROWS = "Filas"
STRUCTURE = "Estructura"
_SAVE = "\n# sigma.save(df, 'nombre_del_dataset_nuevo')\n"
MAX_MELT_ROWS = 5_000_000


def _fresh(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reset_index(drop=True)


# ============================================================================ filter
def _filter(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    mask = evaluate(p["condition"], frame)
    if not pd.api.types.is_bool_dtype(mask):
        raise InvalidInputError('La condición debe dar verdadero o falso, por ejemplo: x1 > 10 and g == "A".')
    keep = mask.fillna(False).astype(bool)
    if p["action"] == "drop":
        keep = ~keep
    after = _fresh(frame[keep])
    if after.empty:
        raise InvalidInputError("Ninguna fila queda con esa condición.")
    verb = "conservan" if p["action"] == "keep" else "quedan tras quitar las que cumplen"
    out = transformed("Filtrar filas", f"{p['condition'].strip()} · se {verb}", frame, after)
    out.tiles(
        [
            ("Filas que quedan", len(after)),
            ("Filas quitadas", len(frame) - len(after)),
            ("% conservado", 100 * len(after) / len(frame)),
        ]
    )
    preview(out, after, list(after.columns))
    return out.set(kept=len(after), removed=len(frame) - len(after)).build()


def _filter_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    negate = "~" if p["action"] == "drop" else ""
    return (
        load_line(dataset)
        + f"df = df[{negate}sigma.calc(df, {p['condition'].strip()!r})].reset_index(drop=True)\n"
        + _SAVE
    )


# ============================================================================ sort
def _sort(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cols = roles["cols"]
    try:
        after = _fresh(frame.sort_values(cols, ascending=not p["descending"], na_position=p["missing"], kind="stable"))
    except TypeError:
        raise InvalidInputError(
            "No se pueden ordenar juntas esas columnas: mezclan tipos de datos incompatibles."
        ) from None
    out = transformed(
        "Ordenar filas", " → ".join(cols) + (" (descendente)" if p["descending"] else " (ascendente)"), frame, after
    )
    preview(out, after, cols)
    return out.build()


def _sort_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        load_line(dataset)
        + f"df = df.sort_values({roles['cols']!r}, ascending={not p['descending']}, na_position={p['missing']!r}, kind='stable').reset_index(drop=True)\n"
        + _SAVE
    )


# ============================================================================ duplicates
def _duplicates(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    subset = roles.get("cols") or None
    keep = {"first": "first", "last": "last", "none": False}[p["keep"]]
    marked = frame.duplicated(subset=subset, keep=keep)
    n_marked = int(marked.sum())
    key = subset or list(frame.columns)
    repeated = frame.groupby(key, dropna=False, sort=False).size()
    repeated = repeated[repeated > 1].sort_values(ascending=False)
    if p["action"] == "flag":
        name = column_name(frame, "duplicado")
        after = frame.copy()
        after[name] = marked.astype("int64")
        created = [name]
    else:
        after = _fresh(frame[~marked])
        created = []
    out = transformed(
        "Filas duplicadas", f"según {', '.join(key) if subset else 'todas las columnas'}", frame, after, created=created
    )
    out.tiles(
        [
            ("Filas marcadas", n_marked),
            ("Grupos repetidos", len(repeated)),
            ("% de filas", 100 * n_marked / max(len(frame), 1)),
        ]
    )
    if len(repeated):
        shown = repeated.head(10)
        out.table(
            "Combinaciones más repetidas",
            [*key[:6], "Veces"],
            [
                [*(cell(v) for v in (idx if isinstance(idx, tuple) else (idx,)))][:6] + [int(n)]
                for idx, n in shown.items()
            ],
            note="Con «Ninguno» se marcan todas las copias; con «Primero»/«Último» solo las demás.",
        )
    else:
        out.alert("success", "No hay filas duplicadas con ese criterio.")
    return out.set(duplicates=n_marked).build()


def _duplicates_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    subset = f"subset={roles['cols']!r}, " if roles.get("cols") else ""
    keep = {"first": "'first'", "last": "'last'", "none": "False"}[p["keep"]]
    if p["action"] == "flag":
        return load_line(dataset) + f"df['duplicado'] = df.duplicated({subset}keep={keep}).astype(int)\n" + _SAVE
    return load_line(dataset) + f"df = df.drop_duplicates({subset}keep={keep}).reset_index(drop=True)\n" + _SAVE


# ============================================================================ sample
def _allocation(sizes: pd.Series, total: int) -> pd.Series:
    """Proportional allocation with largest remainders (every stratum with at least one row keeps one)."""
    exact = sizes / sizes.sum() * total
    take = np.floor(exact).astype(int)
    for label in (exact - take).sort_values(ascending=False).index[: total - int(take.sum())]:
        take[label] += 1
    return take.clip(lower=0).where(take > 0, 1).clip(upper=sizes)


def _sample(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    n_rows = len(frame)
    if p["mode"] == "n":
        size = int(p["size"])
    else:
        if not 0 < p["size"] <= (5 if p["replace"] else 1):
            raise InvalidInputError("La fracción debe estar entre 0 y 1 (hasta 5 si el muestreo es con reemplazo).")
        size = max(int(round(p["size"] * n_rows)), 1)
    if size < 1 or (size > n_rows and not p["replace"]):
        raise InvalidInputError(f"El tamaño debe estar entre 1 y {n_rows} (más solo si el muestreo es con reemplazo).")
    seed = int(p["seed"])
    strata = roles.get("strata")
    if strata:
        groups = frame.groupby(strata, dropna=False, sort=False)
        sizes = groups.size()
        take = (
            _allocation(sizes, size) if not p["replace"] else (sizes / n_rows * size).round().astype(int).clip(lower=1)
        )
        picked = [
            frame.loc[idx].sample(n=int(take[key]), replace=p["replace"], random_state=seed + i)
            for i, (key, idx) in enumerate(groups.groups.items())
        ]
        drawn = pd.concat(picked)
    else:
        drawn = frame.sample(n=size, replace=p["replace"], random_state=seed)
    after = _fresh(drawn.sort_index())
    out = transformed("Muestra aleatoria", f"n = {len(after)} de {n_rows} · semilla {seed}", frame, after)
    out.tiles([("Tamaño de la muestra", len(after)), ("Población", n_rows), ("Fracción", len(after) / n_rows)])
    if strata:
        pop, smp = (
            frame[strata].value_counts(normalize=True, dropna=False),
            after[strata].value_counts(normalize=True, dropna=False),
        )
        out.table(
            f"Composición por «{strata}»",
            ["Estrato", "% población", "% muestra", "Filas en la muestra"],
            [
                [
                    cell(k),
                    100 * pop[k],
                    100 * smp.get(k, 0.0),
                    int((after[strata] == k).sum()) if not pd.isna(k) else int(after[strata].isna().sum()),
                ]
                for k in pop.index
            ],
        )
    return out.set(n=len(after), seed=seed).build()


def _sample_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    arg = f"n={int(p['size'])}" if p["mode"] == "n" else f"frac={p['size']}"
    if roles.get("strata"):
        body = f"df = df.groupby({roles['strata']!r}, group_keys=False).sample({'frac=' + str(round(p['size'] / 1, 6)) if p['mode'] != 'n' else 'frac=' + str(int(p['size'])) + ' / len(df)'}, random_state={int(p['seed'])})\n"
    else:
        body = f"df = df.sample({arg}, replace={bool(p['replace'])}, random_state={int(p['seed'])}).sort_index().reset_index(drop=True)\n"
    return load_line(dataset) + body + _SAVE


# ============================================================================ train / test partition
def _partition(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    test_fraction, valid_fraction, seed = p["test_fraction"], p["validation_fraction"], int(p["seed"])
    if test_fraction + valid_fraction >= 0.95:
        raise InvalidInputError("Las fracciones de prueba y validación deben sumar menos de 0.95.")
    n = len(frame)
    if n < 10:
        raise InvalidInputError("Se necesitan al menos 10 filas para particionar.")
    if p["method"] == "sequential":
        order = frame.sort_values(roles["time"], kind="stable").index if roles.get("time") else frame.index
        n_test, n_valid = int(round(n * test_fraction)), int(round(n * valid_fraction))
        test_idx = order[n - n_test :]
        valid_idx = order[n - n_test - n_valid : n - n_test]
        train_idx = order[: n - n_test - n_valid]
    else:
        rng = np.random.default_rng(seed)
        labels = np.zeros(n, dtype=int)  # 0 train, 1 test, 2 validation
        groups = (
            [np.arange(n)]
            if not roles.get("strata")
            else [np.flatnonzero(g) for g in _strata_masks(frame, roles["strata"])]
        )
        for members in groups:
            shuffled = rng.permutation(members)
            n_test, n_valid = int(round(len(shuffled) * test_fraction)), int(round(len(shuffled) * valid_fraction))
            labels[shuffled[:n_test]] = 1
            labels[shuffled[n_test : n_test + n_valid]] = 2
        train_idx, test_idx, valid_idx = (frame.index[labels == k] for k in (0, 1, 2))
    if len(train_idx) < 2 or len(test_idx) < 1:
        raise InvalidInputError(
            "Con esas fracciones el conjunto de entrenamiento o el de prueba queda demasiado pequeño."
        )
    out = Output(
        "Partición entrenamiento / prueba",
        f"{'secuencial' if p['method'] == 'sequential' else 'aleatoria'} · semilla {seed}",
    )
    parts = [("train", "Entrenamiento", train_idx), ("test", "Prueba", test_idx)] + (
        [("validation", "Validación", valid_idx)] if len(valid_idx) else []
    )
    out.tiles([(label, len(idx)) for _, label, idx in parts])
    out.table(
        "Conjuntos", ["Conjunto", "Filas", "%"], [[label, len(idx), 100 * len(idx) / n] for _, label, idx in parts]
    )
    for key, label, idx in parts:
        out.dataset(key, label, _fresh(frame.loc[idx].sort_index()))
    if roles.get("strata"):
        column = roles["strata"]
        levels = frame[column].value_counts(dropna=False).index
        out.table(
            f"Composición por «{column}» (%)",
            ["Estrato", *(label for _, label, _ in parts)],
            [
                [
                    cell(k),
                    *(
                        100 * float((frame.loc[idx, column] == k).mean())
                        if not pd.isna(k)
                        else 100 * float(frame.loc[idx, column].isna().mean())
                        for _, _, idx in parts
                    ),
                ]
                for k in levels
            ],
        )
    return out.set(**{key: len(idx) for key, _, idx in parts}).build()


def _strata_masks(frame: pd.DataFrame, column: str):
    for level in frame[column].unique():
        yield (frame[column].isna() if pd.isna(level) else frame[column] == level).to_numpy()


def _partition_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    if p["method"] == "sequential":
        sort = f"df = df.sort_values({roles['time']!r})\n" if roles.get("time") else ""
        return (
            load_line(dataset)
            + sort
            + f"cut = int(len(df) * (1 - {p['test_fraction']}))\ntrain, test = df.iloc[:cut], df.iloc[cut:]\n"
        )
    strat = f", stratify=df[{roles['strata']!r}]" if roles.get("strata") else ""
    return (
        "from sklearn.model_selection import train_test_split\n\n"
        + load_line(dataset)
        + f"train, test = train_test_split(df, test_size={p['test_fraction']}, random_state={int(p['seed'])}{strat})\n"
    )


# ============================================================================ select / rename columns
def _select_columns(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cols = roles["cols"]
    after = frame[cols].copy() if p["mode"] == "keep" else frame.drop(columns=cols)
    if after.shape[1] == 0:
        raise InvalidInputError("No puede quitar todas las columnas.")
    renames: dict[str, str] = {}
    for number, line in enumerate((p.get("rename") or "").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.replace("→", "->").split("->")]
        if len(parts) != 2 or not all(parts):
            raise InvalidInputError(f"Renombrar, línea {number}: escriba «nombre actual -> nombre nuevo».")
        old, new = parts
        if old not in after.columns:
            raise InvalidInputError(f"Renombrar: «{old}» no está entre las columnas que quedan.")
        if len(new) > MAX_NAME:
            raise InvalidInputError(f"Los nombres admiten como máximo {MAX_NAME} caracteres.")
        renames[old] = new
    if renames:
        final = [renames.get(c, c) for c in after.columns]
        if len({str(c).lower() for c in final}) != len(final):
            raise InvalidInputError("Tras renombrar, dos columnas tendrían el mismo nombre.")
        after = after.rename(columns=renames)
    out = transformed("Seleccionar, renombrar y ordenar columnas", f"{after.shape[1]} columnas", frame, after)
    out.table(
        "Columnas del resultado",
        ["#", "Columna", "Antes"],
        [[i + 1, c, next((o for o, n in renames.items() if n == c), c)] for i, c in enumerate(after.columns)],
    )
    return out.build()


def _select_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    cols = roles["cols"]
    body = f"df = df[{cols!r}]\n" if p["mode"] == "keep" else f"df = df.drop(columns={cols!r})\n"
    renames = {}
    for line in (p.get("rename") or "").splitlines():
        if "->" in line:
            old, new = (part.strip() for part in line.replace("→", "->").split("->", 1))
            renames[old] = new
    return load_line(dataset) + body + (f"df = df.rename(columns={renames!r})\n" if renames else "") + _SAVE


# ============================================================================ aggregate
_AGG = (
    ("count", "Cantidad de datos"), ("mean", "Media"), ("median", "Mediana"), ("sum", "Suma"),
    ("min", "Mínimo"), ("max", "Máximo"), ("std", "Desv. estándar"), ("var", "Varianza"),
    ("q1", "Primer cuartil"), ("q3", "Tercer cuartil"), ("nunique", "Valores distintos"),
    ("first", "Primer valor"), ("last", "Último valor"),
)  # fmt: skip
_AGG_FUNCTIONS: dict[str, Any] = {
    "q1": lambda s: s.quantile(0.25),
    "q3": lambda s: s.quantile(0.75),
}


def _aggregate(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    by, values, functions = roles["by"], roles["values"], p["functions"]
    if set(by) & set(values):
        raise InvalidInputError("Una columna no puede ser a la vez de agrupación y de valores.")
    if not functions:
        raise InvalidInputError("Elija al menos una función.")
    data = frame[by].copy()
    for value in values:
        data[value] = pd.to_numeric(frame[value], errors="coerce")
    grouped = data.groupby(by, dropna=p["dropna"], observed=True, sort=True)
    specs = {f"{v}_{f}": pd.NamedAgg(column=v, aggfunc=_AGG_FUNCTIONS.get(f, f)) for v in values for f in functions}
    result = grouped.agg(**specs)
    if p["include_n"]:
        result.insert(0, "n", grouped.size())
    after = result.reset_index()
    out = transformed(
        "Agregar por grupos", f"por {', '.join(by)} · {len(after)} grupo(s)", frame, after, created=list(result.columns)
    )
    out.tiles([("Grupos", len(after)), ("Filas de origen", len(frame))])
    preview(out, after, list(after.columns), "Resultado", rows=12)
    return out.set(groups=len(after)).build()


def _aggregate_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    named = ", ".join(f"{v}_{f}=({v!r}, {_AGG_CODE.get(f, repr(f))})" for v in roles["values"] for f in p["functions"])
    return (
        load_line(dataset)
        + f"df = df.groupby({roles['by']!r}, dropna={p['dropna']}).agg({named}).reset_index()\n"
        + _SAVE
    )


_AGG_CODE = {"q1": "lambda s: s.quantile(0.25)", "q3": "lambda s: s.quantile(0.75)"}


# ============================================================================ reshape
def _melt(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    ids, values = roles.get("ids", []), roles["values"]
    var_name, value_name = (p["var_name"] or "variable").strip(), (p["value_name"] or "valor").strip()
    if set(ids) & set(values):
        raise InvalidInputError("Una columna no puede ser a la vez de identificación y de valores.")
    if var_name.lower() == value_name.lower() or {var_name.lower(), value_name.lower()} & {c.lower() for c in ids}:
        raise InvalidInputError(
            "Los nombres de las dos columnas nuevas deben ser distintos entre sí y de las columnas de identificación."
        )
    if len({classify_column(frame[v]) for v in values}) > 1:
        raise InvalidInputError("Las columnas a apilar deben ser todas del mismo tipo (números, texto o fechas).")
    if len(frame) * len(values) > MAX_MELT_ROWS:
        raise InvalidInputError(f"El resultado tendría más de {MAX_MELT_ROWS:,} filas.".replace(",", " "))
    after = pd.melt(frame, id_vars=ids, value_vars=values, var_name=var_name, value_name=value_name)
    out = transformed(
        "Apilar columnas (ancho → largo)",
        f"{len(values)} columnas → «{var_name}» y «{value_name}»",
        frame,
        after,
        created=[var_name, value_name],
    )
    preview(out, after, list(after.columns))
    return out.build()


def _melt_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    ids = roles.get("ids", [])
    return (
        load_line(dataset)
        + f"df = pd.melt(df, id_vars={ids!r}, value_vars={roles['values']!r}, var_name={(p['var_name'] or 'variable')!r}, value_name={(p['value_name'] or 'valor')!r})\n"
        + _SAVE
    )


def _pivot(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    index, columns, value = roles["index"], roles["columns"], roles["values"]
    if columns in index or value in index or columns == value:
        raise InvalidInputError("Las columnas de índice, de categorías y de valores deben ser distintas.")
    data = frame[[*index, columns]].copy()
    data[value] = pd.to_numeric(frame[value], errors="coerce")
    levels = frame[columns].nunique(dropna=True)
    if levels > MAX_NEW_COLUMNS:
        raise InvalidInputError(
            f"«{columns}» tiene {levels} categorías: se crearían demasiadas columnas (máximo {MAX_NEW_COLUMNS})."
        )
    table = pd.pivot_table(data, index=index, columns=columns, values=value, aggfunc=p["aggfunc"], observed=True)
    table.columns = [str(c) for c in table.columns]
    clash = {str(c).lower() for c in index} & {c.lower() for c in table.columns}
    if clash:
        raise InvalidInputError(
            f"Una categoría de «{columns}» se llama igual que una columna del índice ({', '.join(sorted(clash))})."
        )
    after = table.reset_index()
    out = transformed("Pivotar (largo → ancho)", f"{value} por {columns}", frame, after, created=list(table.columns))
    duplicated = int(data.duplicated([*index, columns]).sum())
    if duplicated:
        out.alert(
            "info",
            f"Había {duplicated} filas con la misma combinación de índice y categoría: se resumieron con la función «{p['aggfunc']}».",
        )
    preview(out, after, list(after.columns))
    return out.build()


def _pivot_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        load_line(dataset)
        + f"df = df.pivot_table(index={roles['index']!r}, columns={roles['columns']!r}, values={roles['values']!r}, aggfunc={p['aggfunc']!r}).reset_index()\n"
        + _SAVE
    )


# ============================================================================ combine two datasets
_HOW = (
    ("inner", "Solo las filas con pareja en los dos (intersección)"),
    ("left", "Todas las de este dataset, con lo que coincida del otro"),
    ("right", "Todas las del otro dataset, con lo que coincida de este"),
    ("outer", "Todas las filas de los dos (unión)"),
)
_RELATIONS = (
    ("m:1", "Muchas filas de este dataset por cada fila del otro (consulta de una tabla de referencia)"),
    ("1:1", "Una a una"),
    ("1:m", "Una fila de este dataset con varias del otro"),
    ("m:m", "Sin comprobar (puede multiplicar filas)"),
)
_ORIGIN = {"left_only": "solo en este dataset", "right_only": "solo en el otro", "both": "en ambos"}
_KEY_KINDS = {"numeric": "number", "boolean": "number", "categorical": "text", "text": "text", "datetime": "datetime"}


def _other_frame(p: dict[str, Any]) -> pd.DataFrame:
    other = p.get("other")
    if not isinstance(other, pd.DataFrame):
        raise InvalidInputError("Elija el otro dataset.")
    return other


def _merge(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    other = _other_frame(p)
    left_keys, right_keys = roles["keys"], roles["other_keys"]
    if len(left_keys) != len(right_keys):
        raise InvalidInputError(
            "Elija la misma cantidad de columnas clave en los dos datasets (se emparejan en orden)."
        )
    for left, right in zip(left_keys, right_keys, strict=True):
        if _KEY_KINDS[classify_column(frame[left])] != _KEY_KINDS[classify_column(other[right])]:
            raise InvalidInputError(
                f"«{left}» y «{right}» son de tipos distintos (número, texto o fecha): no se pueden emparejar."
            )
    suffix = (p["suffix"] or "_otro").strip()
    same_key = {r for left, r in zip(left_keys, right_keys, strict=True) if left == r}
    clash = [c for c in other.columns if c in frame.columns and c not in same_key]
    right = other.rename(columns={c: f"{c}{suffix}" for c in clash})
    right_keys_now = [f"{k}{suffix}" if k in clash else k for k in right_keys]
    try:
        merged = frame.merge(
            right, how=p["how"], left_on=left_keys, right_on=right_keys_now, validate=p["relation"], indicator="_origen"
        )
    except pd.errors.MergeError:
        raise InvalidInputError(
            f"Los datos no cumplen la relación elegida ({dict(_RELATIONS)[p['relation']].split(' (')[0].lower()}): "
            "hay claves repetidas donde no debería. Revise las claves o elija «Sin comprobar»."
        ) from None
    origin = merged.pop("_origen").astype(str)
    for left, right_name in zip(left_keys, right_keys_now, strict=True):
        if left != right_name:  # keys named differently: keep the left one, completed from the right where it is empty
            merged[left] = merged[left].where(merged[left].notna(), merged[right_name])
            merged = merged.drop(columns=right_name)
    if p["keep_origin"]:
        merged[column_name(merged, "origen")] = origin.map(_ORIGIN)
    after = merged.reset_index(drop=True)
    if after.empty:
        raise InvalidInputError(
            "Ninguna fila tiene pareja con esas claves: revise que se escriban igual en los dos datasets."
        )
    created = [c for c in after.columns if c not in frame.columns]
    out = transformed(
        "Unir datasets (merge)", f"{', '.join(left_keys)} ↔ {', '.join(right_keys)}", frame, after, created=created
    )
    counts = origin.value_counts()
    out.tiles(
        [("Filas de este dataset", len(frame)), ("Filas del otro", len(other)), ("Filas del resultado", len(after))]
    )
    out.table(
        "Coincidencias entre las claves",
        ["Origen de la fila", "Filas"],
        [[label, int(counts.get(key, 0))] for key, label in _ORIGIN.items()],
    )
    duplicated = int(other.duplicated(right_keys).sum())
    if duplicated:
        out.alert(
            "warning",
            f"El otro dataset repite {duplicated} vez/veces una clave: cada coincidencia genera una fila por cada copia.",
        )
    if clash:
        out.alert(
            "info",
            f"Las columnas {', '.join(clash)} existen en los dos datasets: las del otro se renombraron con «{suffix}».",
        )
    preview(out, after, [*left_keys, *created[:8]])
    return out.set(
        matched=int(counts.get("both", 0)),
        left_only=int(counts.get("left_only", 0)),
        right_only=int(counts.get("right_only", 0)),
    ).build()


def _merge_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        load_line(dataset)
        + f"other = sigma.load({p.get('other')!r})\n"
        + f"df = df.merge(other, how={p['how']!r}, left_on={roles['keys']!r}, right_on={roles['other_keys']!r}, "
        + f"suffixes=('', {(p['suffix'] or '_otro')!r}), validate={p['relation']!r})\n"
        + _SAVE
    )


def _append(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    other = _other_frame(p)
    common = [c for c in frame.columns if c in other.columns]
    if not common:
        raise InvalidInputError("Los dos datasets no tienen ninguna columna con el mismo nombre.")
    mismatched = [c for c in common if _KEY_KINDS[classify_column(frame[c])] != _KEY_KINDS[classify_column(other[c])]]
    if mismatched:
        raise InvalidInputError(
            f"Estas columnas son de tipos distintos en los dos datasets: {', '.join(mismatched[:6])}."
        )
    join = "inner" if p["mode"] == "common" else "outer"
    parts = [frame, other]
    created: list[str] = []
    if p["source"]:
        name = column_name(frame, p["source_name"] or "origen", what="la columna de origen")
        if name in other.columns:
            raise InvalidInputError(f"El otro dataset ya tiene una columna «{name}»: elija otro nombre.")
        parts = [frame.assign(**{name: p["label_main"]}), other.assign(**{name: p["label_other"]})]
        created.append(name)
    after = pd.concat(parts, ignore_index=True, join=join, sort=False)
    only_here = [c for c in frame.columns if c not in other.columns]
    only_there = [c for c in other.columns if c not in frame.columns]
    out = transformed(
        "Apilar datasets (añadir filas)",
        "solo las columnas comunes" if join == "inner" else "todas las columnas",
        frame,
        after,
        created=created,
    )
    out.tiles(
        [("Filas de este dataset", len(frame)), ("Filas añadidas", len(other)), ("Columnas comunes", len(common))]
    )
    if only_here or only_there:
        out.alert(
            "info",
            f"Solo en este dataset: {', '.join(only_here) or '—'} · solo en el otro: {', '.join(only_there) or '—'} "
            + ("(se descartan)." if join == "inner" else "(quedan vacías donde no existen)."),
        )
    preview(out, after, list(after.columns)[:10], "Vista previa", rows=6)
    return out.set(rows=len(after)).build()


def _append_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    join = "inner" if p["mode"] == "common" else "outer"
    return (
        "import pandas as pd\n\n"
        + load_line(dataset)
        + f"other = sigma.load({p.get('other')!r})\n"
        + f"df = pd.concat([df, other], ignore_index=True, join={join!r})\n"
        + _SAVE
    )


# ============================================================================ the catalogue
_REPLACEABLE = {"module": "transformar", "outputs": (RESULT,), "can_replace": True}
_DESCENDING = FieldSpec("descending", "Orden descendente", "boolean", False)

SPECS = (
    AnalysisSpec(
        "filter",
        "Filtrar filas",
        ROWS,
        compute=_filter,
        description="Conserva (o quita) las filas que cumplen una condición escrita como fórmula.",
        params=(
            FieldSpec(
                "condition",
                "Condición",
                "code",
                help='Ejemplos: x1 > 10   ·   g == "A" and x2 < 5   ·   notna(m)   ·   year(t) >= 2005',
            ),
            FieldSpec(
                "action",
                "Qué hacer",
                "select",
                "keep",
                (("keep", "Conservar las filas que la cumplen"), ("drop", "Quitar las filas que la cumplen")),
            ),
        ),
        reference=REFERENCE,
        code=_filter_code,
        example={"roles": {}, "params": {"condition": 'x1 > 8 and g != "C"'}},
        **_REPLACEABLE,
    ),
    AnalysisSpec(
        "sort",
        "Ordenar filas",
        ROWS,
        compute=_sort,
        description="Ordena el dataset por una o varias columnas (orden estable).",
        roles=(RoleSpec("cols", "Ordenar por (en este orden)", "any", multiple=True),),
        params=(
            _DESCENDING,
            FieldSpec("missing", "Faltantes", "select", "last", (("last", "Al final"), ("first", "Al principio"))),
        ),
        code=_sort_code,
        example={"roles": {"cols": ["g", "x1"]}, "params": {}},
        **_REPLACEABLE,
    ),
    AnalysisSpec(
        "duplicates",
        "Detectar y quitar duplicados",
        ROWS,
        compute=_duplicates,
        description="Encuentra filas repetidas (en todas las columnas o solo en algunas) y las quita o las marca.",
        roles=(
            RoleSpec(
                "cols",
                "Columnas que identifican un duplicado (opcional)",
                "any",
                multiple=True,
                optional=True,
                help="Si no elige ninguna se comparan todas.",
            ),
        ),
        params=(
            FieldSpec(
                "keep",
                "Qué copia se conserva",
                "select",
                "first",
                (("first", "La primera"), ("last", "La última"), ("none", "Ninguna (marcar todas las copias)")),
            ),
            FieldSpec(
                "action",
                "Qué hacer",
                "select",
                "remove",
                (("remove", "Quitar las filas marcadas"), ("flag", "Añadir una columna «duplicado» (1/0)")),
            ),
        ),
        code=_duplicates_code,
        example={"roles": {"cols": ["g", "h"]}, "params": {"action": "flag"}},
        **_REPLACEABLE,
    ),
    AnalysisSpec(
        "sample",
        "Muestra aleatoria",
        ROWS,
        compute=_sample,
        description="Extrae una muestra de filas, simple o estratificada proporcionalmente, con semilla para repetirla.",
        roles=(RoleSpec("strata", "Estratos (opcional)", "categorical", optional=True),),
        params=(
            FieldSpec(
                "mode",
                "El tamaño se indica como",
                "select",
                "n",
                (("n", "Número de filas"), ("fraction", "Fracción de las filas")),
            ),
            FieldSpec("size", "Tamaño", "number", 100, minimum=0.0001),
            FieldSpec("replace", "Con reemplazo (bootstrap)", "boolean", False),
            FieldSpec("seed", "Semilla", "integer", 123, minimum=0, maximum=2**31 - 1),
        ),
        code=_sample_code,
        example={"roles": {"strata": "g"}, "params": {"size": 60}},
        **{**_REPLACEABLE, "outputs": (OutputSpec(RESULT_KEY, "Muestra", required=True),)},
    ),
    AnalysisSpec(
        "partition",
        "Partición entrenamiento / prueba",
        ROWS,
        compute=_partition,
        description="Divide el dataset en entrenamiento, prueba (y validación), al azar —con estratos si quiere— o por orden temporal.",
        module="transformar",
        roles=(
            RoleSpec(
                "strata",
                "Estratos (opcional)",
                "categorical",
                optional=True,
                help="Mantiene la proporción de cada categoría en los conjuntos.",
            ),
            RoleSpec("time", "Ordenar por fecha (solo partición secuencial)", "datetime", optional=True),
        ),
        params=(
            FieldSpec(
                "method",
                "Método",
                "select",
                "random",
                (("random", "Aleatoria"), ("sequential", "Secuencial (lo último es prueba; series de tiempo)")),
            ),
            FieldSpec("test_fraction", "Fracción de prueba", "number", 0.3, minimum=0.02, maximum=0.9),
            FieldSpec(
                "validation_fraction",
                "Fracción de validación (0 = sin validación)",
                "number",
                0.0,
                minimum=0,
                maximum=0.9,
            ),
            FieldSpec(
                "seed", "Semilla", "integer", 123, minimum=0, maximum=2**31 - 1, visible_if=("method", ("random",))
            ),
        ),
        outputs=(
            OutputSpec("train", "Entrenamiento", required=True),
            OutputSpec("test", "Prueba", required=True),
            OutputSpec("validation", "Validación"),
        ),
        code=_partition_code,
        example={"roles": {"strata": "g"}, "params": {"validation_fraction": 0.1}},
    ),
    AnalysisSpec(
        "select_columns",
        "Seleccionar, renombrar y ordenar columnas",
        STRUCTURE,
        compute=_select_columns,
        description="Conserva o quita columnas (el orden en que las marque es el orden del resultado) y cambia sus nombres.",
        roles=(RoleSpec("cols", "Columnas", "any", multiple=True),),
        params=(
            FieldSpec(
                "mode",
                "Qué hacer con las marcadas",
                "select",
                "keep",
                (("keep", "Conservar solo esas"), ("drop", "Quitarlas")),
            ),
            FieldSpec(
                "rename",
                "Renombrar (opcional)",
                "code",
                optional=True,
                help="Una por línea: «nombre actual -> nombre nuevo».",
            ),
        ),
        code=_select_code,
        example={"roles": {"cols": ["t", "y", "x1"]}, "params": {"rename": "y -> respuesta"}},
        **_REPLACEABLE,
    ),
    AnalysisSpec(
        "aggregate",
        "Agregar por grupos (resumen)",
        STRUCTURE,
        compute=_aggregate,
        description="Resume columnas numéricas por grupos (media, suma, cuartiles…) y entrega una tabla con una fila por grupo.",
        module="transformar",
        roles=(
            RoleSpec("by", "Agrupar por", "any", multiple=True),
            RoleSpec("values", "Columnas a resumir", "numeric", multiple=True),
        ),
        params=(
            FieldSpec("functions", "Funciones", "multiselect", ["count", "mean"], _AGG),
            FieldSpec("include_n", "Añadir el tamaño del grupo (n)", "boolean", True),
            FieldSpec("dropna", "Omitir el grupo de valores faltantes", "boolean", True),
        ),
        outputs=(OutputSpec(RESULT_KEY, "Tabla resumen", required=True),),
        code=_aggregate_code,
        example={"roles": {"by": ["g", "h"], "values": ["y", "x1"]}, "params": {"functions": ["mean", "std", "q1"]}},
    ),
    AnalysisSpec(
        "melt",
        "Apilar columnas (ancho → largo)",
        STRUCTURE,
        compute=_melt,
        description="Convierte varias columnas de medidas en dos columnas (nombre y valor) para análisis de medidas repetidas o gráficos por grupo.",
        module="transformar",
        roles=(
            RoleSpec("ids", "Columnas de identificación (se repiten)", "any", multiple=True, optional=True),
            RoleSpec("values", "Columnas a apilar", "any", multiple=True, min_count=2),
        ),
        params=(
            FieldSpec("var_name", "Nombre de la columna con los nombres", "text", "variable"),
            FieldSpec("value_name", "Nombre de la columna con los valores", "text", "valor"),
        ),
        outputs=(OutputSpec(RESULT_KEY, "Datos apilados", required=True),),
        code=_melt_code,
        example={"roles": {"ids": ["g"], "values": ["x1", "x2"]}, "params": {}},
    ),
    AnalysisSpec(
        "pivot",
        "Pivotar (largo → ancho)",
        STRUCTURE,
        compute=_pivot,
        description="Convierte las categorías de una columna en columnas, resumiendo los valores repetidos.",
        module="transformar",
        roles=(
            RoleSpec("index", "Filas del resultado (índice)", "any", multiple=True),
            RoleSpec("columns", "Categorías que pasan a columnas", "categorical"),
            RoleSpec("values", "Valores", "numeric"),
        ),
        params=(
            FieldSpec(
                "aggfunc",
                "Si hay varios valores",
                "select",
                "mean",
                (
                    ("mean", "Media"),
                    ("sum", "Suma"),
                    ("median", "Mediana"),
                    ("min", "Mínimo"),
                    ("max", "Máximo"),
                    ("count", "Cantidad"),
                    ("first", "Primero"),
                ),
            ),
        ),
        outputs=(OutputSpec(RESULT_KEY, "Datos pivotados", required=True),),
        code=_pivot_code,
        example={"roles": {"index": ["h"], "columns": "g", "values": "y"}, "params": {}},
    ),
    AnalysisSpec(
        "merge",
        "Unir con otro dataset (merge)",
        STRUCTURE,
        compute=_merge,
        description=(
            "Añade las columnas de otro dataset emparejando filas por una o varias claves "
            "(como BUSCARV o «Añadir variables» de SPSS)."
        ),
        module="transformar",
        roles=(
            RoleSpec("keys", "Clave(s) en este dataset", "any", multiple=True),
            RoleSpec(
                "other_keys",
                "Clave(s) en el otro dataset (en el mismo orden)",
                "any",
                multiple=True,
                dataset_param="other",
            ),
        ),
        params=(
            FieldSpec("other", "El otro dataset", "dataset"),
            FieldSpec("how", "Qué filas conservar", "select", "left", _HOW),
            FieldSpec(
                "relation",
                "Relación esperada entre las filas",
                "select",
                "m:1",
                _RELATIONS,
                help="Protege de multiplicar filas por error: si la relación no se cumple, no se hace la unión.",
            ),
            FieldSpec("suffix", "Añadir a los nombres repetidos del otro dataset", "text", "_otro"),
            FieldSpec("keep_origin", "Añadir una columna «origen» (de dónde viene cada fila)", "boolean", False),
        ),
        outputs=(OutputSpec(RESULT_KEY, "Datos unidos", required=True),),
        code=_merge_code,
        example={
            "roles": {"keys": ["g"], "other_keys": ["g"]},
            "params": {"other": "@lookup", "how": "outer", "keep_origin": True},
        },
    ),
    AnalysisSpec(
        "append",
        "Apilar otro dataset (añadir filas)",
        STRUCTURE,
        compute=_append,
        description="Pone las filas de otro dataset debajo de las de este (columnas con el mismo nombre y tipo).",
        module="transformar",
        params=(
            FieldSpec("other", "El otro dataset", "dataset"),
            FieldSpec(
                "mode",
                "Columnas del resultado",
                "select",
                "union",
                (("union", "Todas (las que falten quedan vacías)"), ("common", "Solo las comunes")),
            ),
            FieldSpec("source", "Añadir una columna que indique de dónde viene cada fila", "boolean", True),
            FieldSpec("source_name", "Nombre de esa columna", "text", "origen", visible_if=("source", ("True",))),
            FieldSpec(
                "label_main", "Valor para las filas de este dataset", "text", "actual", visible_if=("source", ("True",))
            ),
            FieldSpec(
                "label_other", "Valor para las filas del otro", "text", "añadido", visible_if=("source", ("True",))
            ),
        ),
        outputs=(OutputSpec(RESULT_KEY, "Datos apilados", required=True),),
        code=_append_code,
        example={"roles": {}, "params": {"other": "@more", "mode": "common"}},
    ),
)

registry.register(*SPECS)
