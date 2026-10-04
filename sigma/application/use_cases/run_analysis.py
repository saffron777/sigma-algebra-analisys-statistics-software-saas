"""Use cases of the analysis catalogue: list it, run an analysis, show the Python that reproduces one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sigma.application.dto import AnalysisCommand
from sigma.application.ports import AnalysisRunner, DatasetRepository
from sigma.application.use_cases.datasets import clean_dataset_name
from sigma.domain.analysis import registry
from sigma.domain.analysis.modules import BY_KEY, ModuleDef
from sigma.domain.analysis.output import AnalysisOutput
from sigma.domain.analysis.spec import AnalysisSpec, dataset_params, prepare_inputs
from sigma.domain.datasets import Dataset, DatasetInfo
from sigma.domain.errors import InvalidInputError


@dataclass
class AnalysisRun:
    output: AnalysisOutput
    spec: AnalysisSpec
    dataset_name: str | None
    created: list[DatasetInfo] = field(default_factory=list)
    roles: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


class ListAnalyses:
    """The catalogue of one module (what its page builds the form from)."""

    def execute(self, module: str) -> tuple[ModuleDef, list[AnalysisSpec]]:
        if module not in BY_KEY:
            raise InvalidInputError(f"Módulo de análisis desconocido «{module}».")
        return BY_KEY[module], registry.catalog(module)


class RunAnalysis:
    def __init__(self, datasets: DatasetRepository, runner: AnalysisRunner) -> None:
        self._datasets = datasets
        self._runner = runner

    def execute(self, cmd: AnalysisCommand) -> AnalysisRun:
        spec = registry.get(cmd.module, cmd.analysis_id)
        info = None
        frame: pd.DataFrame | None = None
        if spec.needs_dataset:
            if not cmd.dataset_id:
                raise InvalidInputError("Elija un dataset.")
            dataset = self._datasets.get(cmd.dataset_id)
            info, frame = dataset.info, dataset.data
        others = self._other_frames(spec, cmd.params)
        roles, params = prepare_inputs(spec, frame, cmd.roles, cmd.params, {n: d.data for n, d in others.items()})
        self._check_outputs(spec, cmd)
        data = frame if frame is not None else pd.DataFrame()
        # compute gets the other datasets' data where the request had their ids
        output = self._runner.run(
            spec.module, spec.id, data, roles, {**params, **{n: d.data for n, d in others.items()}}
        )
        created = self._save_datasets(spec, cmd, output, info)
        return AnalysisRun(output, spec, info.name if info else None, created, roles, params)

    def _other_frames(self, spec: AnalysisSpec, params: dict[str, Any]) -> dict[str, Dataset]:
        """The other datasets an analysis was pointed at (it is an error to name one that is not the account's)."""
        found: dict[str, Dataset] = {}
        for name in dataset_params(spec):
            identifier = str((params or {}).get(name) or "").strip()
            if identifier:
                found[name] = self._datasets.get(identifier)
        return found

    # ------------------------------------------------------------------ datasets left behind by the analysis
    @staticmethod
    def _check_outputs(spec: AnalysisSpec, cmd: AnalysisCommand) -> None:
        known = {o.key for o in spec.outputs}
        for key, name in cmd.save_as.items():
            if key not in known:
                raise InvalidInputError(f"Este análisis no produce «{key}».")
            if not (name or "").strip():
                raise InvalidInputError("Indique un nombre para el dataset nuevo.")
        if cmd.replace and not spec.can_replace:
            raise InvalidInputError("Este análisis no puede reemplazar el dataset original.")
        for out in spec.outputs:
            if out.required and not cmd.replace and not (cmd.save_as.get(out.key) or "").strip():
                raise InvalidInputError(f"Indique el nombre del dataset nuevo para «{out.label}».")

    def _save_datasets(
        self, spec: AnalysisSpec, cmd: AnalysisCommand, output: AnalysisOutput, source: DatasetInfo | None
    ) -> list[DatasetInfo]:
        created: list[DatasetInfo] = []
        meta = {"derived_from": source.id if source else None, "analysis": f"{spec.module}/{spec.id}"}
        if cmd.replace and spec.outputs:
            offer = output.datasets.get(spec.outputs[0].key)
            if offer is None or source is None:
                raise InvalidInputError("El análisis no produjo una tabla para reemplazar el dataset.")
            created.append(self._datasets.replace_data(source.id, offer.frame))
            return created
        for key, name in cmd.save_as.items():
            offer = output.datasets.get(key)
            if offer is None:
                raise InvalidInputError(f"El análisis no produjo «{key}».")
            created.append(self._datasets.add(clean_dataset_name(name), offer.frame, "derived", meta))
        return created


class GetAnalysisCode:
    """The Python that reproduces an analysis with the same columns and parameters (shown, never executed)."""

    def __init__(self, datasets: DatasetRepository) -> None:
        self._datasets = datasets

    def execute(self, module: str, analysis_id: str, dataset_id: str | None, roles: dict, params: dict) -> str:
        spec = registry.get(module, analysis_id)
        if spec.code is None:
            raise InvalidInputError("Este análisis todavía no tiene código Python equivalente.")
        frame = None
        name = "mis_datos"
        if spec.needs_dataset and dataset_id:
            dataset = self._datasets.get(dataset_id)
            frame, name = dataset.data, dataset.info.name
        others = {n: self._datasets.get(str(params[n])) for n in dataset_params(spec) if (params or {}).get(n)}
        clean_roles, clean_params = prepare_inputs(spec, frame, roles, params, {n: d.data for n, d in others.items()})
        for parameter, other in others.items():
            clean_params[parameter] = other.info.name  # the code refers to datasets by name, like sigma.load()
        return spec.code(clean_roles, clean_params, name)
