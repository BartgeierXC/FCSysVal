import operator as _operator
import os
import yaml
import datetime
import pathlib
from pydantic import BaseModel, PositiveInt, FilePath
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


class SimulationConfig(BaseModel):
    config_path: FilePath
    input_path: Path


class GroupConfig(BaseModel):
    group_time: PositiveInt = None
    steps: PositiveInt = None
    offset: int = None
    trigger_steps: int = None
    trigger: str = None
    tolerance: float = 0.1
    conditions_file: Optional[str] = None  # Table of the passes of each operating condition


class DataSourceConfig(BaseModel):
    # DB source fields
    table: Optional[str] = None
    database: Optional[str] = None
    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None
    conditions: list[str] = []

    # CSV source fields
    csv_path: Optional[Path] = None
    csv_sep: str = ','
    hold_time: Optional[float] = None
    time_axis: Optional[str] = None  # 'elapsed': seconds since first sample
    skip_time: float = 0  # Warm-up excluded from the comparison

    input_variables: list[str] = []
    output_variables: list[str] = []
    input_factors: Optional[list[float]] = None
    output_factors: Optional[list[float]] = None
    input_offset: Optional[list[float]] = None
    output_offset: Optional[list[float]] = None


def _is_box_spec(spec: object) -> bool:
    """Return True if *spec* is a flat ``[lo, hi]`` box bound (not a nested list)."""
    return (
        isinstance(spec, (list, tuple))
        and len(spec) == 2
        and not isinstance(spec[0], (list, tuple))
    )


class RuleSpec(BaseModel):
    """A single relational constraint between two DataFrame columns (or a scalar).

    Examples (YAML)::

        rules:
          - lhs: "p.Si.A [bara]"
            op: ">"
            rhs: "p.So.A [bara]"
          - lhs: "I.S.set [A]"
            op: ">="
            rhs: 20.0
    """

    lhs: str
    op: str
    rhs: Union[str, float]

    def to_callable(self):
        """Return a ``lambda df: ...`` boolean mask for use with *fltr_rules* / *generate_lhs*."""
        _ops = {
            ">": _operator.gt,
            "<": _operator.lt,
            ">=": _operator.ge,
            "<=": _operator.le,
            "==": _operator.eq,
            "!=": _operator.ne,
        }
        if self.op not in _ops:
            raise ValueError(
                f"Unsupported operator '{self.op}'. Use one of {list(_ops)}."
            )
        op_fn = _ops[self.op]
        lhs, rhs = self.lhs, self.rhs
        if isinstance(rhs, str):
            return lambda df: op_fn(df[lhs], df[rhs])
        return lambda df: op_fn(df[lhs], rhs)


class ValConfig(BaseModel):
    """Validation domain configuration: operating bounds and relational rules.

    Bounds
    ------
    Each key is a column name (input variable).  Two formats are supported:

    * **Box bound** ``[lo, hi]`` – the variable is constrained to a fixed
      scalar interval.
    * **Dependent bound** ``[[ref1, lo1, hi1], [ref2, lo2, hi2], ...]`` – the
      feasible range is a piecewise-linear function of the single independent
      (box) variable.  Exactly one box bound must be present when dependent
      bounds are used.

    Rules
    -----
    A list of :class:`RuleSpec` entries that encode relational constraints
    between columns (e.g. inlet pressure > outlet pressure).

    Use :meth:`get_bounds` and :meth:`get_rules` to obtain the objects
    expected by ``fltr_bounds``, ``fltr_rules``, and ``generate_lhs`` from
    ``src.utils.val_doe``.

    Example (YAML)::

        val_config:
          bounds:
            "I.S.set [A]": [20.0, 600.0]
            "p.So.A [bara]":
              - [20,  1.0, 1.2]
              - [200, 1.1, 1.4]
              - [600, 1.2, 1.6]
          rules:
            - lhs: "p.Si.A [bara]"
              op:  ">"
              rhs: "p.So.A [bara]"
    """

    bounds: dict[str, Any] = {}
    rules: list[RuleSpec] = []

    def get_bounds(self) -> dict[str, object]:
        """Return a bounds dict ready for ``fltr_bounds`` / ``generate_lhs``.

        Box specs become ``(lo, hi)`` tuples; dependent specs become
        ``numpy.ndarray`` of shape ``(m, 3)``.
        """
        result: dict[str, object] = {}
        for col, spec in self.bounds.items():
            if _is_box_spec(spec):
                result[col] = (float(spec[0]), float(spec[1]))
            else:
                result[col] = np.array(spec, dtype=float)
        return result

    def get_rules(self) -> list:
        """Return a list of callables ready for ``fltr_rules`` / ``generate_lhs``."""
        return [r.to_callable() for r in self.rules]


class ValidationConfig(BaseModel):
    sim_config: SimulationConfig
    group_config: Optional[GroupConfig] = None
    data_source: DataSourceConfig
    val_config: Optional[ValConfig] = None

    @classmethod
    def from_yaml(cls, config_file: str):
        config_file_path = pathlib.Path(config_file)
        if not config_file_path.is_absolute():
            config_file_path = pathlib.Path(os.getcwd()) / config_file

        config_data = parse_config(str(config_file_path))
        if not config_data:
            return None

        sim_cfg = config_data.get("sim_config", {})
        raw_sim_path = sim_cfg.get("config_path")
        if raw_sim_path:
            sim_path = pathlib.Path(raw_sim_path)
            if not sim_path.is_absolute():
                sim_cfg["config_path"] = str((config_file_path.parent / sim_path).resolve())

        # raw_input_path = sim_cfg.get("input_path")
        # if raw_input_path:
        #     input_path = pathlib.Path(raw_input_path)
        #     if not input_path.is_absolute():
        #         sim_cfg["input_path"] = str((pathlib.Path(os.getcwd()) / input_path).resolve())

        cfg = cls(**config_data)

        # Resolve relative CSV path with workspace-first search, then config-relative fallbacks.
        if cfg.data_source.csv_path is not None and not cfg.data_source.csv_path.is_absolute():
            rel_path = cfg.data_source.csv_path
            cwd_path = pathlib.Path(os.getcwd())
            search_roots = [cwd_path, config_file_path.parent, *config_file_path.parent.parents]
            resolved_path = None
            for root in search_roots:
                candidate = (root / rel_path)
                if candidate.exists():
                    resolved_path = candidate.resolve()
                    break
            cfg.data_source.csv_path = resolved_path or (cwd_path / rel_path).resolve()

        return cfg


def parse_config(config_file: str = 'simulation_config.yml'):
    config_file_path = pathlib.Path(config_file)
    if not config_file_path.is_absolute():
        config_file_path = pathlib.Path(os.getcwd()) / config_file
    with open(config_file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
