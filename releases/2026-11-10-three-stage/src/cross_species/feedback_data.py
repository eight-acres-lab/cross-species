"""Historical TG-GATEs condition-mean inputs for a retrospective feedback pilot.

Role assignment and filtering use metadata only. Target responses are returned for
training/calibration and final diagnostic scoring; callers must respect ``roles``
and perform feature selection and fitted preprocessing on training rows only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


CONTEXT_FEATURES = (
    "time_h", "rat_dose_value", "human_dose_value", "dose_Low", "dose_Middle",
    "dose_High", "is_lps",
)
LIMITATIONS = (
    "Condition-level replicate means do not retain single-cell distributions or replicate variance (Zenodo 20204514).",
    "Historical RMA jointly normalized all available arrays per species before splitting; this pilot remains exploratory and retrospective (source 41a_rma_brainarray.R).",
    "Historical test outcomes have been examined in prior repository analyses; diagnostic_test is not a prospective or untouched test set (artifacts/tg_gates_rat_human_v1/report.md).",
)
ARRAY_FIELDS = (
    "source_delta", "target_delta", "context", "context_feature_ids", "sample_ids",
    "compounds", "feature_ids", "source_feature_ids", "roles",
)


@dataclass
class FeedbackData:
    source_delta: np.ndarray
    target_delta: np.ndarray
    context: np.ndarray
    context_feature_ids: np.ndarray
    sample_ids: np.ndarray
    compounds: np.ndarray
    feature_ids: np.ndarray
    source_feature_ids: np.ndarray
    roles: np.ndarray
    metadata: list[dict]
    warnings: tuple[str, ...]
    input_hashes: dict[str, str]
    experiment_mode: str = "exploratory"
    cache_path: Path | None = None
    cache_hit: bool = False


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(required) - set(fields):
            raise ValueError(f"Missing or duplicate columns in {path}")
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed table: {path}")
    return [{k: v.strip() for k, v in row.items()} for row in rows]


def _unique(rows: list[dict], field: str, name: str) -> dict:
    values = [row[field] for row in rows]
    if any(not v for v in values) or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate {field} in {name}")
    return dict(zip(values, rows))


def _number(value: str, name: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric {name}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {name}: {value!r}")
    return number


def _matrix(path: Path, expected_ids: list[str]):
    with path.open(encoding="utf-8") as handle:
        genes = handle.readline().rstrip("\r\n").split("\t")[1:]
        if not genes or any(not gene for gene in genes) or len(genes) != len(set(genes)):
            raise ValueError(f"Invalid gene header: {path}")
        ids, values = [], []
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(genes) + 1:
                raise ValueError(f"Wrong matrix width: {path}: {fields[0]}")
            vector = np.asarray(fields[1:], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"Nonfinite matrix values: {path}: {fields[0]}")
            ids.append(fields[0])
            values.append(vector)
    if ids != expected_ids:
        raise ValueError(f"Matrix IDs/order differ from metadata: {path}")
    return genes, np.stack(values)


def _truthy(value: str | None) -> bool:
    return (value or "").casefold() in {"1", "true", "yes"}


def _infection(row: dict) -> bool:
    kind = row.get("perturbation_type", "").casefold()
    pathogen = row.get("pathogen", "").casefold()
    return (_truthy(row.get("is_infection")) or _truthy(row.get("is_pathogen"))
            or kind in {"infection", "viral infection", "bacterial infection", "pathogen"}
            or pathogen not in {"", "none", "na", "n/a", "false", "0"})


def _combination(row: dict) -> bool:
    return (_truthy(row.get("is_combination"))
            or row.get("perturbation_type", "").casefold() == "combination"
            or "+" in row["compound"])


def _compound_name(name: str) -> str:
    # TG-GATEs directory names use underscores for spaces. No chemical aliases
    # or Greek-letter transliterations are inferred (TNFa and TNFα stay distinct).
    return " ".join(name.replace("_", " ").casefold().split())


def _unit_name(unit: str | None) -> str:
    unit = (unit or "").strip().replace("µ", "μ").replace(" ", "")
    return {"uM": "μM", "ug/mL": "μg/mL"}.get(unit, unit)


def concentration_match_status(row: dict) -> str:
    """Compare nominal concentrations only when both units have known dimensions."""
    conversions = {
        "M": (1e6, "molar"), "mM": (1e3, "molar"), "μM": (1, "molar"),
        "nM": (1e-3, "molar"), "mg/mL": (1e3, "mass/volume"),
        "μg/mL": (1, "mass/volume"), "ng/mL": (1e-3, "mass/volume"),
    }
    units = [_unit_name(row.get(f"{species}_dose_unit")) for species in ["rat", "human"]]
    if any(unit not in conversions for unit in units):
        return "unknown"
    left, right = [conversions[unit] for unit in units]
    if left[1] != right[1]:
        return "incompatible"
    amounts = [_number(row[f"{species}_dose_value"], f"{species}_dose_value")
               for species in ["rat", "human"]]
    return "matched" if math.isclose(amounts[0] * left[0], amounts[1] * right[0],
                                      rel_tol=1e-9, abs_tol=1e-12) else "different"


def enrich_feedback_metadata(metadata: list[dict], attributes_path: str | Path | None = None):
    """Resolve units without expression data; return copied rows and warnings.

    The source-specific filter is in vitro / Liver / MATERIAL_ID A / Rat or
    Human. All in-vitro liver records in the supplied official TG-GATEs table
    have material code A; this code alone does not identify a cell type.
    Unit lookup requires an exact numeric dose after name/case/time formatting.
    Multiple replicate records may establish one unit; conflicting units cannot.
    """
    units_by_key = {}
    if attributes_path is not None:
        path = Path(attributes_path)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp932")
        reader = csv.DictReader(io.StringIO(text))
        required = {"SPECIES", "EXP_TEST_TYPE", "ORGAN", "MATERIAL_ID", "COMPOUND_NAME",
                    "SACRIFICE_PERIOD", "DOSE_LEVEL", "DOSE", "DOSE_UNIT"}
        if required - set(reader.fieldnames or []):
            raise ValueError(f"Missing source attribute columns: {sorted(required - set(reader.fieldnames or []))}")
        for attribute in reader:
            if None in attribute or any(v is None for v in attribute.values()):
                raise ValueError(f"Malformed source attribute row: {path}")
            species = attribute["SPECIES"].strip().casefold()
            if (species not in {"rat", "human"}
                    or attribute["EXP_TEST_TYPE"].strip().casefold() != "in vitro"
                    or attribute["ORGAN"].strip().casefold() != "liver"
                    or attribute["MATERIAL_ID"].strip().casefold() != "a"):
                continue
            period = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:hr|h|hours?)\s*",
                                  attribute["SACRIFICE_PERIOD"], re.IGNORECASE)
            if period is None:
                continue
            try:
                dose = _number(attribute["DOSE"], "attribute DOSE")
            except ValueError:
                continue
            key = (species, _compound_name(attribute["COMPOUND_NAME"]), float(period[1]),
                   attribute["DOSE_LEVEL"].strip().casefold(), dose)
            units_by_key.setdefault(key, set()).add(_unit_name(attribute["DOSE_UNIT"]))
    enriched, ambiguous = [], 0
    for row in metadata:
        row = row.copy()
        for species in ["rat", "human"]:
            key = (species, _compound_name(row["compound"]), _number(row["time_h"], "time_h"),
                   row["dose_level"].strip().casefold(), _number(row[f"{species}_dose_value"], f"{species}_dose_value"))
            declared = _unit_name(row.get(f"{species}_dose_unit"))
            found = units_by_key.get(key, set()).copy()
            if declared:
                found.add(declared)
            resolved = next(iter(found)) if len(found) == 1 and "" not in found else ""
            row[f"{species}_dose_unit"] = resolved
            row[f"{species}_dose_unit_source"] = (
                "ambiguous" if len(found) > 1 else "attributes" if resolved and key in units_by_key
                else "metadata" if resolved else "unknown")
            ambiguous += len(found) > 1
        row["concentration_match_status"] = concentration_match_status(row)
        enriched.append(row)
    warnings = []
    if ambiguous:
        warnings.append(f"Conflicting unit candidates left unresolved for {ambiguous} species-condition lookups.")
    return enriched, tuple(warnings)


def _load_cache(path: Path, hashes: dict, config: dict) -> FeedbackData:
    with np.load(path, allow_pickle=False) as payload:
        if json.loads(payload["input_hashes_json"].item()) != hashes:
            raise ValueError(f"Cache input hashes differ: {path}")
        if json.loads(payload["config_json"].item()) != config:
            raise ValueError(f"Cache configuration differs: {path}")
        arrays = {name: payload[name].copy() for name in ARRAY_FIELDS}
        metadata = json.loads(payload["metadata_json"].item())
        warnings = tuple(json.loads(payload["warnings_json"].item()))
    n, g = arrays["source_delta"].shape
    if (arrays["target_delta"].shape != (n, g) or arrays["context"].shape != (n, len(CONTEXT_FEATURES))
            or len(metadata) != n or len(arrays["feature_ids"]) != g
            or len(arrays["source_feature_ids"]) != g
            or any(len(arrays[k]) != n for k in ["sample_ids", "compounds", "roles"])
            or arrays["context_feature_ids"].tolist() != list(CONTEXT_FEATURES)
            or any(not np.isfinite(arrays[k]).all() for k in ["source_delta", "target_delta", "context"])):
        raise ValueError(f"Invalid cached data shape or values: {path}")
    return FeedbackData(**arrays, metadata=metadata, warnings=warnings, input_hashes=hashes,
                        cache_path=path, cache_hit=True)


def _save_cache(data: FeedbackData, path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(data, name) for name in ARRAY_FIELDS}
    strings = {"metadata_json": data.metadata, "warnings_json": data.warnings,
               "input_hashes_json": data.input_hashes, "config_json": config}
    arrays.update({name: np.asarray(json.dumps(value, sort_keys=True, allow_nan=False))
                   for name, value in strings.items()})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_feedback_data(
    data_dir: str | Path,
    ortholog_map: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    seed: int = 20261107,
    calibration_fraction: float = 0.2,
    exclude_compounds: Sequence[str] = (),
    exclude_combinations: bool = True,
    include_lps: bool = True,
    attributes_path: str | Path | None = None,
    require_matched_concentration: bool = False,
) -> FeedbackData:
    """Load aligned deltas; cache only in explicitly supplied ``out_dir``.

    Human Entrez IDs define the numerically sorted common axis. Calibration is a
    seeded compound subset of legacy training treatments; all legacy test compounds
    retain the diagnostic role. Explicit infection metadata excludes a compound.
    LPS is marked as a noninfectious stimulus and retained unless disabled.
    """
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    data_dir = Path(data_dir)
    mapping_path = Path(ortholog_map) if ortholog_map else data_dir.parent / "ortholog_map/rat_to_human_one2one.tsv"
    paths = {name: data_dir / name for name in ["pairs_metadata.tsv", "pairs_splits.tsv", "pairs_rat_X.tsv", "pairs_human_Y.tsv"]}
    paths["ortholog_map"] = mapping_path
    default_attributes = data_dir.parent / "metadata/open_tggates_cel_file_attribute.csv"
    resolved_attributes = Path(attributes_path) if attributes_path is not None else default_attributes if default_attributes.is_file() else None
    if resolved_attributes is not None:
        paths["attributes"] = resolved_attributes
    hashes = {name: _hash(path) for name, path in paths.items()}
    config = {"schema_version": 3, "seed": int(seed), "calibration_fraction": float(calibration_fraction),
              "exclude_compounds": sorted({s.strip().casefold() for s in exclude_compounds}),
              "exclude_combinations": bool(exclude_combinations), "include_lps": bool(include_lps),
              "require_matched_concentration": bool(require_matched_concentration),
              "role_assignment_basis": "eligible_compounds_before_concentration_filter"}
    signature = hashlib.sha256(json.dumps({"hashes": hashes, "config": config}, sort_keys=True).encode()).hexdigest()
    cache_path = Path(out_dir) / f"feedback_data_{signature}.npz" if out_dir is not None else None
    if cache_path is not None and cache_path.exists():
        return _load_cache(cache_path, hashes, config)

    metadata = _table(paths["pairs_metadata.tsv"], ["pair_id", "compound", "dose_level", "time_h", "rat_dose_value", "human_dose_value", "split"])
    metadata, unit_warnings = enrich_feedback_metadata(metadata, resolved_attributes)
    by_id = _unique(metadata, "pair_id", "metadata")
    splits = _unique(_table(paths["pairs_splits.tsv"], ["pair_id", "split"]), "pair_id", "splits")
    if set(by_id) != set(splits) or any(row["split"] != splits[key]["split"] for key, row in by_id.items()):
        raise ValueError("Metadata and split IDs/labels differ")
    compound_split, control, times = {}, {}, []
    for i, row in enumerate(metadata):
        compound, split = row["compound"], row["split"]
        if not compound or split not in {"train", "test"}:
            raise ValueError(f"Invalid compound or legacy split: {row['pair_id']}")
        if compound in compound_split and compound_split[compound] != split:
            raise ValueError(f"Compound crosses legacy splits: {compound}")
        compound_split[compound] = split
        times.append(_number(row["time_h"], "time_h"))
        for species in ["rat", "human"]:
            _number(row[f"{species}_dose_value"], f"{species}_dose_value")
        if row["dose_level"].casefold() == "control":
            key = (compound, times[-1])
            if key in control:
                raise ValueError(f"Nonunique compound/time control: {key}")
            control[key] = i
    ids = [row["pair_id"] for row in metadata]
    rat_genes, rat = _matrix(paths["pairs_rat_X.tsv"], ids)
    human_genes, human = _matrix(paths["pairs_human_Y.tsv"], ids)
    mapping = _table(mapping_path, ["rat_entrez", "human_entrez"])
    _unique(mapping, "rat_entrez", "ortholog map")
    _unique(mapping, "human_entrez", "ortholog map")
    if any(not row[k].isdigit() for row in mapping for k in ["rat_entrez", "human_entrez"]):
        raise ValueError("Ortholog IDs must be numeric Entrez identifiers")
    mapping.sort(key=lambda row: int(row["human_entrez"]))
    rat_index, human_index = {g: i for i, g in enumerate(rat_genes)}, {g: i for i, g in enumerate(human_genes)}
    try:
        rat = rat[:, [rat_index[row["rat_entrez"]] for row in mapping]]
        human = human[:, [human_index[row["human_entrez"]] for row in mapping]]
    except KeyError as exc:
        raise ValueError(f"Ortholog absent from matrix header: {exc}") from exc

    requested = set(config["exclude_compounds"])
    excluded = {row["compound"] for row in metadata if row["compound"].casefold() in requested
                or _infection(row) or (exclude_combinations and _combination(row))
                or (not include_lps and row["compound"].casefold() in {"lps", "lipopolysaccharide"})}
    treatment_indices, eligible_indices = [], []
    for i, row in enumerate(metadata):
        if row["dose_level"].casefold() == "control":
            continue
        if (row["compound"], times[i]) not in control:
            raise ValueError(f"Missing compound/time control: {row['pair_id']}")
        if row["compound"] not in excluded:
            eligible_indices.append(i)
            if not require_matched_concentration or row["concentration_match_status"] == "matched":
                treatment_indices.append(i)
    if not treatment_indices:
        raise ValueError("No treatment conditions remain")
    train_compounds = sorted({metadata[i]["compound"] for i in eligible_indices if metadata[i]["split"] == "train"})
    n_calib = math.ceil(len(train_compounds) * calibration_fraction)
    if not 0 < n_calib < len(train_compounds):
        raise ValueError("At least two retained legacy training compounds are required for train/calib roles")
    calibration = set(np.random.default_rng(seed).permutation(train_compounds)[:n_calib])
    controls = [control[(metadata[i]["compound"], times[i])] for i in treatment_indices]
    source_delta = rat[treatment_indices] - rat[controls]
    target_delta = human[treatment_indices] - human[controls]
    selected, context, roles = [], [], []
    levels = ["low", "middle", "high"]
    for i, c in zip(treatment_indices, controls):
        row = metadata[i]
        level = row["dose_level"].casefold()
        if level not in levels:
            raise ValueError(f"Unsupported treatment dose level: {row['dose_level']}")
        is_lps = row["compound"].casefold() in {"lps", "lipopolysaccharide"}
        context.append([times[i], float(row["rat_dose_value"]), float(row["human_dose_value"]),
                        *[float(level == expected) for expected in levels], float(is_lps)])
        role = "diagnostic_test" if row["split"] == "test" else "calib" if row["compound"] in calibration else "train"
        roles.append(role)
        selected.append({**row, "control_pair_id": metadata[c]["pair_id"], "role": role,
                         "context_kind": "noninfectious_lps_stimulus" if is_lps else "chemical_or_stimulus",
                         "is_lps": is_lps})
    context = np.asarray(context, dtype=np.float32)
    if any(not np.isfinite(a).all() for a in [source_delta, target_delta, context]):
        raise ValueError("Nonfinite deltas or context after float32 conversion")
    warnings = [*LIMITATIONS, *unit_warnings]
    if any(not row.get(f"{s}_dose_unit") for row in selected for s in ["rat", "human"]):
        warnings.append("Dose units are missing for retained conditions; equal dose labels or numeric values do not establish concentration matching.")
    unknown = sum(row["concentration_match_status"] == "unknown" for row in selected)
    if unknown:
        warnings.append(f"Nominal concentration comparison is unknown for {unknown} retained conditions; unsupported units are not concentration matched.")
    if require_matched_concentration:
        warnings.append("Only metadata-defined matched nominal concentrations retained; intracellular exposure equivalence is unverified.")
    if excluded:
        warnings.append(f"Metadata-based compound exclusions: {', '.join(sorted(excluded))}")
    data = FeedbackData(
        source_delta=source_delta, target_delta=target_delta, context=context,
        context_feature_ids=np.asarray(CONTEXT_FEATURES), sample_ids=np.asarray([row["pair_id"] for row in selected]),
        compounds=np.asarray([row["compound"] for row in selected]),
        feature_ids=np.asarray([row["human_entrez"] for row in mapping]),
        source_feature_ids=np.asarray([row["rat_entrez"] for row in mapping]),
        roles=np.asarray(roles), metadata=selected, warnings=tuple(warnings), input_hashes=hashes,
        cache_path=cache_path,
    )
    if cache_path is not None:
        _save_cache(data, cache_path, config)
    return data
