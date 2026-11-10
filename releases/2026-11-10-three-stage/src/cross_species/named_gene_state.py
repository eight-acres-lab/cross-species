"""Pinned named-gene State loading and explicit assay-contrast adaptation.

The full 6546-coordinate official model is retained. Callers own the measured
gene mask, normalization, loss mask, donor split and optimization. A zero placed
in an excluded input channel is computational masking, with no measurement or
physical-count interpretation attached to that coordinate.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import importlib
import inspect
import io
from pathlib import Path
import platform
import sys

import torch
from torch import nn


MODEL_REPOSITORY = "arcinstitute/st-x-replogle-full"
MODEL_REVISION = "48ad5f70215ab4c58caa5a68e77d837601d29d35"
CHECKPOINT_SHA256 = "29ae411280d5b10c8a46db497bc3df288589cb44b4bbee4a6d42500a3957ae65"
CHECKPOINT_BYTES = 664162239
SOURCE_COMMIT = "9bbfe78a434a55205e4de834e1ea99f85f7a3add"
SOURCE_HASHES = {
    "state.tx.models.state_transition": "45828e6f5ec4da52d1980fd54881e095ad1e8aaece4689c303fe10a671a57b63",
    "state.tx.models.utils": "30dc320813e326516d22fc47162d5ec571511a25345af3f0e0fc1cef94ffd715",
    "state.tx.models.base": "cfc68762c7e70d6e0da381adb9ca037795e835afa535ab602c6e327f13459ebd",
}
NATIVE_DIMENSIONS = {
    "input_dim": 6546, "output_dim": 6546, "gene_dim": 6546,
    "pert_dim": 2024, "batch_dim": 56, "hidden_dim": 328,
}
CONDITION_COLUMNS = {"LPS": 2024, "unstimulated_assay_control": 2025}
CONTEXT_ROWS = {"human_new_assay": 56, "pig_new_assay": 57, "rabbit_new_assay": 58}
ASSAY_CONTROL = "unstimulated_assay_control"
RETAINED_GENE_PREFIXES = (
    "basal_encoder.", "project_out.", "final_down_then_up.", "gene_decoder.",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_named_gene_state(checkpoint_path, *, device="cpu", source_root=None):
    """Strict-load the exact official checkpoint under the audited runtime.

    Returns ``(official_model_in_eval_mode, metadata)``. If supplied, source_root
    must contain src/state; imported source files are hash checked even when a
    state package has previously been imported. No source patch is applied.
    """
    path = Path(checkpoint_path)
    actual_sha = sha256_file(path)
    if actual_sha != CHECKPOINT_SHA256 or path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Named State checkpoint differs from pinned official bytes")
    if source_root is not None:
        sys.path.insert(0, str(Path(source_root) / "src"))
    modules = {name: importlib.import_module(name) for name in SOURCE_HASHES}
    source_paths = {name: Path(inspect.getfile(module)).resolve() for name, module in modules.items()}
    actual_source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    if actual_source_hashes != SOURCE_HASHES:
        raise ValueError("Loaded official architecture or normalization source differs from pinned hashes")
    import transformers
    if str(torch.__version__) != "2.7.1+cu118" or transformers.__version__ != "4.52.4":
        raise ValueError("Named State helper requires audited torch2.7.1+cu118 / transformers4.52.4 runtime")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    hparams = checkpoint["hyper_parameters"]
    if any(hparams.get(name) != value for name, value in NATIVE_DIMENSIONS.items()):
        raise ValueError("Pinned checkpoint native dimensions changed")
    genes = list(hparams["gene_names"])
    if len(genes) != 6546 or len(set(genes)) != 6546:
        raise ValueError("Native gene coordinates must be 6546 unique names")
    cls = modules["state.tx.models.state_transition"].StateTransitionPerturbationModel
    # Constructor initializes temporary parameters before loading. Preserve the
    # caller's RNG state so repeated loads do not alter cell sampling streams.
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        model = cls(**deepcopy(hparams))
    status = model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    metadata = {
        "repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
        "checkpoint_sha256": actual_sha, "checkpoint_bytes": path.stat().st_size,
        "checkpoint_global_step": checkpoint.get("global_step"),
        "source_commit": SOURCE_COMMIT, "source_files": {k: str(v) for k, v in source_paths.items()},
        "source_files_sha256": actual_source_hashes,
        "strict_missing_keys": status.missing_keys, "strict_unexpected_keys": status.unexpected_keys,
        "dimensions": dict(NATIVE_DIMENSIONS), "gene_names": genes,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "transformers": transformers.__version__, "cuda": torch.version.cuda},
        "log1p_from_raw_counts": model.log1p_from_raw_counts,
        "historical_training_normalization_verified": False,
        "helper_sha256": sha256_file(__file__),
    }
    return model, metadata


class ExtendedPerturbationEncoder(nn.Module):
    """Retain the original 2024-column multiplication; add two zero columns."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        reference = next(original.parameters())
        self.extension = nn.Linear(2, 328, bias=False, device=reference.device, dtype=reference.dtype)
        nn.init.zeros_(self.extension.weight)

    def forward(self, value):
        if value.shape[-1] != 2026:
            raise ValueError("Expected 2024 original plus 2 explicit assay condition columns")
        return self.original(value[..., :2024]) + self.extension(value[..., 2024:])


def extend_named_model(model):
    """Copy a freshly loaded official model and append condition/context entries.

    All original gene parameters, intermediate gene paths, and original56batch
    rows remain intact. Construct before running a graph-producing forward on
    the original model, because the official forward caches non-leaf tensors.
    This function preserves RNG streams. Reconstruct this extension before
    strict-loading any later adapted state_dict.
    """
    if any(getattr(model, name, None) != value for name, value in NATIVE_DIMENSIONS.items()):
        raise ValueError("Expected an unextended pinned named-gene model")
    adapted = deepcopy(model)
    old_batch = adapted.batch_encoder
    if not isinstance(old_batch, nn.Embedding) or old_batch.weight.shape != (56, 328):
        raise ValueError("Expected original56×328batch embedding")
    reference = old_batch.weight
    devices = [reference.device.index or 0] if reference.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        adapted.pert_encoder = ExtendedPerturbationEncoder(adapted.pert_encoder)
        new_batch = nn.Embedding(59, 328, device=reference.device, dtype=reference.dtype)
    with torch.no_grad():
        new_batch.weight.zero_()
        new_batch.weight[:56].copy_(reference)
    adapted.batch_encoder = new_batch
    adapted.pert_dim = 2026
    adapted.batch_dim = 59
    adapted.hparams["pert_dim"] = 2026
    adapted.hparams["batch_dim"] = 59
    adapted.hparams["task_extension"] = {
        "condition_columns": dict(CONDITION_COLUMNS), "context_rows": dict(CONTEXT_ROWS),
        "initialization": "new weights zero; all original weights retained",
        "readout": "clamp_min(controls + (f_condition - f_assay_control), 0)",
        "measurement_contract": "caller supplies common observed-axis input mask and excludes masked outputs from loss",
    }
    adapted.train(model.training)
    return adapted


def _context_indices(batch, n_cells, device):
    if isinstance(batch, str):
        if batch not in CONTEXT_ROWS:
            raise ValueError("Unknown assay context: " + batch)
        batch = CONTEXT_ROWS[batch]
    if isinstance(batch, int):
        indices = torch.full((n_cells,), batch, device=device, dtype=torch.long)
    elif isinstance(batch, torch.Tensor):
        if batch.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("Context indices must be integers")
        indices = batch.to(device=device, dtype=torch.long)
        if indices.ndim == 0:
            indices = indices.expand(n_cells)
        if indices.shape != (n_cells,):
            raise ValueError("Supply one context index per actual cell")
    else:
        raise TypeError("batch must be a context name, integer row, or integer tensor")
    if bool(((indices < 0) | (indices >= 59)).any()):
        raise ValueError("Context row outside the retained/appended embedding")
    return indices


def predict_contrast(model, controls, condition="LPS", batch="human_new_assay"):
    """Return differentiable full-axis anchored and native predictions.

    ``controls`` has shape(actual_cells,6546), including the caller's fixed mask.
    Call ``model.eval()`` during fitting to disable dropout while leaving
    autograd enabled. Both native calls participate in the gradient. The return
    dictionary contains prediction, native_condition, native_control, contrast.
    Losses and reported outcomes must index the observed coordinates afterward.

    Subtract native outputs before adding controls so exactly equal condition
    outputs give bitwise control predictions at initialization. Both new zero
    columns initially cancel shared-weight gradients; condition-column updates
    then permit shared-weight gradients as the condition encodings diverge.
    """
    if model.training:
        raise ValueError("Contrast readout requires model.eval(); autograd remains enabled")
    if model.pert_dim != 2026 or model.batch_dim != 59 or not isinstance(model.pert_encoder, ExtendedPerturbationEncoder):
        raise ValueError("Use extend_named_model before the assay contrast readout")
    if condition not in CONDITION_COLUMNS:
        raise ValueError("Condition must name an explicit appended assay column")
    if controls.ndim != 2 or controls.shape[1] != 6546 or controls.shape[0] < 1:
        raise ValueError("Expected actual cell×6546native-coordinate controls")
    reference = next(model.parameters())
    if controls.device != reference.device or controls.dtype != reference.dtype:
        raise ValueError("Control tensor device/dtype must match the model")
    context = _context_indices(batch, len(controls), controls.device)
    perturbation = controls.new_zeros((len(controls), 2026))
    perturbation[:, CONDITION_COLUMNS[condition]] = 1
    control_perturbation = controls.new_zeros((len(controls), 2026))
    control_perturbation[:, CONDITION_COLUMNS[ASSAY_CONTROL]] = 1
    native_condition = model({"ctrl_cell_emb": controls, "pert_emb": perturbation, "batch": context}, padded=False)
    native_control = model({"ctrl_cell_emb": controls, "pert_emb": control_perturbation, "batch": context}, padded=False)
    contrast = native_condition - native_control
    prediction = (controls + contrast).clamp_min(0)
    return {"prediction": prediction, "native_condition": native_condition,
            "native_control": native_control, "contrast": contrast}
