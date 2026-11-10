"""Full native-gene State adaptation for the GSE325993 two-condition assay.

All6546native gene channels and pretrained gene tensors remain present. The
caller supplies an explicit measured-coordinate mask. POU5F1 is absent from
this native axis, so the interface predicts downstream response only.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import inspect

import torch
from torch import nn

from . import named_gene_state as frozen_named


FROZEN_NAMED_HELPER_SHA256 = "b0255a74b96f33dcd7476c7cdd260d6bb168901d25d07e75c92271a1656327e8"
NATIVE_DIMENSIONS = dict(frozen_named.NATIVE_DIMENSIONS)


@dataclass(frozen=True)
class TwoConditionAssay:
    perturbation: str
    control: str
    contexts: tuple[str, str]

    def __post_init__(self):
        if not self.perturbation or not self.control or self.perturbation == self.control:
            raise ValueError("Two distinct explicit assay condition names required")
        if len(self.contexts) != 2 or len(set(self.contexts)) != 2 or not all(self.contexts):
            raise ValueError("Two distinct named assay contexts required")


POU5F1_ASSAY = TwoConditionAssay("POU5F1_CRISPRi", "NTC_dox5d", ("human_iPSC", "cynomolgus_iPSC"))


def load_native_model(checkpoint_path, *, device="cpu", source_root=None):
    """Reuse the unmodified pinned loader and its checkpoint/source/runtime checks."""
    helper_file = inspect.getfile(frozen_named)
    if frozen_named.sha256_file(helper_file) != FROZEN_NAMED_HELPER_SHA256:
        raise ValueError("Original named-gene helper differs from its frozen version")
    model, metadata = frozen_named.load_named_gene_state(checkpoint_path, device=device, source_root=source_root)
    if "POU5F1" in metadata["gene_names"]:
        raise ValueError("Native target-gene coverage differs from the established assay interface")
    metadata["assay_adapter_scope"] = "Downstream expression response; native gene axis does not contain POU5F1"
    metadata["frozen_named_helper_sha256"] = FROZEN_NAMED_HELPER_SHA256
    return model, metadata


class AssayPerturbationEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        ref = next(original.parameters())
        self.extension = nn.Linear(2, 328, bias=False, device=ref.device, dtype=ref.dtype)
        nn.init.zeros_(self.extension.weight)

    def forward(self, value):
        if value.shape[-1] != 2026:
            raise ValueError("Expected original2024plus2explicit assay columns")
        return self.original(value[..., :2024]) + self.extension(value[..., 2024:])


def extend_assay_model(model, *, assay=POU5F1_ASSAY):
    """Copy the fresh native model, append zero condition/context parameters.

    No original gene layer is replaced or sliced. Old perturbation columns are
    frozen; disconnected gene_decoder parameters are excluded from training.
    Restore old batch rows after each optimizer step using the helper below.
    """
    if not isinstance(assay, TwoConditionAssay):
        raise TypeError("Expected a named two-condition/two-context assay")
    if any(getattr(model, k, None) != v for k, v in NATIVE_DIMENSIONS.items()):
        raise ValueError("Expected fresh unextended official6546gene model")
    adapted = deepcopy(model)
    old_batch = adapted.batch_encoder
    if not isinstance(old_batch, nn.Embedding) or old_batch.weight.shape != (56, 328):
        raise ValueError("Expected original56×328context embedding")
    ref = old_batch.weight
    devices = [ref.device.index or 0] if ref.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        adapted.pert_encoder = AssayPerturbationEncoder(adapted.pert_encoder)
        batch = nn.Embedding(58, 328, device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        batch.weight.zero_()
        batch.weight[:56].copy_(ref)
    adapted.batch_encoder = batch
    adapted.pert_dim, adapted.batch_dim = 2026, 58
    adapted.hparams["pert_dim"], adapted.hparams["batch_dim"] = 2026, 58
    adapted.hparams["assay_extension"] = {
        "specification": asdict(assay),
        "condition_columns": {assay.perturbation: 2024, assay.control: 2025},
        "context_rows": {name: 56 + i for i, name in enumerate(assay.contexts)},
        "initialization": "all new columns/rows zero; original gene parameters retained",
        "target_gene_is_native_output": False,
    }
    first = next(m for m in adapted.pert_encoder.original.modules() if isinstance(m, nn.Linear))
    first.weight.requires_grad_(False)
    if isinstance(adapted.gene_decoder, nn.Module):
        adapted.gene_decoder.requires_grad_(False)
    adapted.train(model.training)
    return adapted


def preserve_old_batch_rows(model):
    """Snapshot original rows once per fresh fit; keep this tensor outside state_dict."""
    return model.batch_encoder.weight[:56].detach().clone()


@torch.no_grad()
def restore_old_batch_rows(model, original_rows):
    if original_rows.shape != (56, 328):
        raise ValueError("Expected saved original56batch rows")
    model.batch_encoder.weight[:56].copy_(original_rows)


def make_assay_optimizer(model, *, condition_lr=1e-3, core_lr=1e-5, weight_decay=0.0005,
                         betas=(0.9, 0.999), eps=1e-8):
    """Default software-probe settings; the new scientific protocol owns selection."""
    extension = list(model.pert_encoder.extension.parameters())
    ids = {id(p) for p in extension}
    core = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    parameters = extension + core
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Repeated optimizer parameter")
    optimizer = torch.optim.AdamW([{"params": extension, "lr": condition_lr}, {"params": core, "lr": core_lr}],
                                  weight_decay=weight_decay, betas=betas, eps=eps)
    return optimizer, parameters


def predict_assay_contrast(model, controls, observed_mask, *, context="human_iPSC", condition=None):
    """Return differentiable full-axis masked-baseline anchored predictions.

    The measured mask comes from the caller's frozen gene array, with one bool
    per original6546coordinate. Excluded input channels are computational zeros.
    The full native paths remain active internally. The caller selects measured
    outputs for every loss/metric; output channels outside the mask are latent.
    """
    if model.training:
        raise ValueError("Use model.eval() with autograd enabled for condition contrasts")
    if not isinstance(model.pert_encoder, AssayPerturbationEncoder) or model.batch_dim != 58:
        raise ValueError("Expected this assay's extended model")
    if controls.ndim != 2 or controls.shape[1] != 6546 or controls.shape[0] < 1:
        raise ValueError("Expected actual cell×6546native-coordinate input")
    if observed_mask.dtype != torch.bool or observed_mask.shape != (6546,):
        raise ValueError("Explicit boolean native-coordinate mask required")
    ref = next(model.parameters())
    if controls.device != ref.device or controls.dtype != ref.dtype or observed_mask.device != controls.device:
        raise ValueError("Model, input and mask must share device; input dtype must match model")
    if not bool(observed_mask.any()):
        raise ValueError("At least one measured native coordinate required")
    extension = model.hparams["assay_extension"]
    spec = extension["specification"]
    condition = spec["perturbation"] if condition is None else condition
    if condition not in extension["condition_columns"] or context not in extension["context_rows"]:
        raise ValueError("Unknown named assay condition/context")
    masked_controls = controls.masked_fill(~observed_mask, 0)
    pert = controls.new_zeros((len(controls), 2026))
    ctrl = controls.new_zeros((len(controls), 2026))
    pert[:, extension["condition_columns"][condition]] = 1
    ctrl[:, extension["condition_columns"][spec["control"]]] = 1
    batch = torch.full((len(controls),), extension["context_rows"][context], device=controls.device, dtype=torch.long)
    native_condition = model({"ctrl_cell_emb": masked_controls, "pert_emb": pert, "batch": batch}, padded=False)
    native_control = model({"ctrl_cell_emb": masked_controls, "pert_emb": ctrl, "batch": batch}, padded=False)
    contrast = native_condition - native_control
    prediction = (masked_controls + contrast).clamp_min(0)
    return {"prediction": prediction, "native_condition": native_condition, "native_control": native_control,
            "contrast": contrast, "masked_controls": masked_controls}
