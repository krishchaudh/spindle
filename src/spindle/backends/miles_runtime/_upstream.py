"""Structural types for the upstream modules Spindle patches.

Installers receive the modules they patch as parameters so tests can inject
fakes. These protocols give ty the pinned upstream surface at those seams:

* a replacement whose signature no longer matches the pinned signature fails
  ty at the ``patch()`` call, and
* the ``TYPE_CHECKING`` trip-wire assignments at the bottom fail ty when a
  future upstream pin renames, removes, or re-shapes a member Spindle relies
  on, forcing the protocol and the replacements to be updated together.

Patched members carry the exact pinned signatures (miles @ MILES_COMMIT,
megatron-bridge @ BRIDGE_COMMIT, Megatron-LM @ the miles release lock).
Members Spindle only calls, or passes through as patch dependents, are
declared loosely: their existence is trip-wired, their signatures are not.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import (
        Qwen3VLModel as _RealQwen3VLModel,
    )
    from megatron.bridge.peft import multi_lora_layers as _real_layers
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.fusions import fused_bias_geglu as _real_fused
    from miles.backends.megatron_utils import model as _real_megatron_model
    from miles.backends.megatron_utils.lora import model as _real_lora_model
    from miles.backends.training_utils import checkpoint_io as _real_checkpoint_io
    from miles.backends.training_utils.data import DataIterator
    from miles.backends.training_utils.loss_hub import (
        logit_processors as _real_logit_processors,
    )
    from miles.backends.training_utils.loss_hub import math_utils as _real_math_utils
    from miles.backends.training_utils.loss_hub import (
        tinker_losses as _real_tinker_losses,
    )
    from miles.ray.rollout import train_data_conversion as _real_conversion
    from miles.ray.specs import train as _real_train_specs
    from miles.tinker import runtime as _real_runtime
    from miles.utils.lora.arguments import validate_multi_lora_args  # noqa: F401
    from miles.utils.sampling_mask import RolloutSamplingMask
    from miles.utils.types import RolloutBatch
    from miles_plugins.models import qwen3_vl as _real_miles_qwen3_vl
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import (
        model as _real_bridge_model,
    )

    import miles.utils.lora.arguments as _real_lora_arguments


class LogitProcessorsModule(Protocol):
    """miles.backends.training_utils.loss_hub.logit_processors."""

    def build_local_sampling_mask(
        self,
        logits: torch.Tensor,
        sampling_mask: RolloutSamplingMask,
        response_indices: Sequence[int] | torch.Tensor,
        *,
        tp_rank: int,
    ) -> torch.Tensor: ...


class MegatronModelModule(Protocol):
    """miles.backends.megatron_utils.model."""

    def get_batch(
        self,
        data_iterator: DataIterator,
        keys: Sequence[str],
        pad_multiplier: int = 128,
        qkv_format: str = "thd",
        get_position_ids: bool = False,
        allgather_cp: bool = False,
    ) -> dict: ...


class LoraModelModule(Protocol):
    """miles.backends.megatron_utils.lora.model."""

    def get_data_iterator(
        self,
        args: Namespace,
        model: torch.nn.Module | Sequence[torch.nn.Module],
        rollout_data: RolloutBatch,
    ) -> tuple[list[DataIterator], list[int]]: ...

    def run_forward_backward(
        self,
        args: Namespace,
        batch_id: int,
        model: Sequence[DDP],
        rollout_data: RolloutBatch,
        *,
        forward_only: bool = False,
    ) -> Any: ...


class TinkerLossesModule(Protocol):
    """miles.backends.training_utils.loss_hub.tinker_losses (pinned v0.1.1)."""

    def _target_logprobs(
        self, args: Namespace, batch: RolloutBatch, logits: torch.Tensor
    ) -> Any: ...

    def _response_masks(
        self, batch: RolloutBatch, log_probs: list[torch.Tensor]
    ) -> Any: ...

    def get_log_probs_and_entropy(
        self,
        logits: torch.Tensor,
        *,
        args: Namespace,
        unconcat_tokens: list[torch.Tensor],
        total_lengths: list[int],
        response_lengths: list[int],
        with_entropy: bool = False,
        entropy_requires_grad: bool = True,
        non_loss_data: bool = True,
        max_seq_lens: list[int] | None = None,
        rollout_sampling_mask: Sequence[RolloutSamplingMask] | None = None,
    ) -> dict: ...


class TinkerLossesV012Module(TinkerLossesModule, Protocol):
    """v0.1.2-only tinker_losses members. The pin is v0.1.1, so there is no
    trip-wire for this protocol; mirror Miles main and update on pin bumps."""

    def _local_response_masks(
        self, args: Namespace, batch: RolloutBatch, log_probs: list[torch.Tensor]
    ) -> Any: ...

    def _gather_per_datum_outputs(
        self,
        args: Namespace,
        batch: RolloutBatch,
        log_probs: list[torch.Tensor],
        per_datum_losses: Any,
    ) -> Any: ...


class MathUtilsModule(Protocol):
    """miles.backends.training_utils.loss_hub.math_utils."""

    def compute_log_probs(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        process_group: dist.ProcessGroup | None,
        *,
        true_on_policy_mode: bool = False,
        vocab_size: int | None = None,
        sampling_mask: torch.Tensor | None = None,
    ) -> Any: ...


class CheckpointIoModule(Protocol):
    """miles.backends.training_utils.checkpoint_io."""

    def write_checkpoint_dir(
        self,
        path: str | Path,
        write_shards: Callable[[Path], None],
        metadata: dict | None = None,
        *,
        overwrite: bool = True,
        completion_marker: str | None = None,
    ) -> None: ...


class ConversionModule(Protocol):
    """miles.ray.rollout.train_data_conversion."""

    def _package_shards(
        self, args: Any, data: dict[str, Any], partitions: Any
    ) -> Any: ...


class TinkerRuntimeModule(Protocol):
    """miles.tinker.runtime."""

    def _build_train_data(self, slot_datums: list) -> dict: ...


class TrainSpecsModule(Protocol):
    """miles.ray.specs.train."""

    def _compute_spec_trainer(
        self, args: Any, *, role: str, num_nodes: int, num_gpus_per_node: int
    ) -> Any: ...


class LoraArgumentsModule(Protocol):
    """miles.utils.lora.arguments."""

    def validate_multi_lora_args(self, args: Any) -> Any: ...


class MultiLoRALinearClass(Protocol):
    """megatron.bridge.peft.multi_lora_layers.MultiLoRALinear."""

    def init_adapter_slot(self, idx: int, rank: int, alpha: float) -> None: ...

    def clear_adapter_slot(self, idx: int) -> None: ...

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any: ...


class MultiLoRAGroupedExpertLinearClass(Protocol):
    """megatron.bridge.peft.multi_lora_layers.MultiLoRAGroupedExpertLinear."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any: ...


class MultiLoraLayersModule(Protocol):
    """megatron.bridge.peft.multi_lora_layers. Patched members are strict;
    call-only members are declared loosely (existence trip-wired only)."""

    @property
    def MultiLoRALinear(self) -> type[MultiLoRALinearClass]: ...

    @property
    def MultiLoRAGroupedExpertLinear(
        self,
    ) -> type[MultiLoRAGroupedExpertLinearClass]: ...

    def _make_slot_routing_hook(
        self, moe_layer: Any, expert_layers: list
    ) -> Callable: ...

    @property
    def parallel_state(self) -> Any: ...

    def gather_from_sequence_parallel_region(
        self, *args: Any, **kwargs: Any
    ) -> Any: ...

    def gather_from_tensor_model_parallel_region(
        self, *args: Any, **kwargs: Any
    ) -> Any: ...

    def reduce_from_tensor_model_parallel_region(
        self, *args: Any, **kwargs: Any
    ) -> Any: ...

    def scatter_to_sequence_parallel_region(self, *args: Any, **kwargs: Any) -> Any: ...

    def all2all_hp2sp(self, *args: Any, **kwargs: Any) -> Any: ...

    def _narrow_token_counts_to_window(self, *args: Any, **kwargs: Any) -> Any: ...

    def _dense_multi_lora_mm(self, *args: Any, **kwargs: Any) -> Any: ...


class FusedBiasGegluModule(Protocol):
    """megatron.core.fusions.fused_bias_geglu. The pinned source is also
    SHA-256 verified at install time in host_syncs."""

    def weighted_bias_quick_geglu_impl(
        self,
        input: Any,
        bias: Any,
        weights: Any,
        fp8_input_store: bool = False,
        linear_offset: float = 0.0,
        clamp_value: Any = None,
    ) -> Any: ...


class MilesQwen3VLModule(Protocol):
    """miles_plugins.models.qwen3_vl. Call-only; the pinned module is
    unannotated, so member existence is trip-wired, signatures are not."""

    def _parse_packed_thd(self, args: Any, kwargs: Any) -> Any: ...

    def _prepare_cp_local_context(self, parsed: Any) -> Any: ...

    def _build_packed_positions(
        self, model: Any, parsed: Any, kwargs: Any, orig_get_rope_index: Any
    ) -> Any: ...


class BridgeModelModule(Protocol):
    """The Megatron-Bridge qwen3_vl model module passed to qwen3_vl_cp."""

    @property
    def Qwen3VLModel(self) -> type[_RealQwen3VLModel]: ...

    def get_rope_index(self, *args: Any, **kwargs: Any) -> Any: ...


if TYPE_CHECKING:
    # Trip-wires: these assignments fail ty when a future upstream pin no
    # longer satisfies the protocol Spindle patches against.
    _check_logit_processors: LogitProcessorsModule = _real_logit_processors
    _check_megatron_model: MegatronModelModule = _real_megatron_model
    _check_lora_model: LoraModelModule = _real_lora_model
    _check_tinker_losses: TinkerLossesModule = _real_tinker_losses
    _check_math_utils: MathUtilsModule = _real_math_utils
    _check_checkpoint_io: CheckpointIoModule = _real_checkpoint_io
    _check_conversion: ConversionModule = _real_conversion
    _check_runtime: TinkerRuntimeModule = _real_runtime
    _check_train_specs: TrainSpecsModule = _real_train_specs
    _check_lora_arguments: LoraArgumentsModule = _real_lora_arguments
    _check_layers: MultiLoraLayersModule = _real_layers
    _check_fused: FusedBiasGegluModule = _real_fused
    _check_miles_qwen3_vl: MilesQwen3VLModule = _real_miles_qwen3_vl
    _check_bridge_model: BridgeModelModule = _real_bridge_model
