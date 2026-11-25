"""LoRA adapter state dict <-> Flower ArrayRecord.

Keys are written in sorted order so every node and the server agree on the
layout; only tensors whose name contains ``lora_`` may be sent (a guard that
no base-model weight or data ever leaves a node by mistake).
"""

from __future__ import annotations

from typing import Any


class NotAnAdapterError(ValueError):
    """Raised when something other than LoRA adapter weights would be sent."""


def check_adapter_only(state: dict[str, Any]) -> None:
    bad = [k for k in state if "lora_" not in k]
    if bad:
        raise NotAnAdapterError(f"refusing to send non-adapter tensors: {bad[:3]}")


def state_to_record(state: dict[str, Any]) -> Any:
    """torch state dict (LoRA only) -> ArrayRecord, keys sorted."""
    from flwr.app import ArrayRecord

    check_adapter_only(state)
    ordered = {k: state[k].detach().to("cpu").contiguous() for k in sorted(state)}
    return ArrayRecord(torch_state_dict=ordered)


def record_to_state(record: Any) -> dict[str, Any]:
    """ArrayRecord -> torch state dict (float tensors on CPU)."""
    state = dict(record.to_torch_state_dict())
    check_adapter_only(state)
    return state


def payload_bytes(record: Any) -> int:
    """Bytes of the adapter arrays in a record (the payload that crosses the network)."""
    return int(sum(arr.numpy().nbytes for arr in record.values()))


__all__ = ["NotAnAdapterError", "check_adapter_only", "payload_bytes", "record_to_state", "state_to_record"]
