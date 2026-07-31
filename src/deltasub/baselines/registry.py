from __future__ import annotations
from .adapters import DARTAdapter, DeltaSubAdapter, MSViTAdapter, SubViTAdapter, ViTAdapter

def build_registry(source_roots: dict | None = None):
    roots = source_roots or {}
    adapters = [ViTAdapter(roots.get("dinov2")), DeltaSubAdapter(roots.get("dinov2")),
                SubViTAdapter(roots.get("dinov2")), MSViTAdapter(roots.get("msvit")),
                DARTAdapter(roots.get("dart"))]
    result = {x.method_id: x for x in adapters}
    if len(result) != len(adapters): raise ValueError("duplicate adapter method identifier")
    return result

def adapter_statuses(source_roots=None):
    return [{"method_id": a.method_id, "display_name": a.display_name,
             "implementation_label": a.implementation_label, "status": a.status.value,
             "reason": a.evidence.reason, "source_revision": a.source_revision,
             "license_status": a.license_status} for a in build_registry(source_roots).values()]
