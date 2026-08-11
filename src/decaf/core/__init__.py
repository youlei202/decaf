"""Model-agnostic DECAF equations and reproducibility primitives."""

from decaf.core.bootstrap import (
    BootstrapResult,
    bootstrap_mean,
    paired_bootstrap_mean_difference,
)
from decaf.core.decomposition import (
    COMPONENT_NAMES,
    PRIMARY_EPSILON,
    audit_conservation,
    decompose,
    endpoint_effect,
    endpoint_orientation,
    route_response,
)
from decaf.core.manifests import (
    atomic_write_json,
    build_file_manifest,
    sha256_file,
    verify_file_manifest,
)
from decaf.core.metrics import (
    component_fractions,
    endpoint_magnitude,
    pearson_correlation,
    safe_ratio,
    signed_evidence,
)
from decaf.core.quadrature import (
    StreamingTrapezoid,
    integrate_components,
    trapezoid,
    validate_grid,
)
from decaf.core.receipts import (
    aggregate_global_status,
    build_member_receipt,
    finalize_global_receipt,
    write_global_receipt,
    write_member_receipt,
)
from decaf.core.trajectories import (
    StreamingDECAFAccumulator,
    trajectory_scores,
)

__all__ = [
    "COMPONENT_NAMES",
    "PRIMARY_EPSILON",
    "BootstrapResult",
    "StreamingDECAFAccumulator",
    "StreamingTrapezoid",
    "aggregate_global_status",
    "atomic_write_json",
    "audit_conservation",
    "bootstrap_mean",
    "build_file_manifest",
    "build_member_receipt",
    "component_fractions",
    "decompose",
    "endpoint_effect",
    "endpoint_magnitude",
    "endpoint_orientation",
    "finalize_global_receipt",
    "integrate_components",
    "paired_bootstrap_mean_difference",
    "pearson_correlation",
    "route_response",
    "safe_ratio",
    "sha256_file",
    "signed_evidence",
    "trajectory_scores",
    "trapezoid",
    "validate_grid",
    "verify_file_manifest",
    "write_global_receipt",
    "write_member_receipt",
]
