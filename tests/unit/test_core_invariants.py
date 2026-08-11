"""Mandatory scientific invariants for the authoritative DECAF core."""

from __future__ import annotations

import numpy as np
import pytest

from decaf.core.decomposition import (
    audit_conservation,
    decompose,
    endpoint_effect,
)
from decaf.core.quadrature import StreamingTrapezoid, integrate_components, trapezoid
from decaf.core.trajectories import StreamingDECAFAccumulator, trajectory_scores


def test_active_aligned_response_routes_entirely_to_e() -> None:
    response = np.array([[0.0, 0.1, 0.2], [0.0, -0.1, -0.2]])
    components = decompose(response, np.array([0.2, -0.2]), epsilon=0.2)

    assert components["endpoint_active_sample"].all()
    np.testing.assert_array_equal(components["E"], np.abs(response))
    np.testing.assert_array_equal(components["C"], np.zeros_like(response))
    np.testing.assert_array_equal(components["F"], np.zeros_like(response))


def test_active_opposed_response_routes_entirely_to_c() -> None:
    response = np.array([[0.0, -0.4, -0.8], [0.0, 0.4, 0.8]])
    components = decompose(response, np.array([1.0, -1.0]), epsilon=0.1)

    np.testing.assert_array_equal(components["C"], np.abs(response))
    np.testing.assert_array_equal(components["E"], np.zeros_like(response))
    np.testing.assert_array_equal(components["F"], np.zeros_like(response))


def test_endpoint_null_response_routes_entirely_to_f() -> None:
    response = np.array([[0.0, 0.7, -0.2], [0.0, -0.3, 0.5]])
    components = decompose(response, np.array([0.099, -0.099]), epsilon=0.1)

    assert not components["endpoint_active_sample"].any()
    np.testing.assert_array_equal(components["F"], np.abs(response))
    np.testing.assert_array_equal(components["E"], np.zeros_like(response))
    np.testing.assert_array_equal(components["C"], np.zeros_like(response))


def test_conservation_holds_pointwise_and_after_integration() -> None:
    grid = np.array([0.0, 0.2, 0.65, 1.0])
    response = np.array(
        [
            [0.0, 0.8, -0.4, 0.6],
            [0.0, -0.2, 0.7, -0.9],
            [0.0, 0.3, -0.2, 0.01],
        ]
    )
    components = decompose(response, epsilon=0.1)
    integrated = integrate_components(grid, components)

    assert components["identity_audit"]["passed"]
    assert integrated["identity_audit"]["passed"]
    np.testing.assert_allclose(
        components["Abs"],
        components["E"] + components["C"] + components["F"],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        integrated["Abs"],
        integrated["E"] + integrated["C"] + integrated["F"],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_endpoint_swap_invariance() -> None:
    positive = np.array([[0.3, 0.8, 0.1], [0.7, 0.2, 0.9]])
    negative = np.array([[0.4, 0.1, 0.8], [0.1, 0.6, 0.3]])
    response = endpoint_effect(positive, negative)
    forward = decompose(response, response[:, -1], epsilon=0.1)
    swapped = decompose(
        endpoint_effect(negative, positive),
        -response[:, -1],
        epsilon=0.1,
    )

    for name in ("E", "C", "F", "Abs", "Net"):
        np.testing.assert_array_equal(forward[name], swapped[name])


def test_positive_affine_score_scaling_with_scaled_threshold() -> None:
    positive = np.array([[0.1, 0.5, 0.9], [0.6, 0.4, 0.2]])
    negative = np.array([[0.3, 0.2, 0.1], [0.2, 0.5, 0.8]])
    scale = 3.5
    offset = -4.2
    response = endpoint_effect(positive, negative)
    original = decompose(response, epsilon=0.15)
    transformed = decompose(
        endpoint_effect(scale * positive + offset, scale * negative + offset),
        epsilon=scale * 0.15,
    )

    for name in ("E", "C", "F", "Abs", "Net"):
        np.testing.assert_allclose(
            transformed[name],
            scale * original[name],
            rtol=2.0e-14,
            atol=2.0e-14,
        )
    np.testing.assert_array_equal(
        transformed["endpoint_active_sample"],
        original["endpoint_active_sample"],
    )


def test_finite_grid_and_streaming_implementations_agree() -> None:
    grid = np.array([0.0, 0.1, 0.4, 0.75, 1.0])
    response = np.array(
        [
            [0.0, 0.2, -0.3, 0.5, 0.7],
            [0.0, -0.4, 0.6, -0.2, -0.8],
            [0.0, 0.3, -0.1, 0.2, 0.01],
        ]
    )
    endpoint = response[:, -1]
    finite = trajectory_scores(grid, response, endpoint, epsilon=0.1)
    streaming = StreamingDECAFAccumulator(endpoint, epsilon=0.1)
    for stage, position in enumerate(grid):
        streaming.update(float(position), response[:, stage])
    accumulated = streaming.finalize()

    for name in ("M", "E", "C", "F", "Abs", "Net", "signed_E"):
        np.testing.assert_allclose(
            accumulated[name],
            finite[name],
            rtol=0.0,
            atol=2.0e-15,
        )


def test_float32_scores_are_promoted_before_identity_checks() -> None:
    response = np.array(
        [
            [0.0, 0.10000001, -0.20000002, 0.30000004],
            [0.0, -0.10000001, 0.20000002, 0.00000001],
        ],
        dtype=np.float32,
    )
    components = decompose(response, epsilon=0.02)
    integrated = integrate_components(np.linspace(0.0, 1.0, 4), components)

    for name in ("E", "C", "F", "Abs", "Net"):
        assert components[name].dtype == np.float64
        assert integrated[name].dtype == np.float64
    assert components["identity_audit"]["passed"]
    assert integrated["identity_audit"]["passed"]


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: decompose([]), ValueError),
        (lambda: decompose([0.0, np.nan]), ValueError),
        (lambda: decompose([0.0 + 1.0j]), TypeError),
        (lambda: decompose([True, False]), TypeError),
        (lambda: decompose([0.0, 1.0], epsilon=0.0), ValueError),
        (
            lambda: decompose(
                [[0.0, 1.0], [0.0, -1.0]],
                [[0.0, 1.0], [0.0, -1.0]],
            ),
            ValueError,
        ),
        (lambda: trapezoid([0.0, 1.0], [0.0, 0.0]), ValueError),
        (lambda: trapezoid([0.0, 1.0], [0.1, 1.0]), ValueError),
        (lambda: trapezoid([0.0, 1.0, 2.0], [0.0, 1.0]), ValueError),
    ],
)
def test_core_validation_errors(call: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        call()  # type: ignore[operator]


def test_conservation_audit_rejects_corruption() -> None:
    corrupt = {
        "E": np.array([1.0]),
        "C": np.array([0.0]),
        "F": np.array([0.0]),
        "Abs": np.array([2.0]),
    }
    assert not audit_conservation(corrupt)["passed"]
    with pytest.raises(AssertionError):
        audit_conservation(corrupt, raise_on_error=True)


def test_streaming_validation_errors() -> None:
    stream = StreamingTrapezoid()
    stage = {name: np.array([0.0]) for name in ("E", "C", "F", "Abs")}
    with pytest.raises(ValueError, match="first"):
        stream.update(0.1, stage)

    stream = StreamingTrapezoid()
    stream.update(0.0, stage)
    with pytest.raises(ValueError, match="at least two"):
        stream.finalize()
    with pytest.raises(ValueError, match="strictly increasing"):
        stream.update(0.0, stage)
