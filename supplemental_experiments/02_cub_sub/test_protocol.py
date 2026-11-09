import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import behavior_targets, decompose, matched_pairs, select_rows


def test_decomposition_conservation_gate_and_opposition():
    stages = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    r = np.array(
        [
            [0, 0.2, -0.4, 0.2, 0.2],
            [0, 0.2, -0.4, 0.2, 0.01],
            [0, -0.2, 0.4, -0.2, -0.2],
            [0, 0.1, 0.1, 0.1, 0.02],
        ]
    )
    result = decompose(r, stages, 0.02)
    assert np.allclose(result["E"] + result["C"] + result["F"], result["Abs"])
    assert result["F"][1] == result["Abs"][1]
    assert result["E"][1] == result["C"][1] == 0
    assert result["C"][0] > 0
    assert result["E"][0] == result["E"][2]
    assert result["C"][0] == result["C"][2]
    assert result["active"][3]  # equality belongs to the active side
    with pytest.raises(ValueError, match="identical"):
        decompose(r + 0.1, stages, 0.02)


def test_pairing_never_crosses_bird_or_attribute_group():
    attrs = ["has_wing_color--black", "has_wing_color--white", "has_eye_color--yellow"]
    rows = [
        {"row_id": i, "bird_label": bird, "attr_label": attr}
        for i, (bird, attr) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])
    ]
    pairs = matched_pairs(rows, attrs, 42)
    assert len(pairs) == 2
    lookup = {r["row_id"]: r for r in rows}
    for pair in pairs:
        a, b = lookup[pair["plus_row"]], lookup[pair["minus_row"]]
        assert a["bird_label"] == b["bird_label"]
        assert {a["attr_label"], b["attr_label"]} == {0, 1}
        assert pair["pair_type"] == "same_class_group_cross_instance"
    assert matched_pairs(list(reversed(rows)), attrs, 42) == pairs


def test_selection_has_no_model_or_outcome_dependence():
    rows = [{"row_id": i, "bird_label": 0, "attr_label": i // 5} for i in range(10)]
    chosen = select_rows(rows, 2, 7)
    assert len(chosen) == 4
    augmented = [dict(row, model_score=1000 - row["row_id"], correctness=False) for row in rows]
    assert [r["row_id"] for r in chosen] == [r["row_id"] for r in select_rows(augmented, 2, 7)]


def test_behavior_labels_use_only_heldout_views_and_intended_attributes():
    clean = np.array([[0.8, 0.1], [0.3, 0.3]])
    view = np.array([[[0.1, 0.8], [0.7, 0.1]], [[0.7, 0.2], [0.3, 0.3]]])
    predictions = np.array([[10, 11], [10, 10]])
    view_predictions = np.array([[[11, 10], [10, 11]], [[11, 10], [10, 10]]])
    targets = behavior_targets(
        clean,
        view,
        predictions,
        view_predictions,
        np.array([10, 10]),
        np.array([11, 11]),
        0.02,
        0.2,
    )
    assert targets["offpath_reversal"].tolist() == [True, False]
    assert targets["active"].tolist() == [True, False]
    assert targets["recognition_fraction"].tolist() == [0.5, 0]
    assert targets["offpath_instability"].all()


def test_cache_rejects_wrong_run_order_and_corruption(tmp_path):
    import importlib.util

    specification = importlib.util.spec_from_file_location(
        "sub_cache_test", Path(__file__).with_name("run.py")
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    path = tmp_path / "scores.npz"
    module.atomic_npz(path, values=np.array([1, 2]))
    module.cache_receipt(path, "immutable_run_hash", ["pair1", 10, 11])
    module.verify_cache(path, "immutable_run_hash", ["pair1", 10, 11])
    for identity, ids in [
        ("changed_run", ["pair1", 10, 11]),
        ("immutable_run_hash", ["pair1", 11, 10]),
    ]:
        with pytest.raises(ValueError, match="mismatch"):
            module.verify_cache(path, identity, ids)
    module.atomic_npz(path, values=np.array([3, 4]))
    with pytest.raises(ValueError, match="mismatch"):
        module.verify_cache(path, "immutable_run_hash", ["pair1", 10, 11])


def test_resume_rejects_changed_source_or_preprocessing_without_overwriting(tmp_path):
    import importlib.util

    specification = importlib.util.spec_from_file_location(
        "sub_identity_test", Path(__file__).with_name("run.py")
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    source = tmp_path / "scoring.py"
    source.write_text("return probabilities[:, 0]\n")
    path = tmp_path / "config.json"
    original = {
        "implementation_sha256": {"run.py": module.checksum(source)},
        "preprocess": "Resize(224), CenterCrop(224), Normalize(CLIP)",
    }
    identity = module.freeze_configuration(path, original)
    original_bytes = path.read_bytes()
    source.write_text("return probabilities[:, 1]\n")
    changed_source = dict(original, implementation_sha256={"run.py": module.checksum(source)})
    changed_preprocessing = dict(
        original, preprocess="Resize(256), CenterCrop(224), Normalize(CLIP)"
    )
    for changed in [changed_source, changed_preprocessing]:
        with pytest.raises(ValueError, match="different config/implementation"):
            module.freeze_configuration(path, changed)
        assert path.read_bytes() == original_bytes
    assert module.freeze_configuration(path, original) == identity
