"""Dataset discovery, frozen splits, and bug 1's regression.

**Bug 1.** The legacy loader excluded mask directories with
``"mask" not in f.parts``. BBBC038's directory is ``masks`` — *plural* — so the
test never matched and excluded nothing: 29,461 of the 30,131 files it loaded
were per-instance segmentation masks against 670 real images. The model was
trained overwhelmingly on white blobs on black.

The first test below rebuilds that exact tree in miniature and asserts both
halves: that the legacy predicate does nothing, and that
:func:`~sciforensics.global_match.data.discover` returns only the images. It
reproduces the *defect* rather than only checking the fix, so a regression
cannot pass by accident.

No dataset download required — the trees are synthesised, which is also why
these run in CI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sciforensics.global_match.data import (
    BBBC038,
    BIOFORS,
    FLAT,
    DataError,
    Manifest,
    Split,
    assign_split,
    build_manifest,
    check_no_leakage,
    discover,
    layout_for,
)

# BBBC038's real shape, scaled down: one `images/` file per sample, many
# `masks/` files. The 44:1 mask-to-image ratio is the ratio in the real corpus.
IMAGES = 12
MASKS_PER_IMAGE = 44


def _bbbc038_tree(root: Path) -> tuple[int, int]:
    """Build ``<hash>/images/<hash>.png`` + ``<hash>/masks/<n>.png``."""
    for sample in range(IMAGES):
        stem = f"{sample:040x}"
        images = root / stem / "images"
        masks = root / stem / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        # Content must differ per file or content-addressed splitting collapses
        # them all into one hash -- which is correct behaviour, but would make
        # this fixture test nothing about splitting.
        (images / f"{stem}.png").write_bytes(b"image-" + stem.encode())
        for instance in range(MASKS_PER_IMAGE):
            (masks / f"{instance}.png").write_bytes(f"mask-{stem}-{instance}".encode())
    return IMAGES, IMAGES * MASKS_PER_IMAGE


def test_bug1_legacy_predicate_excluded_nothing(tmp_path: Path) -> None:
    """The defect itself: ``"mask" not in parts`` never matches ``"masks"``."""
    images, masks = _bbbc038_tree(tmp_path)
    every_png = sorted(tmp_path.rglob("*.png"))
    assert len(every_png) == images + masks

    # Verbatim legacy filter.
    legacy = [f for f in every_png if "mask" not in f.parts]

    assert len(legacy) == images + masks, (
        "the legacy predicate is expected to exclude nothing -- if this fails the "
        "fixture no longer reproduces bug 1"
    )
    mask_share = masks / (images + masks)
    assert mask_share > 0.8, f"fixture should be mask-dominated, got {mask_share:.1%}"


def test_bug1_discover_returns_only_images(tmp_path: Path) -> None:
    """The fix: exclude by resolved layout, so only ``images/`` survives."""
    images, _ = _bbbc038_tree(tmp_path)
    found = discover(tmp_path)

    assert len(found) == images, f"expected {images} images, got {len(found)}"
    assert all(path.parent.name == "images" for path in found)
    assert not any("masks" in path.parts for path in found)


def test_layout_is_detected_by_structure_not_directory_name(tmp_path: Path) -> None:
    """A corpus copied to any path is still BBBC038 if it is shaped like it."""
    _bbbc038_tree(tmp_path / "somebody_elses_name")
    assert layout_for(tmp_path / "somebody_elses_name") is BBBC038


def test_flat_corpus_falls_back_without_error(tmp_path: Path) -> None:
    for index in range(4):
        (tmp_path / f"figure_{index}.png").write_bytes(f"fig{index}".encode())
    assert layout_for(tmp_path) is FLAT
    assert len(discover(tmp_path)) == 4


def test_exclusion_matches_components_not_substrings(tmp_path: Path) -> None:
    """A file whose *name* contains "mask" is still an image.

    Substring matching is not merely too weak, it is also too strong: a figure
    from a paper about masking would have been dropped silently.
    """
    (tmp_path / "masking_efficiency_fig3.png").write_bytes(b"a real figure")
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "gt.png").write_bytes(b"ground truth")

    found = discover(tmp_path)
    assert [p.name for p in found] == ["masking_efficiency_fig3.png"]


def test_unreadable_root_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        discover(tmp_path / "nope")


def test_empty_corpus_is_an_error_not_an_empty_dataset(tmp_path: Path) -> None:
    """Silently training on zero images is how bug 1 went unnoticed for so long."""
    (tmp_path / "readme.txt").write_text("no images here", encoding="utf-8")
    with pytest.raises(DataError, match="no images found"):
        discover(tmp_path)


def test_unknown_layout_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="unknown dataset layout"):
        layout_for(tmp_path, name="not-a-dataset")


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
# Real digests, not `f"{n:064x}"`: zero-padding on the left makes the top 8 hex
# digits always "00000000", so every synthetic value would land in the first
# bucket and the ratio test would pass vacuously. `assign_split` reads the high
# nibbles precisely because a real SHA-256 is uniform there.
def _digest(value: int) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


RATIOS = ((Split.TRAIN, 0.8), (Split.VAL, 0.1), (Split.CALIB, 0.05), (Split.TEST, 0.05))


def test_split_assignment_is_stable_under_additions() -> None:
    """The property a seeded shuffle does *not* have.

    A seeded shuffle depends on the number and order of inputs, so adding one
    file re-partitions the corpus and silently moves existing files across the
    train/eval boundary. Content addressing makes a file's split a property of
    the file alone.
    """
    digests = [_digest(value) for value in range(200)]
    before = {d: assign_split(d, RATIOS) for d in digests}

    extra = [_digest(value) for value in range(200, 400)]
    after = {d: assign_split(d, RATIOS) for d in digests + extra}

    for digest in digests:
        assert before[digest] == after[digest], "adding files moved an existing file"


def test_split_ratios_are_approximately_honoured() -> None:
    ratios = ((Split.TRAIN, 0.8), (Split.VAL, 0.2))
    counts = {Split.TRAIN: 0, Split.VAL: 0}
    trials = 4000
    for value in range(trials):
        counts[assign_split(_digest(value), ratios)] += 1
    train_share = counts[Split.TRAIN] / trials
    assert 0.77 < train_share < 0.83, train_share


def test_manifest_round_trips(tmp_path: Path) -> None:
    _bbbc038_tree(tmp_path / "corpus")
    manifest = build_manifest(tmp_path / "corpus", dataset="bbbc038-mini")
    path = manifest.write(tmp_path / "splits" / "bbbc038.json")

    reloaded = Manifest.read(path)
    assert reloaded.files == manifest.files
    assert reloaded.splits == manifest.splits
    assert sum(reloaded.counts().values()) == IMAGES


def test_manifest_records_only_images(tmp_path: Path) -> None:
    """Bug 1, end to end through the manifest."""
    _bbbc038_tree(tmp_path / "corpus")
    manifest = build_manifest(tmp_path / "corpus", dataset="bbbc038-mini")
    assert sum(manifest.counts().values()) == IMAGES
    assert all("masks" not in path for path in manifest.files.values())


def test_byte_identical_duplicates_share_a_split(tmp_path: Path) -> None:
    """Otherwise the same figure under two names can straddle the boundary."""
    root = tmp_path / "corpus"
    root.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (root / name).write_bytes(b"the very same bytes")

    manifest = build_manifest(root, dataset="dupes")
    assert len(manifest.files) == 1, "duplicates should collapse to one hash"
    assert sum(len(v) for v in manifest.duplicates.values()) == 2


def test_biofors_is_evaluation_only(tmp_path: Path) -> None:
    """The rule lives in code, not in a README sentence.

    BioFors is the published benchmark (ICCV'21); the prototype trained on it,
    which forecloses evaluating on it.
    """
    root = tmp_path / "biofors"
    root.mkdir()
    for index in range(20):
        (root / f"panel_{index}.png").write_bytes(f"panel{index}".encode())

    manifest = build_manifest(root, dataset="biofors", name="biofors")
    counts = manifest.counts()
    assert counts[Split.TEST.value] == 20
    assert counts[Split.TRAIN.value] == 0

    with pytest.raises(DataError, match="cannot be used for training"):
        build_manifest(root, dataset="biofors", name="biofors", evaluation_only=False)


def test_biofors_layout_declares_its_reason() -> None:
    assert BIOFORS.held_out
    assert "ICCV" in BIOFORS.held_out_reason


def test_leakage_check_catches_a_shared_hash() -> None:
    shared = "f" * 64
    train = Manifest(
        dataset="train-side",
        layout="flat",
        files={shared: "a.png"},
        splits={shared: Split.TRAIN.value},
    )
    evaluation = Manifest(
        dataset="eval-side",
        layout="flat",
        files={shared: "b.png"},
        splits={shared: Split.TEST.value},
    )

    with pytest.raises(DataError, match="both a training and an evaluation split"):
        check_no_leakage(train, evaluation)


def test_leakage_check_passes_on_disjoint_manifests() -> None:
    train = Manifest(
        dataset="t", layout="flat", files={"a" * 64: "a.png"}, splits={"a" * 64: Split.TRAIN.value}
    )
    evaluation = Manifest(
        dataset="e", layout="flat", files={"b" * 64: "b.png"}, splits={"b" * 64: Split.TEST.value}
    )
    check_no_leakage(train, evaluation)


def test_calibration_split_is_distinct_from_test(tmp_path: Path) -> None:
    """B5 must not fit a calibrator on the set it reports against."""
    root = tmp_path / "corpus"
    root.mkdir()
    for index in range(400):
        (root / f"f{index}.png").write_bytes(f"content-{index}".encode())

    manifest = build_manifest(root, dataset="big")
    counts = manifest.counts()
    assert counts[Split.CALIB.value] > 0
    assert counts[Split.TEST.value] > 0

    calib = {d for d, s in manifest.splits.items() if s == Split.CALIB.value}
    test = {d for d, s in manifest.splits.items() if s == Split.TEST.value}
    assert not (calib & test)


def test_bad_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"dataset": "x"}', encoding="utf-8")
    with pytest.raises(DataError, match="missing keys"):
        Manifest.read(path)
