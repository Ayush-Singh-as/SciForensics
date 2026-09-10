"""Dataset discovery and frozen splits. Stage B1.

**Bug 1 — 82% of the training set was binary segmentation masks.** The legacy
loader collected images with ``rglob`` and then filtered them with::

    self.fnames = [f for f in self.fnames if "mask" not in f.parts]

BBBC038's layout is ``<hash>/images/<hash>.png`` alongside
``<hash>/masks/<n>.png`` — **``masks``, plural**. ``"mask" not in parts`` tests
for the exact string ``"mask"``, so it never matched ``"masks"`` and excluded
nothing. Of 30,131 files loaded, 29,461 were per-instance masks; the real image
count is **670**. The model was trained overwhelmingly on white-blobs-on-black.

The fix is structural, not a corrected string. Substring matching over path
components is the wrong tool: it is simultaneously too weak (misses ``masks``,
``mask_gt``, ``ground_truth``) and too strong (a legitimate figure from a paper
about *masking* is silently dropped). Instead each dataset declares its own
layout as a :class:`Layout`, and discovery asks the layout which files are
images. A dataset whose layout is not recognised is an error rather than a
best-effort guess, because a silent 44x inflation of the training set with the
wrong modality is exactly the failure that motivated this module.

**Splits are frozen and content-addressed.** A manifest records the SHA-256 of
every file, and membership is decided by hashing content rather than by
filename or by a random seed. Two consequences that matter:

* Re-running discovery after adding data cannot silently move an existing file
  between train and eval.
* Byte-identical duplicates land in the *same* split by construction, so a file
  present twice under different names cannot leak across the boundary.

**BioFors is held out entirely.** It is the benchmark for this exact task
(ICCV'21), and the prototype consumed it as unlabeled *training* data. Training
on it forecloses evaluating on it, so :data:`HELD_OUT` refuses to hand it to a
training split at all -- the check lives in code, not in a README sentence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from sciforensics.runtime import get_logger

_log = get_logger(__name__)

#: Extensions we accept as figure images. Deliberately not a superset of what
#: OpenCV can open: exotic formats in a scientific corpus are far more often a
#: stray file than an intended sample.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


class Split(str, Enum):
    """Which side of the frozen boundary a file sits on."""

    TRAIN = "train"
    VAL = "val"
    #: Held out for calibration only (stage B5). Kept distinct from TEST so a
    #: calibrator cannot be fitted on the set it is then reported against.
    CALIB = "calib"
    TEST = "test"


class DataError(ValueError):
    """Raised when a corpus does not match its declared layout."""


@dataclass(frozen=True)
class Layout:
    """How one dataset stores images, and what else lives beside them.

    ``image_dirs`` and ``exclude_dirs`` are matched against *path components*,
    exactly and case-insensitively -- never as substrings. That is the whole
    correction: ``"masks"`` is excluded because the layout says the component is
    called ``masks``, not because the letters ``m-a-s-k`` appear somewhere in
    the path.
    """

    name: str
    #: Path components under which images live. Empty means "anywhere not
    #: excluded", for datasets that are simply a flat directory of figures.
    image_dirs: frozenset[str] = frozenset()
    #: Path components that never contain training images: segmentation masks,
    #: ground-truth overlays, annotations.
    exclude_dirs: frozenset[str] = frozenset()
    #: True when this dataset must never be used for training. Enforced by
    #: :func:`build_manifest`, so the rule cannot be forgotten.
    held_out: bool = False
    #: Why it is held out, surfaced in the error when someone tries.
    held_out_reason: str = ""

    def accepts(self, path: Path, root: Path) -> bool:
        """Is ``path`` a training-eligible image under this layout?"""
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return False

        try:
            parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
        except ValueError:  # pragma: no cover - defensive
            return False

        # Two independent gates. Excluded wins: a file under `masks/` is not an
        # image even if it also sits under `images/`.
        excluded = any(part in self.exclude_dirs for part in parts)
        wanted = not self.image_dirs or any(part in self.image_dirs for part in parts)
        return wanted and not excluded


#: BBBC038 (Kaggle 2018 Data Science Bowl). ``<hash>/images/<hash>.png`` next to
#: ``<hash>/masks/<n>.png``. **This is bug 1's dataset**: 29,461 of the 30,131
#: files the prototype loaded came from `masks`.
BBBC038 = Layout(
    name="bbbc038",
    image_dirs=frozenset({"images"}),
    exclude_dirs=frozenset({"masks", "mask"}),
)

#: Polimi western blots. `tampered_blots` ships pixel ground-truth masks and is
#: reserved for localisation evaluation; `automatically_tampered` is usable for
#: training.
POLIMI = Layout(
    name="polimi",
    exclude_dirs=frozenset({"mask", "masks", "ground_truth", "gt"}),
)

#: BioFors (ICCV'21) -- *the* benchmark for this task. Never trained on.
BIOFORS = Layout(
    name="biofors",
    exclude_dirs=frozenset({"mask", "masks", "annotations", "ground_truth"}),
    held_out=True,
    held_out_reason=(
        "BioFors is the published benchmark for this task (ICCV'21). The prototype consumed it "
        "as unlabeled training data; training on it forecloses evaluating on it, so it is "
        "reserved for evaluation only."
    ),
)

#: A plain directory of figures, for a corpus with no dataset-specific layout.
FLAT = Layout(name="flat", exclude_dirs=frozenset({"mask", "masks"}))

LAYOUTS: dict[str, Layout] = {layout.name: layout for layout in (BBBC038, POLIMI, BIOFORS, FLAT)}

#: Datasets that must never reach a training split.
HELD_OUT = frozenset(name for name, layout in LAYOUTS.items() if layout.held_out)


def layout_for(root: Path, *, name: str | None = None) -> Layout:
    """Resolve a dataset's layout, by name or by inspecting the tree.

    Detection is by *structure*, not by directory name, so a corpus copied to
    ``data/train/nuclei`` is still recognised as BBBC038.

    Raises
    ------
    DataError
        If ``name`` is given but unknown.
    """
    if name is not None:
        try:
            return LAYOUTS[name.lower()]
        except KeyError:
            raise DataError(f"unknown dataset layout {name!r}; known: {sorted(LAYOUTS)}") from None

    for candidate in (BBBC038,):
        if candidate.image_dirs and _has_component(root, candidate.image_dirs):
            _log.info("detected %s layout under %s", candidate.name, root)
            return candidate

    _log.info("no dataset-specific layout matched %s; treating as a flat corpus", root)
    return FLAT


def _has_component(root: Path, components: Iterable[str]) -> bool:
    """Does any directory directly under a child of ``root`` match?

    Bounded to two levels: enough to see ``<hash>/images`` without walking a
    30,000-entry tree just to identify it.
    """
    wanted = {c.lower() for c in components}
    for child in _iterdirs(root, limit=64):
        for grandchild in _iterdirs(child, limit=8):
            if grandchild.name.lower() in wanted:
                return True
    return False


def _iterdirs(path: Path, *, limit: int) -> Iterator[Path]:
    if not path.is_dir():
        return
    for index, entry in enumerate(sorted(path.iterdir())):
        if index >= limit:
            return
        if entry.is_dir():
            yield entry


def discover(root: str | Path, *, name: str | None = None) -> list[Path]:
    """Training-eligible images under ``root``, sorted for determinism.

    The counterpart of bug 1: this returns 670 files for BBBC038, not 30,131.
    """
    root = Path(root)
    if not root.is_dir():
        raise DataError(f"corpus directory not found: {root}")

    layout = layout_for(root, name=name)
    found = sorted(p for p in root.rglob("*") if p.is_file() and layout.accepts(p, root))

    if not found:
        raise DataError(
            f"no images found under {root} for the {layout.name!r} layout. "
            f"Expected files with suffixes {sorted(IMAGE_SUFFIXES)}"
            + (f" beneath a {sorted(layout.image_dirs)} directory" if layout.image_dirs else "")
            + "."
        )
    _log.info("discovered %d images under %s (%s layout)", len(found), root, layout.name)
    return found


def file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def assign_split(digest: str, ratios: Sequence[tuple[Split, float]]) -> Split:
    """Deterministically place a file by the *content* of its hash.

    Not ``random.shuffle`` with a seed: a seeded shuffle depends on the number
    and order of inputs, so adding one file re-partitions the whole corpus.
    Hashing content means a file's split is a property of the file alone --
    stable under additions, and identical for byte-identical duplicates, which
    is what stops the same image leaking across the boundary under two names.
    """
    total = sum(weight for _, weight in ratios)
    if total <= 0:
        raise DataError("split ratios must sum to a positive value")

    # Top 8 hex digits -> [0, 1). Plenty of resolution for corpus-scale splits.
    position = int(digest[:8], 16) / 0x1_0000_0000 * total
    upto = 0.0
    for split, weight in ratios:
        upto += weight
        if position < upto:
            return split
    return ratios[-1][0]


DEFAULT_RATIOS: tuple[tuple[Split, float], ...] = (
    (Split.TRAIN, 0.80),
    (Split.VAL, 0.10),
    # Calibration is separate from test so stage B5 cannot fit on the set it
    # reports against.
    (Split.CALIB, 0.05),
    (Split.TEST, 0.05),
)


@dataclass
class Manifest:
    """A frozen, content-addressed split.

    Written to ``benchmarks/splits/*.json`` and read back rather than
    recomputed, so a published metric can always be traced to the exact files
    behind it.
    """

    dataset: str
    layout: str
    #: sha256 -> relative path, so the manifest is portable across machines.
    files: dict[str, str] = field(default_factory=dict)
    #: sha256 -> split.
    splits: dict[str, str] = field(default_factory=dict)
    #: Duplicate content found during the scan: sha256 -> extra paths.
    duplicates: dict[str, list[str]] = field(default_factory=dict)

    def paths(self, split: Split, root: str | Path) -> list[Path]:
        root = Path(root)
        return sorted(
            root / self.files[digest]
            for digest, value in self.splits.items()
            if value == split.value
        )

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {s.value: 0 for s in Split}
        for value in self.splits.values():
            out[value] = out.get(value, 0) + 1
        return out

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset": self.dataset,
                "layout": self.layout,
                "counts": self.counts(),
                "files": self.files,
                "splits": self.splits,
                "duplicates": self.duplicates,
            },
            indent=2,
            sort_keys=True,
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = {"dataset", "layout", "files", "splits"} - set(data)
        if missing:
            raise DataError(f"manifest {path} is missing keys: {sorted(missing)}")
        return cls(
            dataset=data["dataset"],
            layout=data["layout"],
            files=data["files"],
            splits=data["splits"],
            duplicates=data.get("duplicates", {}),
        )


def build_manifest(
    root: str | Path,
    *,
    dataset: str,
    name: str | None = None,
    ratios: Sequence[tuple[Split, float]] = DEFAULT_RATIOS,
    evaluation_only: bool | None = None,
) -> Manifest:
    """Discover, hash and split a corpus.

    Parameters
    ----------
    evaluation_only
        Force every file into :attr:`Split.TEST`. Defaults to the layout's own
        ``held_out`` flag, so BioFors is evaluation-only without the caller
        having to remember.

    Raises
    ------
    DataError
        If a held-out dataset is asked for a training split.
    """
    root = Path(root)
    layout = layout_for(root, name=name)
    held_out = layout.held_out if evaluation_only is None else evaluation_only

    if layout.held_out and evaluation_only is False:
        raise DataError(f"{layout.name} cannot be used for training. {layout.held_out_reason}")

    files = discover(root, name=layout.name)
    manifest = Manifest(dataset=dataset, layout=layout.name)

    for path in files:
        digest = file_sha256(path)
        relative = path.relative_to(root).as_posix()
        if digest in manifest.files:
            # Byte-identical duplicate. Recorded rather than dropped: in a
            # figure corpus a repeat is itself a finding, and it must not be
            # counted twice in a metric.
            manifest.duplicates.setdefault(digest, []).append(relative)
            continue
        manifest.files[digest] = relative
        manifest.splits[digest] = (
            Split.TEST.value if held_out else assign_split(digest, ratios).value
        )

    if manifest.duplicates:
        _log.warning(
            "%d duplicate files (byte-identical) collapsed in %s; they share a split",
            sum(len(v) for v in manifest.duplicates.values()),
            dataset,
        )
    _log.info("manifest %s: %s", dataset, manifest.counts())
    return manifest


def check_no_leakage(*manifests: Manifest) -> None:
    """Assert that no content hash appears in both a train and an eval split.

    Cheap to run and catastrophic to skip: a single leaked figure turns a
    reported metric into a memorisation score.
    """
    train_side: dict[str, str] = {}
    eval_side: dict[str, str] = {}
    for manifest in manifests:
        for digest, split in manifest.splits.items():
            target = train_side if split == Split.TRAIN.value else eval_side
            target[digest] = manifest.dataset

    overlap = set(train_side) & set(eval_side)
    if overlap:
        sample = sorted(overlap)[:5]
        raise DataError(
            f"{len(overlap)} file(s) appear in both a training and an evaluation split; "
            f"first few hashes: {sample}. Every reported metric would be contaminated."
        )
