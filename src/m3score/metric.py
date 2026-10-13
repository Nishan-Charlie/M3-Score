"""The M3-Score metric: accumulate features, then compute the three axes."""

from __future__ import annotations

import dataclasses
import logging
import warnings as _warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from . import protocol as _protocol
from . import stats as _stats
from ._errors import M3ScoreWarning
from ._version import __version__
from .backbone import Backbone, load_backbone
from .io import ImageSource, SkippedItem, clear_volume_caches, open_source, sample_indices
from .io.arrays import check_images, to_float

_log = logging.getLogger("m3score")
_DEFAULT_BACKBONE = f"{_protocol.BACKBONE_ID}@{_protocol.BACKBONE_REVISION}"


@dataclass(frozen=True)
class M3Result:
    """The three M3 axes for one real/generated comparison, and what produced them.

    ``fidelity`` is the MMD^2 at the fidelity layer (lower is better),
    ``memorization`` the near-duplicate rate (lower is better), ``coverage`` the
    real-radius coverage (higher is better). The statistics are ``None`` when they
    were skipped.

    ``fidelity_floor`` is the same statistic between random halves of the real set,
    with its standard deviation over the halvings; ``fidelity_floor_by_subject`` says
    whether the halves were subject-disjoint. Because each half is half the size of
    the real set, the floor overstates what a full-size real-vs-real comparison would
    read. When the split is by image its mean is zero by construction and only the
    spread carries information.
    """

    fidelity: float
    memorization: float
    coverage: float
    memorization_nn_dist: float
    n_real: int
    n_gen: int
    fidelity_ci: tuple[float, float] | None = None
    fidelity_p: float | None = None
    fidelity_z: float | None = None
    fidelity_floor: float | None = None
    fidelity_floor_std: float | None = None
    fidelity_floor_by_subject: bool | None = None
    n_real_subjects: int | None = None
    n_gen_subjects: int | None = None
    protocol: str = _protocol.PROTOCOL_VERSION
    backbone: str = _DEFAULT_BACKBONE
    layers: tuple[int, int, int] = _protocol.LAYERS
    warnings: tuple[str, ...] = ()

    @property
    def fidelity_noise_level(self) -> float | None:
        """Floor plus two standard deviations.

        A fidelity value at or below this cannot be told apart from real-vs-real
        variation at this sample size.
        """
        if self.fidelity_floor is None or self.fidelity_floor_std is None:
            return None
        return self.fidelity_floor + 2.0 * self.fidelity_floor_std

    def to_dict(self) -> dict:
        """A JSON-serialisable copy."""
        fields = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in dataclasses.asdict(self).items()
        }
        fields["fidelity_noise_level"] = self.fidelity_noise_level
        return fields

    def __str__(self) -> str:
        return _format(self)


def _count_line(label: str, n: int, subjects: int | None) -> str:
    line = f"  {label:<10} {n:,} images"
    return line if subjects is None else f"{line}, {subjects:,} subjects"


def _format(result: M3Result) -> str:
    lines = [
        f"M3-Score  protocol {result.protocol}  (m3-score {__version__})",
        _count_line("real", result.n_real, result.n_real_subjects),
        _count_line("generated", result.n_gen, result.n_gen_subjects),
        "",
        f"  {'axis':<14}{'value':<9}{'better':<9}detail",
    ]
    detail = []
    if result.fidelity_ci is not None:
        low, high = result.fidelity_ci
        detail.append(f"95% CI [{low:.4f}, {high:.4f}]")
    if result.fidelity_floor is not None:
        split = "by subject" if result.fidelity_floor_by_subject else "by image"
        detail.append(
            f"real-vs-real floor {result.fidelity_floor:.4f} "
            f"+/- {result.fidelity_floor_std:.4f} ({split})"
        )
    lines.append(
        f"  {'fidelity':<14}{result.fidelity:<9.4f}{'lower':<9}{'; '.join(detail)}".rstrip()
    )
    lines.append(
        f"  {'memorization':<14}{result.memorization:<9.3f}{'lower':<9}"
        f"mean nearest-real distance {result.memorization_nn_dist:.3g}"
    )
    lines.append(
        f"  {'coverage':<14}{result.coverage:<9.3f}{'higher':<9}k = {_protocol.COVERAGE_K}"
    )
    noise_level = result.fidelity_noise_level
    if noise_level is not None:
        verdict = "above" if result.fidelity > noise_level else "not above"
        lines += [
            "",
            f"  noise level: {noise_level:.4f} (floor + 2 sd, from halves of the real set); "
            f"fidelity is {verdict} it",
        ]
    if result.fidelity_p is not None:
        lines += [
            f"  permutation test: z = {result.fidelity_z:.1f}, p = {result.fidelity_p:.3g} "
            f"(descriptive, not a model-selection tool)",
        ]
    if result.warnings:
        lines += ["", "  warnings:", *(f"    - {note}" for note in result.warnings)]
    return "\n".join(lines)


class M3Score:
    """Accumulates image features, then computes fidelity, memorization and coverage.

    Use :meth:`score` for two complete image sets, or :meth:`update` batch by batch
    followed by :meth:`compute`. The defaults give the ``m3-v1`` protocol. Passing
    ``backbone`` or ``layers`` gives a result labelled ``custom`` that is not
    comparable with ``m3-v1`` values.
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        backbone: str | torch.nn.Module | None = None,
        layers: Sequence[int] | None = None,
        batch_size: int = 32,
        seed: int = 0,
        progress: bool = False,
    ) -> None:
        if backbone is not None and layers is None:
            raise ValueError(
                "a custom backbone needs layers=(fidelity, memorization, coverage); the m3-v1 "
                "depths 12/9/4 belong to RadioDINO-s16 and do not transfer to another encoder"
            )
        if layers is not None:
            layers = tuple(int(value) for value in layers)
            if len(layers) != 3 or min(layers) < 1:
                raise ValueError(
                    "layers must be three block numbers counted from 1: "
                    "(fidelity, memorization, coverage)"
                )
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "device 'cuda' was requested but CUDA is not available; use device='cpu'"
            )

        custom = backbone is not None or layers is not None
        self.layers: tuple[int, int, int] = layers if layers is not None else _protocol.LAYERS
        self.protocol = _protocol.CUSTOM_PROTOCOL if custom else _protocol.PROTOCOL_VERSION
        self.batch_size = batch_size
        self.seed = seed
        self.progress = progress
        self._backbone_spec = backbone
        self._backbone: Backbone | None = None
        # keyed by True (real) / False (generated)
        self._features: dict[bool, dict[int, list[torch.Tensor]]] = {}
        self._groups: dict[bool, list[str]] = {}
        self._ungrouped: dict[bool, bool] = {}
        self._notes: dict[bool, list[str]] = {}
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self, *, keep_real: bool = False) -> None:
        """Forget accumulated images. ``keep_real`` keeps the real set for the next round."""
        for is_real in (False,) if keep_real else (True, False):
            self._features[is_real] = {layer: [] for layer in set(self.layers)}
            self._groups[is_real] = []
            self._ungrouped[is_real] = False
            self._notes[is_real] = []

    def _get_backbone(self) -> Backbone:
        if self._backbone is None:
            backbone = load_backbone(self._backbone_spec, self.device)
            if max(self.layers) > backbone.num_layers:
                raise ValueError(
                    f"layers {self.layers} reach beyond the backbone's {backbone.num_layers} blocks"
                )
            _log.info("loaded backbone %s on %s", backbone.name, self.device)
            self._backbone = backbone
        return self._backbone

    def _count(self, real: bool) -> int:
        return sum(f.shape[0] for f in self._features[real][self.layers[0]])

    def _ingest(self, prepared: torch.Tensor, real: bool, groups: Sequence | None) -> None:
        total = self._count(True) + self._count(False) + prepared.shape[0]
        if total > _protocol.MAX_TOTAL_IMAGES:
            raise ValueError(
                f"{total:,} images accumulated, more than the {_protocol.MAX_TOTAL_IMAGES:,} "
                f"this metric holds: its statistics need memory that grows with the square of "
                f"that number (about {24 * total * total / 1e9:.0f} GB here). Keep each set to "
                f"about {_protocol.MAX_IMAGES:,} images: draw a random subset before update(), "
                f"or use score(..., n=...)"
            )
        features = self._get_backbone().extract(prepared, self.layers, self.device)
        for layer, value in features.items():
            self._features[real][layer].append(value)
        if groups is None:
            self._ungrouped[real] = True
        else:
            self._groups[real].extend(str(group) for group in groups)

    def _known_groups(self, real: bool) -> list[str] | None:
        """Subject of every accumulated image, or ``None`` unless all of them were labelled."""
        if self._ungrouped[real] or not self._groups[real]:
            return None
        return list(self._groups[real])

    # ------------------------------------------------------------------ input

    def update(self, images, *, real: bool, groups: Sequence | None = None) -> None:
        """Add a batch of images.

        ``images`` is a tensor or array shaped ``(N, H, W)``, ``(N, 1, H, W)`` or
        ``(N, 3, H, W)``: uint8, or floating point in [0, 1]. ``groups`` optionally
        gives one subject identifier per image.
        """
        tensor, divisor = check_images(images)
        if groups is not None and len(groups) != tensor.shape[0]:
            raise ValueError(f"got {len(groups)} group labels for {tensor.shape[0]} images")
        backbone = self._get_backbone()
        for start in range(0, tensor.shape[0], self.batch_size):
            stop = start + self.batch_size
            prepared = backbone.prepare(to_float(tensor[start:stop], divisor))
            self._ingest(prepared, real, None if groups is None else list(groups[start:stop]))

    def _ingest_source(self, source: ImageSource, indices: list[int], real: bool) -> None:
        backbone = self._get_backbone()
        description = "m3score: real" if real else "m3score: generated"
        starts = range(0, len(indices), self.batch_size)
        for start in tqdm(starts, desc=description, disable=not self.progress):
            images, groups = [], []
            for index in indices[start : start + self.batch_size]:
                try:
                    image = source.load(index)
                except SkippedItem as skipped:
                    self._notes[real].append(f"skipped {skipped}")
                    continue
                # each image is resized on its own, so sizes may differ within a source
                images.append(backbone.prepare(image[None]))
                if source.groups is not None:
                    groups.append(source.groups[index])
            if images:
                batch_groups = groups if source.groups is not None else None
                self._ingest(torch.cat(images), real, batch_groups)

    def score(
        self,
        real,
        gen,
        *,
        n: int | None = None,
        stats: bool = True,
        include: str | None = None,
        group_pattern: str | None = None,
        slice_band: tuple[float, float] | None = None,
        window: tuple[float, float] | None = None,
    ) -> M3Result:
        """Score a real set against a generated set. Resets the metric first.

        Each of ``real`` and ``gen`` is a folder or file path, a tensor, or an array.
        ``n`` images are drawn at random from each set (default: all, up to 5,000).
        ``include`` and ``group_pattern`` describe the real set and must match it.
        Each is applied to the generated set only if it matches there too, because
        generated files are usually named differently.
        """
        self.reset()
        try:
            # Both inputs are opened and sampled before any image is decoded, so a
            # mistake in the generated path does not cost the whole real extraction.
            opened = []
            for is_real, origin, seed in ((True, real, self.seed), (False, gen, self.seed + 1)):
                source = open_source(
                    origin,
                    include=include,
                    include_optional=not is_real,
                    group_pattern=group_pattern,
                    group_pattern_optional=not is_real,
                    slice_band=slice_band,
                    window=window,
                )
                indices = sample_indices(len(source), n, source.groups, seed)
                _log.info(
                    "%s set: scoring %d of %d images",
                    "real" if is_real else "generated",
                    len(indices),
                    len(source),
                )
                opened.append((is_real, source, indices))
            for is_real, source, indices in opened:
                self._ingest_source(source, indices, is_real)
                # read after loading: readers add notes about what they find in the files
                self._notes[is_real].extend(source.warnings)
            return self.compute(stats=stats)
        finally:
            clear_volume_caches()

    # ---------------------------------------------------------------- compute

    def compute(self, *, stats: bool = True) -> M3Result:
        """The three axes for everything accumulated so far.

        ``stats=False`` skips the bootstrap interval, permutation test and
        real-vs-real floor.
        """
        fidelity_layer, memorization_layer, coverage_layer = self.layers
        n_real = sum(f.shape[0] for f in self._features[True][fidelity_layer])
        n_gen = sum(f.shape[0] for f in self._features[False][fidelity_layer])
        if n_real < _protocol.MIN_REAL_IMAGES or n_gen < _protocol.MIN_GEN_IMAGES:
            raise ValueError(
                f"need at least {_protocol.MIN_REAL_IMAGES} real and "
                f"{_protocol.MIN_GEN_IMAGES} generated images; "
                f"got {n_real} real and {n_gen} generated"
            )

        def features(real: bool, layer: int) -> torch.Tensor:
            return torch.cat(self._features[real][layer]).to(self.device)

        real_groups, gen_groups = self._known_groups(True), self._known_groups(False)
        notes = [*self._notes[True], *self._notes[False]]

        real_f, gen_f = features(True, fidelity_layer), features(False, fidelity_layer)
        fidelity = _stats.mmd2_rbf(real_f, gen_f)
        interval = p_value = z_score = floor = floor_std = floor_by_subject = None
        if stats:
            interval = _stats.bootstrap_ci(real_f, gen_f, seed=self.seed)
            p_value, z_score, _ = _stats.permutation_test(real_f, gen_f, seed=self.seed)
            floor_result = _stats.real_floor(real_f, real_groups, seed=self.seed)
            if floor_result is not None:
                floor, floor_std, floor_by_subject = floor_result
                if real_groups is not None and not floor_by_subject:
                    notes.append(
                        "the real-vs-real floor was split by image, not by subject, "
                        "because only one subject is known"
                    )
        if real_groups is None:
            # The reference-cohort finding: a real set drawn from few subjects inflates
            # every distance, and without subject labels that cannot be detected.
            notes.append(
                "real subjects unknown: subject diversity could not be checked, a random draw "
                "could not be balanced across subjects, and the real-vs-real floor is split by "
                "image; name them with group_pattern (--group-pattern) or update(groups=...)"
            )
        real_m = features(True, memorization_layer)
        duplicated = _stats.duplicate_fraction(real_m)
        if duplicated > 0.01:
            notes.append(
                f"{duplicated:.0%} of the real images have an exact duplicate in the real set: "
                f"duplicates shrink the nearest-neighbour distances the axes are measured "
                f"against, so memorization reads low (0 once more than half are duplicated) "
                f"and coverage is distorted; remove them"
            )
        memorization, nn_dist = _stats.memorization(real_m, features(False, memorization_layer))
        coverage = _stats.coverage(features(True, coverage_layer), features(False, coverage_layer))

        n_real_subjects = len(set(real_groups)) if real_groups is not None else None
        n_gen_subjects = len(set(gen_groups)) if gen_groups is not None else None
        if min(n_real, n_gen) < _protocol.MIN_IMAGES_WARN:
            notes.append(
                f"fewer than {_protocol.MIN_IMAGES_WARN} images in a set ({n_real:,} real, "
                f"{n_gen:,} generated): estimates are less repeatable at this size"
            )
        if n_real_subjects is not None and n_real_subjects < _protocol.MIN_SUBJECTS_WARN:
            notes.append(
                f"the real set spans only {n_real_subjects} subjects: low subject diversity "
                f"inflates distances and weakens the comparison"
            )
        if self.protocol != _protocol.PROTOCOL_VERSION:
            notes.append("custom backbone or layers: values are not comparable with m3-v1 results")

        unique_notes = tuple(dict.fromkeys(notes))
        for note in unique_notes:
            _warnings.warn(note, M3ScoreWarning, stacklevel=2)
        return M3Result(
            fidelity=fidelity,
            memorization=memorization,
            coverage=coverage,
            memorization_nn_dist=nn_dist,
            n_real=n_real,
            n_gen=n_gen,
            fidelity_ci=interval,
            fidelity_p=p_value,
            fidelity_z=z_score,
            fidelity_floor=floor,
            fidelity_floor_std=floor_std,
            fidelity_floor_by_subject=floor_by_subject,
            n_real_subjects=n_real_subjects,
            n_gen_subjects=n_gen_subjects,
            protocol=self.protocol,
            backbone=self._get_backbone().name,
            layers=self.layers,
            warnings=unique_notes,
        )


def score(
    real,
    gen,
    *,
    device: torch.device | str | None = None,
    backbone: str | torch.nn.Module | None = None,
    layers: Sequence[int] | None = None,
    batch_size: int = 32,
    seed: int = 0,
    progress: bool = True,
    n: int | None = None,
    stats: bool = True,
    include: str | None = None,
    group_pattern: str | None = None,
    slice_band: tuple[float, float] | None = None,
    window: tuple[float, float] | None = None,
) -> M3Result:
    """Score a real image set against a generated one in a single call.

    Builds an :class:`M3Score`, which loads the backbone. To score several pairs, make
    one ``M3Score`` and call its ``score`` method repeatedly.
    """
    metric = M3Score(
        device,
        backbone=backbone,
        layers=layers,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
    )
    return metric.score(
        real,
        gen,
        n=n,
        stats=stats,
        include=include,
        group_pattern=group_pattern,
        slice_band=slice_band,
        window=window,
    )
