from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as torch_F


RegionName = str
SourceName = str


@dataclass
class AlignFlowConfig:
    enabled: bool = True
    order: int = 1
    # Optional independent order for the cross-edit drift forecast.  Leaving
    # it unset preserves the historical shared-order behavior.
    edit_order: Optional[int] = None
    probe_steps: int = 3
    probe_stride: int = 4
    sigma_min: float = 0.90
    max_propagated_turns: int = 4
    store_every: int = 1
    step_anchor_every: int = 5
    step_anchor_offset: int = 0
    min_exact_interval: int = 8
    warmup_steps: int = 2
    region_dilate: int = 1
    kappa_preserved: float = 1.35
    kappa_edited: float = 0.70
    # Optional threshold when no external semantic edit mask was supplied;
    # inferred masks may still route regions without changing this safety tier.
    global_kappa_edited: Optional[float] = None
    # Optional reset schedule when no external edit mask was supplied.
    # Externally masked local edits keep step_anchor_every when this is unset.
    global_step_anchor_every: Optional[int] = None
    global_step_anchor_offset: Optional[int] = None
    # Optional longer probe/warmup window when no external mask was supplied.
    # This is input-semantics aware, not dataset aware.
    global_warmup_steps: Optional[int] = None
    base_threshold: float = 1.0
    error_budget: float = 0.010
    phase_budget_scales: Tuple[float, float, float] = (0.90, 1.00, 1.10)
    ema_decay: float = 0.85
    ratio_min: float = 0.05
    ratio_max: float = 20.0
    min_edit_observations: int = 1
    # Prefer a calibrated cross-edit draft only when its regional error remains
    # close to the best admissible source.  This activates the edit axis without
    # trusting a materially worse draft merely to increase routing counts.
    edit_preference_factor: float = 1.25
    initial_step_ratio: float = 0.70
    phi_min: float = 0.05
    phi_max: float = 20.0
    infer_mask_percentile: float = 0.85
    token_activity_tolerance: float = 0.05
    preserved_activity_trigger: float = 0.25
    preserved_kappa_contraction: float = 0.50
    dynamic_region_expansion: bool = True
    strict_periodic_reset: bool = True
    force_exact_turn: bool = False
    force_final_exact: bool = False
    final_exact_steps: int = 1
    # The paper path predicts CFG-guided outputs.  ``guided_residual`` is the
    # numerically equivalent residual coordinate g(x_t, t) - x_t, reconstructed
    # back to a CFG-guided output before calibration and routing.  Branch and
    # adaptive modes are retained only for engineering ablations.
    step_draft_mode: str = "guided"
    # Damps polynomial extrapolation toward the latest exact residual.  One is
    # the original Taylor forecast and zero is the order-0 hold value.
    step_extrapolation_scale: float = 1.0
    # Keep a small, calibrated cross-edit contribution enabled in the
    # edit-preferred candidate.  The fixed low bound avoids turning a routing
    # count increase into an unconstrained edit-axis overwrite.
    edit_step_blend: bool = True
    edit_blend_decay: float = 0.50
    edit_blend_alpha_min: float = 0.05
    edit_blend_alpha_max: float = 0.05
    cache_on_cpu: bool = True
    preserve_if_exact_available: bool = True
    verbose: bool = False

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, object]]) -> "AlignFlowConfig":
        cfg = cls()
        if not values:
            return cfg
        valid = set(cls.__dataclass_fields__.keys())
        for key, value in values.items():
            if key in valid:
                setattr(cfg, key, value)
        cfg.order = max(0, min(int(cfg.order), 2))
        if cfg.edit_order is not None:
            cfg.edit_order = max(0, min(int(cfg.edit_order), 2))
        cfg.probe_steps = max(1, int(cfg.probe_steps))
        cfg.probe_stride = max(1, int(cfg.probe_stride))
        cfg.min_edit_observations = max(1, int(cfg.min_edit_observations))
        cfg.edit_preference_factor = max(1.0, float(cfg.edit_preference_factor))
        cfg.store_every = max(1, int(cfg.store_every))
        cfg.step_anchor_every = max(1, int(cfg.step_anchor_every))
        cfg.step_anchor_offset = int(cfg.step_anchor_offset)
        cfg.final_exact_steps = max(1, int(cfg.final_exact_steps))
        cfg.min_exact_interval = max(1, int(cfg.min_exact_interval))
        cfg.warmup_steps = max(0, int(cfg.warmup_steps))
        cfg.max_propagated_turns = max(1, int(cfg.max_propagated_turns))
        if cfg.global_step_anchor_every is not None:
            cfg.global_step_anchor_every = max(1, int(cfg.global_step_anchor_every))
        if cfg.global_step_anchor_offset is not None:
            cfg.global_step_anchor_offset = int(cfg.global_step_anchor_offset)
        if cfg.global_warmup_steps is not None:
            cfg.global_warmup_steps = max(0, int(cfg.global_warmup_steps))
        if len(cfg.phase_budget_scales) != 3:
            raise ValueError("phase_budget_scales must contain exactly three values")
        cfg.phase_budget_scales = tuple(
            max(float(value), 1e-6) for value in cfg.phase_budget_scales
        )
        cfg.phi_min = max(float(cfg.phi_min), 1e-6)
        cfg.phi_max = max(float(cfg.phi_max), cfg.phi_min)
        cfg.initial_step_ratio = max(
            cfg.ratio_min, min(cfg.ratio_max, float(cfg.initial_step_ratio))
        )
        cfg.step_extrapolation_scale = max(
            0.0, min(1.0, float(cfg.step_extrapolation_scale))
        )
        cfg.infer_mask_percentile = max(
            0.0, min(1.0, float(cfg.infer_mask_percentile))
        )
        cfg.token_activity_tolerance = max(
            0.0, float(cfg.token_activity_tolerance)
        )
        cfg.preserved_activity_trigger = max(
            0.0, min(1.0, float(cfg.preserved_activity_trigger))
        )
        cfg.preserved_kappa_contraction = max(
            0.0, min(1.0, float(cfg.preserved_kappa_contraction))
        )
        cfg.edit_blend_decay = max(0.0, min(1.0, float(cfg.edit_blend_decay)))
        cfg.edit_blend_alpha_min = float(cfg.edit_blend_alpha_min)
        cfg.edit_blend_alpha_max = max(
            float(cfg.edit_blend_alpha_max), cfg.edit_blend_alpha_min
        )
        cfg.step_draft_mode = str(cfg.step_draft_mode).strip().lower()
        if cfg.step_draft_mode not in {
            "guided", "guided_residual", "adaptive", "branch"
        }:
            raise ValueError(
                "step_draft_mode must be 'guided', 'guided_residual', "
                "'adaptive', or 'branch'"
            )
        return cfg


@dataclass
class CacheEntry:
    step_index: int
    progress: float
    value: torch.Tensor


@dataclass
class AlignFlowDecision:
    edited_source: SourceName
    preserved_source: SourceName
    needs_exact: bool
    reason: str
    update_calibrator: bool
    token_kv: bool = False


class GuidedTrajectoryBank:
    def __init__(self, cache_on_cpu: bool = True) -> None:
        self.cache_on_cpu = cache_on_cpu
        self.entries: List[CacheEntry] = []
        self.turn_id = -1
        self.consecutive_propagated_turns = 0

    def has_entries(self) -> bool:
        return len(self.entries) > 0

    def store_turn(self, entries: Sequence[CacheEntry], turn_id: int) -> None:
        if not entries:
            return
        self.entries = sorted(
            [
                CacheEntry(
                    step_index=e.step_index,
                    progress=float(e.progress),
                    value=self._detach_for_cache(e.value),
                )
                for e in entries
            ],
            key=lambda item: item.progress,
        )
        self.turn_id = turn_id

    def lookup(
        self,
        progress: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.entries:
            return None
        if len(self.entries) == 1:
            return self.entries[0].value.to(device=device, dtype=dtype)

        points = [entry.progress for entry in self.entries]
        idx = bisect.bisect_left(points, float(progress))
        if idx <= 0:
            return self.entries[0].value.to(device=device, dtype=dtype)
        if idx >= len(self.entries):
            return self.entries[-1].value.to(device=device, dtype=dtype)

        left = self.entries[idx - 1]
        right = self.entries[idx]
        denom = max(right.progress - left.progress, 1e-8)
        alpha = float((progress - left.progress) / denom)
        left_value = left.value.to(device=device, dtype=dtype)
        right_value = right.value.to(device=device, dtype=dtype)
        return left_value.lerp(right_value, alpha)

    def _detach_for_cache(self, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach().to(dtype=torch.float16)
        if self.cache_on_cpu:
            value = value.cpu()
        return value.contiguous()


class PolynomialForecaster:
    def __init__(self, order: int, cache_on_cpu: bool = False) -> None:
        self.order = max(0, min(int(order), 2))
        self.cache_on_cpu = cache_on_cpu
        self.entries: List[CacheEntry] = []

    def reset(self) -> None:
        self.entries.clear()

    def add(self, step_index: int, progress: float, value: torch.Tensor) -> None:
        cached = value.detach()
        if self.cache_on_cpu:
            cached = cached.to(dtype=torch.float16).cpu()
        self.entries.append(CacheEntry(step_index, float(progress), cached.contiguous()))
        self.entries.sort(key=lambda item: item.progress)

    def forecast(
        self,
        progress: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.entries:
            return None
        if len(self.entries) == 1 or self.order == 0:
            return self.entries[-1].value.to(device=device, dtype=dtype)

        count = min(self.order + 1, len(self.entries))
        points = self.entries[-count:]
        progress = float(progress)
        result: Optional[torch.Tensor] = None
        for i, item_i in enumerate(points):
            basis = 1.0
            for j, item_j in enumerate(points):
                if i == j:
                    continue
                denom = item_i.progress - item_j.progress
                if abs(denom) < 1e-8:
                    continue
                basis *= (progress - item_j.progress) / denom
            value = item_i.value.to(device=device, dtype=dtype) * basis
            result = value if result is None else result + value
        return result


class EditAxisTaylorForecaster(PolynomialForecaster):
    """Low-order edit-axis expansion in normalized denoising progress."""

    def __init__(self, order: int, cache_on_cpu: bool = False) -> None:
        super().__init__(order=order, cache_on_cpu=cache_on_cpu)
        self._newton_points: List[float] = []
        self._newton_coefficients: List[torch.Tensor] = []

    def reset(self) -> None:
        super().reset()
        self._newton_points.clear()
        self._newton_coefficients.clear()

    def add(self, step_index: int, progress: float, value: torch.Tensor) -> None:
        super().add(step_index, progress, value)
        count = min(self.order + 1, len(self.entries))
        self.entries = self.entries[-count:]
        points = self.entries
        xs = [float(item.progress) for item in points]
        coefficients = [item.value for item in points]
        for level in range(1, count):
            for index in range(count - 1, level - 1, -1):
                denominator = xs[index] - xs[index - level]
                if abs(denominator) < 1e-8:
                    self._newton_points = []
                    self._newton_coefficients = []
                    return
                coefficients[index] = (
                    coefficients[index] - coefficients[index - 1]
                ) / denominator
        self._newton_points = xs
        self._newton_coefficients = coefficients

    def forecast(
        self,
        progress: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.entries:
            return None
        if not self._newton_coefficients:
            return self.entries[-1].value.to(device=device, dtype=dtype)
        count = len(self._newton_coefficients)
        coefficients = [
            value.to(device=device, dtype=dtype)
            for value in self._newton_coefficients
        ]
        if count == 1:
            return coefficients[0]
        result = coefficients[-1]
        target = float(progress)
        for index in range(count - 2, -1, -1):
            result = (
                coefficients[index]
                + (target - self._newton_points[index]) * result
            )
        return result


class CrossEditForecaster:
    def __init__(self, cfg: AlignFlowConfig, bank: GuidedTrajectoryBank) -> None:
        self.cfg = cfg
        self.bank = bank
        self.reset_turn()

    def reset_turn(self) -> None:
        self.active = False
        self.disabled_reason = ""
        self.eta = 0.0
        self.sigma = 0.0
        drift_order = self.cfg.order if self.cfg.edit_order is None else self.cfg.edit_order
        self.drift = EditAxisTaylorForecaster(
            order=drift_order, cache_on_cpu=False
        )
        self.probe_count = 0
        self.latest_drift: Optional[torch.Tensor] = None

    def can_attempt(self) -> bool:
        if not self.bank.has_entries():
            self.disabled_reason = "no_previous_turn"
            return False
        if self.bank.consecutive_propagated_turns >= self.cfg.max_propagated_turns:
            self.disabled_reason = "edit_staleness_reset"
            return False
        return True

    def observe_probe(
        self,
        step_index: int,
        progress: float,
        current_guided: torch.Tensor,
    ) -> None:
        if not self.can_attempt():
            self.active = False
            return
        prev = self.bank.lookup(
            progress,
            device=current_guided.device,
            dtype=current_guided.dtype,
        )
        if prev is None:
            self.active = False
            self.disabled_reason = "missing_previous_anchor"
            return
        drift_value = current_guided.detach() - prev
        self.latest_drift = drift_value
        self.drift.add(step_index, progress, drift_value)
        self.probe_count += 1

        if self.probe_count == 1:
            self.sigma = cosine_similarity(current_guided, prev)
            if self.sigma < self.cfg.sigma_min:
                self.active = False
                self.disabled_reason = "low_cross_edit_similarity"
                return
            self.eta = (self.sigma - self.cfg.sigma_min) / max(1.0 - self.cfg.sigma_min, 1e-8)
            self.eta = float(max(0.0, min(1.0, self.eta)))

        drift_order = self.cfg.order if self.cfg.edit_order is None else self.cfg.edit_order
        min_probes = min(drift_order + 1, self.cfg.probe_steps)
        self.active = self.probe_count >= min_probes and self.sigma >= self.cfg.sigma_min

    def observe_anchor(
        self,
        step_index: int,
        progress: float,
        current_guided: torch.Tensor,
    ) -> None:
        if not self.active or not self.can_attempt():
            return
        prev = self.bank.lookup(
            progress,
            device=current_guided.device,
            dtype=current_guided.dtype,
        )
        if prev is None:
            return
        drift_value = current_guided.detach() - prev
        self.latest_drift = drift_value
        self.drift.add(step_index, progress, drift_value)

    def forecast(
        self,
        progress: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.active:
            return None
        prev = self.bank.lookup(progress, device=device, dtype=dtype)
        drift = self.drift.forecast(progress, device=device, dtype=dtype)
        if prev is None or drift is None:
            return None
        return prev + float(self.eta) * drift


class ClosedLoopCalibrator:
    def __init__(self, cfg: AlignFlowConfig) -> None:
        self.cfg = cfg
        self.ema: Dict[Tuple[RegionName, SourceName, int], float] = {}
        self.observations: Dict[Tuple[RegionName, SourceName, int], int] = {}

    def score(self, region: RegionName, source: SourceName, phase: int) -> float:
        default = self.cfg.initial_step_ratio if source == "step" else 1.0
        return self.ema.get((region, source, phase), default)

    def observe(
        self,
        region: RegionName,
        source: SourceName,
        phase: int,
        error: float,
        budget: float,
    ) -> None:
        budget = max(float(budget), 1e-12)
        ratio = float(error) / budget
        ratio = max(self.cfg.ratio_min, min(self.cfg.ratio_max, ratio))
        key = (region, source, phase)
        old = self.score(region, source, phase)
        self.ema[key] = self.cfg.ema_decay * old + (1.0 - self.cfg.ema_decay) * ratio
        self.observations[key] = self.observations.get(key, 0) + 1

    def threshold(self, region: RegionName) -> float:
        if region == "preserved":
            return self.cfg.base_threshold * self.cfg.kappa_preserved
        return self.cfg.base_threshold * self.cfg.kappa_edited

    def threshold_scale(self, region: RegionName, source: SourceName, phase: int) -> float:
        ratio = max(self.score(region, source, phase), 1e-12)
        return max(self.cfg.phi_min, min(self.cfg.phi_max, 1.0 / ratio))


class AlignFlowController:
    def __init__(
        self,
        config: Optional[AlignFlowConfig | Mapping[str, object]] = None,
        bank: Optional[GuidedTrajectoryBank] = None,
    ) -> None:
        if isinstance(config, AlignFlowConfig):
            self.cfg = config
        else:
            self.cfg = AlignFlowConfig.from_mapping(config)
        self.bank = bank or GuidedTrajectoryBank(cache_on_cpu=self.cfg.cache_on_cpu)
        self.calibrator = ClosedLoopCalibrator(self.cfg)
        self.step_forecaster = PolynomialForecaster(order=self.cfg.order, cache_on_cpu=False)
        self.cond_residual_forecaster = PolynomialForecaster(
            order=self.cfg.order, cache_on_cpu=False
        )
        self.uncond_residual_forecaster = PolynomialForecaster(
            order=self.cfg.order, cache_on_cpu=False
        )
        self.step_variant_ema: Dict[Tuple[str, int], float] = {}
        self._step_variant_drafts: Dict[str, Optional[torch.Tensor]] = {}
        self._last_step_variant = "guided"
        self._raw_edit_hat: Optional[torch.Tensor] = None
        self.edit_blend_alpha: Dict[int, float] = {}
        self.cross_forecaster = CrossEditForecaster(self.cfg, self.bank)
        self.turn_id = 0
        self.total_steps = 0
        self.progress_values: List[float] = []
        self.edited_mask: Optional[torch.Tensor] = None
        self.preserved_mask: Optional[torch.Tensor] = None
        self.current_exact_entries: List[CacheEntry] = []
        self.last_exact_index = -10**9
        self.turn_used_cross_edit = False
        self.force_exact_turn = False
        self.mask_was_provided = True
        self.region_revision = 0
        self.kappa_preserved_current = float(self.cfg.kappa_preserved)
        self.probe_indices: Tuple[int, ...] = ()
        self.stats: Dict[str, int] = {}

    def begin_turn(
        self,
        timesteps: Sequence[torch.Tensor],
        input_mask: Optional[torch.Tensor],
        target_shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.total_steps = len(timesteps)
        self.progress_values = normalized_timestep_progress(timesteps)
        self.step_forecaster.reset()
        self.cond_residual_forecaster.reset()
        self.uncond_residual_forecaster.reset()
        self.step_variant_ema.clear()
        self._step_variant_drafts.clear()
        self._last_step_variant = "guided"
        self._raw_edit_hat = None
        self.edit_blend_alpha.clear()
        self.cross_forecaster.reset_turn()
        self.force_exact_turn = bool(
            self.cfg.force_exact_turn
            or (
                self.cfg.strict_periodic_reset
                and self.bank.has_entries()
                and self.bank.consecutive_propagated_turns
                >= self.cfg.max_propagated_turns
            )
        )
        self.force_exact_reason = (
            "protocol_exact"
            if self.cfg.force_exact_turn
            else "edit_staleness_reset"
        )
        self.mask_was_provided = input_mask is not None
        self.region_revision = 0
        self.kappa_preserved_current = float(self.cfg.kappa_preserved)
        self.probe_indices = tuple(
            index
            for index in (
                offset * self.cfg.probe_stride
                for offset in range(self.cfg.probe_steps)
            )
            if index < self.total_steps
        )

        # Edit reliability is specific to the current edit pair.
        # Keep the learned step-cache scores, but recalibrate edit cache
        # independently at every new turn.
        self.calibrator.ema = {
            key: value
            for key, value in self.calibrator.ema.items()
            if key[1] != "edit"
        }
        self.calibrator.observations = {
            key: value
            for key, value in self.calibrator.observations.items()
            if key[1] != "edit"
        }

        self.current_exact_entries = []
        self.last_exact_index = -10**9
        self.turn_used_cross_edit = False
        self.stats = {
            "exact": 0,
            "token_kv_used": 0,
            "token_kv_recorded": 0,
            "skip": 0,
            "step_used": 0,
            "step_guided_used": 0,
            "step_branch_used": 0,
            "branch_residual_recorded": 0,
            "branch_residual_evaluated": 0,
            "edit_used": 0,
            "probe": 0,
            "anchor": 0,
            "reset_exact": 0,
            "inferred_mask": 0,
            "region_expansions": 0,
        }
        self.edited_mask, self.preserved_mask = build_region_masks(
            input_mask=input_mask,
            target_shape=target_shape,
            device=device,
            dtype=dtype,
            dilate_radius=self.cfg.region_dilate,
        )
        if self.cfg.verbose:
            logging.info(
                "AlignFlow begin turn=%s prev_entries=%s prev_turn=%s",
                self.turn_id,
                len(self.bank.entries),
                self.bank.turn_id,
            )

    def progress(self, step_index: int) -> float:
        if len(self.progress_values) == self.total_steps and 0 <= step_index < len(
            self.progress_values
        ):
            return self.progress_values[step_index]
        if self.total_steps <= 1:
            return 0.0
        return float(step_index) / float(self.total_steps - 1)

    def phase(self, step_index: int) -> int:
        p = self.progress(step_index)
        if p < 1.0 / 3.0:
            return 0
        if p < 2.0 / 3.0:
            return 1
        return 2

    def error_budget(self, step_index: int) -> float:
        progress = self.progress(step_index)
        position = progress * 2.0
        left = min(int(position), 1)
        alpha = position - left
        scales = self.cfg.phase_budget_scales
        scale = scales[left] * (1.0 - alpha) + scales[left + 1] * alpha
        return float(self.cfg.error_budget) * float(scale)

    def region_threshold(self, region: RegionName) -> float:
        kappa = (
            self.kappa_preserved_current
            if region == "preserved"
            else float(self.cfg.kappa_edited)
        )
        return float(self.cfg.base_threshold) * kappa

    def make_drafts(
        self,
        step_index: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_input: Optional[torch.Tensor] = None,
        guide_scale: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        p = self.progress(step_index)
        phase = self.phase(step_index)
        branch_only = self.cfg.step_draft_mode == "branch"
        guided_only = self.cfg.step_draft_mode in {"guided", "guided_residual"}
        guided_residual = self.cfg.step_draft_mode == "guided_residual"
        guided_hat = None
        if not branch_only:
            guided_hat = self.step_forecaster.forecast(
                p, device=device, dtype=dtype
            )
            if guided_residual and guided_hat is not None:
                if latent_input is None:
                    raise RuntimeError(
                        "guided_residual prediction requires latent_input"
                    )
                if self.step_forecaster.entries:
                    latest = self.step_forecaster.entries[-1].value.to(
                        device=device, dtype=dtype
                    )
                    guided_hat = latest + self.cfg.step_extrapolation_scale * (
                        guided_hat - latest
                    )
                guided_hat = latent_input.to(device=device, dtype=dtype) + guided_hat
        guided_score = self.step_variant_ema.get(("guided", phase))
        branch_score = self.step_variant_ema.get(("branch", phase))
        branch_is_preferred = branch_only or (
            not guided_only
            and branch_score is not None
            and (guided_score is None or branch_score <= guided_score * 0.98)
        )
        anchor_every = self.cfg.step_anchor_every
        calibration_due = (
            step_index - self.last_exact_index >= self.cfg.min_exact_interval
            or (
                anchor_every is not None
                and anchor_every > 0
                and step_index > 0
                and step_index % anchor_every == 0
            )
        )
        cond_residual = None
        uncond_residual = None
        if branch_only or branch_is_preferred or (
            calibration_due and not guided_only
        ):
            cond_residual = self.cond_residual_forecaster.forecast(
                p, device=device, dtype=dtype
            )
            uncond_residual = self.uncond_residual_forecaster.forecast(
                p, device=device, dtype=dtype
            )
        branch_hat = None
        if (
            latent_input is not None
            and guide_scale is not None
            and cond_residual is not None
            and uncond_residual is not None
        ):
            latent = latent_input.to(device=device, dtype=dtype)
            cond_hat = latent + cond_residual
            uncond_hat = latent + uncond_residual
            branch_hat = uncond_hat + float(guide_scale) * (cond_hat - uncond_hat)
        if branch_only:
            self._step_variant_drafts = {"branch": branch_hat}
            self._last_step_variant = "branch"
            step_hat = branch_hat
        else:
            self._step_variant_drafts = {
                "guided": guided_hat,
                "branch": branch_hat,
            }
            self._last_step_variant = "guided"
            step_hat = guided_hat
            if branch_hat is not None and branch_is_preferred:
                self._last_step_variant = "branch"
                step_hat = branch_hat
        raw_edit_hat = self.cross_forecaster.forecast(p, device=device, dtype=dtype)
        self._raw_edit_hat = raw_edit_hat
        edit_hat = raw_edit_hat
        blend_alpha = self.edit_blend_alpha.get(phase)
        if (
            self.cfg.edit_step_blend
            and blend_alpha is not None
            and step_hat is not None
            and raw_edit_hat is not None
        ):
            edit_hat = step_hat + float(blend_alpha) * (raw_edit_hat - step_hat)
        return step_hat, edit_hat

    def decide(
        self,
        step_index: int,
        step_hat: Optional[torch.Tensor],
        edit_hat: Optional[torch.Tensor],
    ) -> AlignFlowDecision:
        if self.force_exact_turn:
            return AlignFlowDecision(
                edited_source="exact",
                preserved_source="exact",
                needs_exact=True,
                reason=self.force_exact_reason,
                update_calibrator=False,
            )
        force_reason = self._forced_exact_reason(step_index, step_hat, edit_hat)
        if force_reason is not None:
            return AlignFlowDecision(
                edited_source="exact",
                preserved_source="exact",
                needs_exact=True,
                reason=force_reason,
                update_calibrator=force_reason == "budget_anchor",
            )

        phase = self.phase(step_index)
        edited_empty = self._region_is_empty("edited")
        preserved_empty = self._region_is_empty("preserved")

        edited = "step"
        if not edited_empty:
            edited = self._select_source(
                region="edited",
                phase=phase,
                candidates=["step"] if step_hat is not None else [],
            )

        preserved = "step"
        if not preserved_empty:
            preserved_candidates: List[SourceName] = []
            if edit_hat is not None:
                preserved_candidates.append("edit")
            if step_hat is not None:
                preserved_candidates.append("step")
            preserved = self._select_source(
                region="preserved",
                phase=phase,
                candidates=preserved_candidates,
            )

        needs_exact = (edited == "exact" and not edited_empty) or (
            preserved == "exact" and not preserved_empty
        )
        return AlignFlowDecision(
            edited_source=edited,
            preserved_source=preserved,
            needs_exact=needs_exact,
            reason="calibrated_exact" if needs_exact else "skip",
            update_calibrator=needs_exact,
        )

    def compose(
        self,
        decision: AlignFlowDecision,
        exact: Optional[torch.Tensor],
        step_hat: Optional[torch.Tensor],
        edit_hat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        source_map = {
            "exact": exact,
            "step": step_hat,
            "edit": edit_hat,
        }
        edited_value = source_map.get(decision.edited_source)
        preserved_value = source_map.get(decision.preserved_source)
        fallback_value = exact if exact is not None else step_hat if step_hat is not None else edit_hat

        if edited_value is None:
            if self._region_is_empty("edited") and fallback_value is not None:
                edited_value = fallback_value
            elif exact is None:
                raise RuntimeError("AlignFlow edited region requires exact output but exact is None.")
            else:
                edited_value = exact
        if preserved_value is None:
            if self._region_is_empty("preserved") and fallback_value is not None:
                preserved_value = fallback_value
            elif exact is None:
                raise RuntimeError("AlignFlow preserved region requires exact output but exact is None.")
            else:
                preserved_value = exact

        if not self.cfg.preserve_if_exact_available and exact is not None:
            return exact
        if self.edited_mask is None or self.preserved_mask is None:
            return edited_value
        return self.edited_mask * edited_value + self.preserved_mask * preserved_value

    def observe_exact(
        self,
        step_index: int,
        exact: torch.Tensor,
        step_hat: Optional[torch.Tensor],
        edit_hat: Optional[torch.Tensor],
        update_calibrator: bool,
        monitor_regions: bool = False,
        exact_cond: Optional[torch.Tensor] = None,
        exact_uncond: Optional[torch.Tensor] = None,
        latent_input: Optional[torch.Tensor] = None,
    ) -> None:
        p = self.progress(step_index)
        phase = self.phase(step_index)

        if step_index in self.probe_indices:
            self.cross_forecaster.observe_probe(step_index, p, exact)
            self.stats["probe"] += 1
            if (
                not self.mask_was_provided
                and self.stats["inferred_mask"] == 0
                and self.cross_forecaster.latest_drift is not None
            ):
                self._infer_regions_from_probe(self.cross_forecaster.latest_drift)
        elif update_calibrator:
            self.cross_forecaster.observe_anchor(step_index, p, exact)

        if update_calibrator:
            if step_hat is not None:
                self._observe_source(
                    "edited", "step", phase, step_index, exact, step_hat
                )
                self._observe_source(
                    "preserved", "step", phase, step_index, exact, step_hat
                )
            calibrated_edit_hat = edit_hat
            if (
                self.cfg.edit_step_blend
                and step_hat is not None
                and self._raw_edit_hat is not None
            ):
                observed_alpha = least_squares_blend_alpha(
                    step=step_hat,
                    edit=self._raw_edit_hat,
                    exact=exact,
                    mask=self.preserved_mask,
                    alpha_min=self.cfg.edit_blend_alpha_min,
                    alpha_max=self.cfg.edit_blend_alpha_max,
                )
                previous_alpha = self.edit_blend_alpha.get(phase, observed_alpha)
                blend_alpha = (
                    self.cfg.edit_blend_decay * previous_alpha
                    + (1.0 - self.cfg.edit_blend_decay) * observed_alpha
                )
                self.edit_blend_alpha[phase] = blend_alpha
                calibrated_edit_hat = step_hat + blend_alpha * (
                    self._raw_edit_hat - step_hat
                )
                if self.cfg.verbose:
                    logging.info(
                        "AlignFlow edit-step-blend turn=%d phase=%d "
                        "observed_alpha=%.6f alpha=%.6f",
                        self.turn_id,
                        phase,
                        observed_alpha,
                        blend_alpha,
                    )
            if calibrated_edit_hat is not None:
                self._observe_source(
                    "preserved",
                    "edit",
                    phase,
                    step_index,
                    exact,
                    calibrated_edit_hat,
                )
            for variant, draft in self._step_variant_drafts.items():
                if draft is None:
                    continue
                if variant == "branch":
                    self.stats["branch_residual_evaluated"] += 1
                error = masked_relative_mse(draft, exact, None)
                ratio = max(
                    self.cfg.ratio_min,
                    min(
                        self.cfg.ratio_max,
                        error / max(self.error_budget(step_index), 1e-12),
                    ),
                )
                key = (variant, phase)
                previous = self.step_variant_ema.get(key, ratio)
                self.step_variant_ema[key] = (
                    self.cfg.ema_decay * previous
                    + (1.0 - self.cfg.ema_decay) * ratio
                )
            if monitor_regions:
                self._monitor_preserved_activity(step_index, exact)

        if self.cfg.step_draft_mode != "branch":
            step_observation = exact
            if self.cfg.step_draft_mode == "guided_residual":
                if latent_input is None:
                    raise RuntimeError(
                        "guided_residual observation requires latent_input"
                    )
                step_observation = exact - latent_input.to(
                    device=exact.device, dtype=exact.dtype
                )
            self.step_forecaster.add(step_index, p, step_observation)
        if (
            self.cfg.step_draft_mode not in {"guided", "guided_residual"}
            and exact_cond is not None
            and exact_uncond is not None
            and latent_input is not None
        ):
            latent = latent_input.detach().to(
                device=exact_cond.device, dtype=exact_cond.dtype
            )
            self.cond_residual_forecaster.add(
                step_index, p, exact_cond.detach() - latent
            )
            self.uncond_residual_forecaster.add(
                step_index, p, exact_uncond.detach() - latent
            )
            self.stats["branch_residual_recorded"] += 1
        self.last_exact_index = step_index
        if step_index % self.cfg.store_every == 0:
            cached_value = exact.detach()
            self.current_exact_entries.append(
                CacheEntry(step_index=step_index, progress=p, value=cached_value)
            )

    def observe_token_exact(
        self,
        step_index: int,
        composed: torch.Tensor,
        edited_exact: torch.Tensor,
        step_hat: Optional[torch.Tensor],
    ) -> Optional[float]:
        """Fast-recalibrate edited step reuse after a sparse exact forward."""
        phase = self.phase(step_index)
        observed_ratio = None
        if step_hat is not None:
            error = masked_relative_mse(
                step_hat,
                edited_exact,
                self.edited_mask,
            )
            observed_ratio = max(
                self.cfg.ratio_min,
                min(
                    self.cfg.ratio_max,
                    error / max(self.error_budget(step_index), 1e-12),
                ),
            )
            key = ("edited", "step", phase)
            previous = self.calibrator.ema.get(key, 1.0)
            self.calibrator.ema[key] = float(observed_ratio)
            if self.cfg.verbose:
                logging.info(
                    "AlignFlow token-kv fast-calibration turn=%d step=%d "
                    "phase=%d previous=%.6f observed=%.6f threshold=%.6f",
                    self.turn_id,
                    step_index,
                    phase,
                    previous,
                    observed_ratio,
                    self.region_threshold("edited"),
                )
        self.step_forecaster.add(step_index, self.progress(step_index), composed)
        self.last_exact_index = step_index
        return observed_ratio

    def mark_token_kv_recorded(self) -> None:
        self.stats["token_kv_recorded"] += 1

    def finish_step(
        self, decision: AlignFlowDecision, step_index: Optional[int] = None
    ) -> None:
        if step_index is not None:
            key = "exact_indices" if decision.needs_exact else "skip_indices"
            self.stats.setdefault(key, []).append(int(step_index))
        if decision.needs_exact:
            if decision.token_kv:
                self.stats["token_kv_used"] += 1
            else:
                self.stats["exact"] += 1
            if decision.reason == "budget_anchor":
                self.stats["anchor"] += 1
            if decision.reason == "edit_staleness_reset":
                self.stats["reset_exact"] += 1
        else:
            self.stats["skip"] += 1
        draft_was_composed = not (
            decision.needs_exact and not self.cfg.preserve_if_exact_available
        )
        if draft_was_composed and ((
            decision.edited_source == "step" and not self._region_is_empty("edited")
        ) or (
            decision.preserved_source == "step" and not self._region_is_empty("preserved")
        )):
            self.stats["step_used"] += 1
            self.stats[f"step_{self._last_step_variant}_used"] += 1
        if draft_was_composed and ((
            decision.edited_source == "edit" and not self._region_is_empty("edited")
        ) or (
            decision.preserved_source == "edit" and not self._region_is_empty("preserved")
        )):
            self.stats["edit_used"] += 1
            self.turn_used_cross_edit = True
            if step_index is not None:
                self.stats.setdefault("edit_indices", []).append(int(step_index))

    def finish_turn(self) -> None:
        if self.current_exact_entries:
            self.bank.store_turn(self.current_exact_entries, self.turn_id)
        if self.turn_used_cross_edit:
            self.bank.consecutive_propagated_turns += 1
        else:
            self.bank.consecutive_propagated_turns = 0
        if self.cfg.verbose:
            logging.info(
                "AlignFlow finish turn=%s stats=%s sigma=%.4f eta=%.4f",
                self.turn_id,
                self.stats,
                self.cross_forecaster.sigma,
                self.cross_forecaster.eta,
            )
        self.turn_id += 1

    def _infer_regions_from_probe(self, drift: torch.Tensor) -> None:
        energy = drift.detach().float().pow(2).mean(dim=0, keepdim=True)
        flat = energy.flatten()
        if flat.numel() == 0 or float(flat.max().item()) <= 0.0:
            return
        threshold = torch.quantile(flat, self.cfg.infer_mask_percentile)
        edited = (energy >= threshold).to(dtype=self.edited_mask.dtype)
        edited = self._dilate_token_mask(edited)
        if bool(edited.sum().item() <= 0.0) or bool(
            edited.sum().item() >= edited.numel()
        ):
            return
        self._set_region_masks(edited)
        self.stats["inferred_mask"] += 1
        if self.cfg.verbose:
            logging.info(
                "AlignFlow inferred mask turn=%d edited_ratio=%.6f percentile=%.3f",
                self.turn_id,
                float(edited.float().mean().item()),
                self.cfg.infer_mask_percentile,
            )

    def _monitor_preserved_activity(
        self, step_index: int, exact: torch.Tensor
    ) -> None:
        if self._region_is_empty("preserved") or not self.bank.has_entries():
            return
        prev = self.bank.lookup(
            self.progress(step_index), device=exact.device, dtype=exact.dtype
        )
        if prev is None:
            return
        numerator = (exact.detach().float() - prev.detach().float()).pow(2).sum(
            dim=0, keepdim=True
        )
        denominator = exact.detach().float().pow(2).sum(
            dim=0, keepdim=True
        ).clamp_min(1e-12)
        relative_energy = numerator / denominator
        preserved_tokens = self.preserved_mask[:1] > 0.5
        active = preserved_tokens & (
            relative_energy > self.cfg.token_activity_tolerance
        )
        preserved_count = int(preserved_tokens.sum().item())
        activity = float(active.sum().item()) / max(preserved_count, 1)
        trigger = max(self.cfg.preserved_activity_trigger, 1e-8)
        severity = min(1.0, activity / trigger)
        self.kappa_preserved_current = 1.0 + (
            float(self.cfg.kappa_preserved) - 1.0
        ) * (1.0 - self.cfg.preserved_kappa_contraction * severity)

        if self.cfg.dynamic_region_expansion and activity > trigger:
            current_edited = self.edited_mask[:1]
            boundary = self._dilate_token_mask(current_edited) > current_edited
            spill = active & boundary
            edited = torch.maximum(
                current_edited, spill.to(self.edited_mask.dtype)
            )
            if bool(torch.any(edited > self.edited_mask[:1]).item()):
                self._set_region_masks(edited)
                self.stats["region_expansions"] += 1

        if self.cfg.verbose:
            logging.info(
                "AlignFlow preserved activity turn=%d step=%d activity=%.6f "
                "kappa=%.6f revision=%d",
                self.turn_id,
                step_index,
                activity,
                self.kappa_preserved_current,
                self.region_revision,
            )

    def _dilate_token_mask(self, mask: torch.Tensor) -> torch.Tensor:
        radius = int(self.cfg.region_dilate)
        if radius <= 0:
            return mask
        return torch_F.max_pool3d(
            mask.unsqueeze(0),
            kernel_size=(1, radius * 2 + 1, radius * 2 + 1),
            stride=1,
            padding=(0, radius, radius),
        ).squeeze(0)

    def _set_region_masks(self, edited: torch.Tensor) -> None:
        channels = int(self.edited_mask.shape[0])
        edited = edited.clamp(0, 1).expand(
            channels, *edited.shape[1:]
        ).contiguous()
        self.edited_mask = edited
        self.preserved_mask = (1.0 - edited).contiguous()
        self.region_revision += 1

    def _forced_exact_reason(
        self,
        step_index: int,
        step_hat: Optional[torch.Tensor],
        edit_hat: Optional[torch.Tensor],
    ) -> Optional[str]:
        warmup_steps = self.cfg.warmup_steps
        if (
            self.cfg.global_warmup_steps is not None
            and not self.mask_was_provided
        ):
            warmup_steps = self.cfg.global_warmup_steps
        if step_index < warmup_steps:
            return "warmup"
        if step_index in self.probe_indices:
            return "probe"
        # Each phase has an independent error calibration.  Do not apply the
        # default prior to the first sample of a new phase: measure one exact
        # boundary anchor before allowing reuse in that phase.
        if (
            step_index > 0
            and self.phase(step_index) != self.phase(step_index - 1)
        ):
            return "budget_anchor"
        if (
            self.cfg.force_final_exact
            and step_index >= self.total_steps - self.cfg.final_exact_steps
        ):
            return "budget_anchor"
        if step_hat is None and edit_hat is None:
            return "no_cache_source"
        if step_index - self.last_exact_index >= self.cfg.min_exact_interval:
            return "budget_anchor"
        anchor_every = self.cfg.step_anchor_every
        anchor_offset = self.cfg.step_anchor_offset
        if (
            self.cfg.global_step_anchor_every is not None
            and not self.mask_was_provided
        ):
            anchor_every = self.cfg.global_step_anchor_every
        if (
            self.cfg.global_step_anchor_offset is not None
            and not self.mask_was_provided
        ):
            anchor_offset = self.cfg.global_step_anchor_offset
        # A forced probe/phase anchor already refreshes the predictor.  Restart
        # the periodic interval from that exact observation instead of paying
        # for a second, redundant anchor a step or two later.
        if (
            (step_index - anchor_offset) % anchor_every == 0
            and (
                # The warmup block is not an inserted adaptive anchor.  Keep
                # the first configured periodic anchor after warmup even when
                # its phase offset places it fewer than ``anchor_every`` steps
                # after the final warmup sample.
                self.last_exact_index < warmup_steps
                or step_index - self.last_exact_index >= anchor_every
            )
        ):
            return "budget_anchor"
        return None

    def _select_source(
        self,
        region: RegionName,
        phase: int,
        candidates: Sequence[SourceName],
    ) -> SourceName:
        if not candidates:
            return "exact"

        threshold = self.region_threshold(region)
        if (
            region == "edited"
            and self.cfg.global_kappa_edited is not None
            and not self.mask_was_provided
        ):
            threshold = (
                self.cfg.base_threshold * float(self.cfg.global_kappa_edited)
            )
        scores: Dict[SourceName, float] = {}
        eligible: List[SourceName] = []
        for source in candidates:
            key = (region, source, phase)
            # A cross-edit source is not trusted before this turn has measured
            # its counterfactual error at an exact anchor.
            if source == "edit":
                observations = self.calibrator.observations.get(key, 0)
                if observations < self.cfg.min_edit_observations:
                    continue
            score = self.calibrator.score(region, source, phase)
            scores[source] = score
            threshold_scale = self.calibrator.threshold_scale(region, source, phase)
            if threshold_scale >= 1.0 / max(threshold, 1e-12):
                eligible.append(source)

        selected: SourceName = "exact"
        if eligible:
            selected = min(eligible, key=lambda source: scores[source])
            if (
                region == "preserved"
                and "edit" in eligible
                and scores["edit"]
                <= scores[selected] * self.cfg.edit_preference_factor
            ):
                selected = "edit"

        if self.cfg.verbose:
            logging.info(
                "AlignFlow min-error-select turn=%d phase=%d selected=%s "
                "scores=%s threshold=%.6f",
                self.turn_id,
                phase,
                selected,
                {source: round(score, 6) for source, score in scores.items()},
                threshold,
            )
        return selected

    def _observe_source(
        self,
        region: RegionName,
        source: SourceName,
        phase: int,
        step_index: int,
        exact: torch.Tensor,
        draft: torch.Tensor,
    ) -> None:
        mask = self.preserved_mask if region == "preserved" else self.edited_mask
        err = masked_relative_mse(draft, exact, mask)
        self.calibrator.observe(
            region=region,
            source=source,
            phase=phase,
            error=err,
            budget=self.error_budget(step_index),
        )

    def _region_is_empty(self, region: RegionName) -> bool:
        mask = self.preserved_mask if region == "preserved" else self.edited_mask
        if mask is None:
            return False
        return bool(mask.detach().float().sum().item() <= 0.0)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.detach().float().flatten()
    right_f = right.detach().float().flatten()
    denom = left_f.norm() * right_f.norm()
    if denom.item() <= 1e-12:
        return 0.0
    return float(torch.dot(left_f, right_f) / denom)


def normalized_timestep_progress(
    timesteps: Sequence[torch.Tensor],
) -> List[float]:
    """Map scheduler timesteps to normalized denoising progress."""
    if len(timesteps) == 0:
        return []
    values = [
        float(torch.as_tensor(timestep).detach().float().flatten()[0].cpu())
        for timestep in timesteps
    ]
    if len(values) == 1:
        return [0.0]
    denominator = values[-1] - values[0]
    if abs(denominator) <= 1e-12:
        return [
            float(index) / float(len(values) - 1)
            for index in range(len(values))
        ]
    return [
        max(0.0, min(1.0, (value - values[0]) / denominator))
        for value in values
    ]


def masked_relative_mse(
    draft: torch.Tensor,
    exact: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> float:
    draft_f = draft.detach().float()
    exact_f = exact.detach().float()
    if mask is not None:
        mask_f = mask.detach().float()
        draft_f = draft_f * mask_f
        exact_f = exact_f * mask_f
    denom = exact_f.pow(2).sum().clamp_min(1e-12)
    num = (draft_f - exact_f).pow(2).sum()
    return float((num / denom).item())


def least_squares_blend_alpha(
    step: torch.Tensor,
    edit: torch.Tensor,
    exact: torch.Tensor,
    mask: Optional[torch.Tensor],
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> float:
    """Fit step + alpha * (edit - step) to an exact guided anchor."""
    step_f = step.detach().float()
    direction = edit.detach().float() - step_f
    target = exact.detach().float() - step_f
    if mask is not None:
        mask_f = mask.detach().float()
        direction = direction * mask_f
        target = target * mask_f
    denominator = direction.pow(2).sum()
    if denominator.item() <= 1e-12:
        return float(alpha_min)
    alpha = float((direction * target).sum() / denominator)
    return max(float(alpha_min), min(float(alpha_max), alpha))


def build_region_masks(
    input_mask: Optional[torch.Tensor],
    target_shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    dilate_radius: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    channels, depth, height, width = [int(x) for x in target_shape]
    if input_mask is None:
        edited = torch.ones(1, depth, height, width, device=device, dtype=dtype)
    else:
        mask = input_mask.detach().float()
        if mask.ndim == 5:
            mask = mask[0]
        if mask.ndim == 4:
            mask = mask.mean(dim=0, keepdim=True)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported mask shape for AlignFlow: {tuple(mask.shape)}")
        mask = mask.unsqueeze(0).to(device=device)
        edited = torch_F.interpolate(
            mask,
            size=(depth, height, width),
            mode="nearest",
        ).squeeze(0)
        edited = (edited > 0.5).to(dtype=dtype)

    if dilate_radius > 0:
        radius = int(dilate_radius)
        edited = torch_F.max_pool3d(
            edited.unsqueeze(0),
            kernel_size=(1, radius * 2 + 1, radius * 2 + 1),
            stride=1,
            padding=(0, radius, radius),
        ).squeeze(0)

    edited = edited.clamp(0, 1).expand(channels, depth, height, width).contiguous()
    preserved = (1.0 - edited).contiguous()
    return edited, preserved


def parse_probe_indices(values: Optional[Iterable[int]], total_steps: int) -> List[int]:
    if values is None:
        return []
    result = []
    for value in values:
        idx = int(value)
        if 0 <= idx < total_steps:
            result.append(idx)
    return sorted(set(result))
