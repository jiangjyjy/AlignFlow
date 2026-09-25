from __future__ import annotations

import gc
import logging
import math
import random
import sys
import time
from contextlib import contextmanager
from typing import Mapping, Optional, Sequence

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .alignflow import AlignFlowConfig, AlignFlowController
from .text2video import (
    FlowDPMSolverMultistepScheduler,
    FlowUniPCMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .token_kv_cache_int8 import (
    TokenKVCacheBank,
    TokenKVContext,
    token_indices_from_latent_mask,
)
from .vace import WanVace


class WanVaceAlignFlow(WanVace):
    """Wan2.1 VACE pipeline with AlignFlow inference-time cache scheduling.

    This class keeps the official WanVace preprocessing/model loading path and
    replaces only the denoising loop. A single Python process must reuse the
    same WanVaceAlignFlow instance for cross-edit caching to be useful.
    """

    def _load_negative_context_cache(
        self, prompt: str, device: torch.device
    ) -> Optional[list[torch.Tensor]]:
        cache = getattr(self, "_negative_context_cache", None)
        if cache is None or prompt not in cache:
            return None
        return [tensor.to(device=device, copy=True) for tensor in cache[prompt]]

    def _store_negative_context_cache(
        self, prompt: str, context: Sequence[torch.Tensor]
    ) -> None:
        if not hasattr(self, "_negative_context_cache"):
            self._negative_context_cache = {}
        self._negative_context_cache[prompt] = tuple(
            tensor.detach().to(device="cpu", copy=True) for tensor in context
        )

    def _get_alignflow_controller(
        self,
        alignflow: Optional[bool | Mapping[str, object] | AlignFlowConfig],
        session: str,
    ) -> Optional[AlignFlowController]:
        if not alignflow:
            return None
        if not hasattr(self, "_alignflow_sessions"):
            self._alignflow_sessions = {}

        if isinstance(alignflow, AlignFlowConfig):
            cfg = alignflow
        elif isinstance(alignflow, Mapping):
            cfg = AlignFlowConfig.from_mapping(alignflow)
        else:
            cfg = AlignFlowConfig()

        if not cfg.enabled:
            return None

        key = session or "default"
        controller = self._alignflow_sessions.get(key)
        if controller is None:
            controller = AlignFlowController(cfg)
            self._alignflow_sessions[key] = controller
        else:
            controller.cfg = cfg
        return controller

    def clear_alignflow_session(self, session: str = "default") -> None:
        if hasattr(self, "_alignflow_sessions"):
            self._alignflow_sessions.pop(session, None)
        if hasattr(self, "_token_kv_sessions"):
            self._token_kv_sessions.pop(session, None)

    def _get_token_kv_bank(self, session: str) -> TokenKVCacheBank:
        if not hasattr(self, "_token_kv_sessions"):
            self._token_kv_sessions = {}
        key = session or "default"
        bank = self._token_kv_sessions.get(key)
        if bank is None:
            bank = TokenKVCacheBank(cache_device=self.device)
            self._token_kv_sessions[key] = bank
        return bank

    def generate(
        self,
        input_prompt,
        input_frames,
        input_masks,
        input_ref_images,
        size=(1280, 720),
        frame_num=81,
        context_scale=1.0,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        alignflow: Optional[bool | Mapping[str, object] | AlignFlowConfig] = None,
        alignflow_session: str = "default",
        alignflow_auto_mask: bool = False,
        token_kv: bool = True,
        token_kv_steps: Sequence[int] = (6, 7, 9, 11, 12, 16, 21, 29),
        token_kv_refresh_every: int = 3,
        token_kv_max_uses_per_turn: int = 4,
        token_kv_max_score_factor: float = 2.0,
        token_kv_allow_anchor: bool = False,
    ):
        r"""Generate a video with optional AlignFlow cache acceleration.

        Extra args:
            alignflow: False/None disables AlignFlow. True uses default config.
            alignflow_auto_mask: keep VACE's generation mask unchanged, but let
                AlignFlow infer routing regions from exact probe residuals.
                A dict or AlignFlowConfig overrides controller hyperparameters.
            alignflow_session: cache key for a continuous multi-turn edit chain.
        """

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        context_null = self._load_negative_context_cache(n_prompt, self.device)
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if context_null is None:
                context_null = self.text_encoder([n_prompt], self.device)
                self._store_negative_context_cache(n_prompt, context_null)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            if context_null is None:
                context_null = self.text_encoder([n_prompt], torch.device("cpu"))
                self._store_negative_context_cache(n_prompt, context_null)
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)
        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        ) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)
        self.model.to(self.device)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps,
                    device=self.device,
                    shift=shift,
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas,
                )
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise
            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}

            controller = self._get_alignflow_controller(
                alignflow=alignflow,
                session=alignflow_session,
            )
            if controller is not None:
                controller.begin_turn(
                    timesteps=timesteps,
                    input_mask=(
                        None
                        if alignflow_auto_mask
                        else (input_masks[0] if input_masks else None)
                    ),
                    target_shape=target_shape,
                    device=self.device,
                    dtype=torch.float32,
                )

            token_kv_bank = None
            token_edited_indices = None
            token_preserved_indices = None
            token_kv_steps = {int(value) for value in token_kv_steps}
            token_kv_refresh_every = max(1, int(token_kv_refresh_every))
            token_kv_max_uses_per_turn = max(
                0, int(token_kv_max_uses_per_turn)
            )
            token_kv_max_score_factor = max(
                1.0, float(token_kv_max_score_factor)
            )
            token_kv_uses = 0
            token_kv_refresh_turn = False
            token_kv_region_revision = -1
            if controller is not None and token_kv:
                token_kv_bank = self._get_token_kv_bank(alignflow_session)
                token_kv_refresh_turn = (
                    controller.turn_id == 0
                    or not token_kv_bank.entries
                    or controller.turn_id % token_kv_refresh_every == 0
                )
                if token_kv_refresh_turn:
                    token_kv_bank.clear()
                try:
                    token_edited_indices, token_preserved_indices = (
                        token_indices_from_latent_mask(
                            controller.edited_mask,
                            self.patch_size,
                        )
                    )
                    token_kv_region_revision = controller.region_revision
                except ValueError as exc:
                    logging.info(
                        "AlignFlow branch-residual sparse path disabled for turn=%d: %s",
                        controller.turn_id,
                        exc,
                    )
                    token_kv_bank = None

            def token_kv_context(mode, branch, step_index):
                if (
                    token_kv_bank is None
                    or controller.region_revision != token_kv_region_revision
                ):
                    return None
                return TokenKVContext(
                    bank=token_kv_bank,
                    mode=mode,
                    branch=branch,
                    step_index=step_index,
                    edited_indices=token_edited_indices,
                    preserved_indices=token_preserved_indices,
                    full_seq_len=seq_len,
                )

            dit_forward_events = []
            dit_forward_cpu_seconds = 0.0

            def timed_model_forward(**kwargs):
                nonlocal dit_forward_cpu_seconds
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    result = self.model(**kwargs)
                    end_event.record()
                    dit_forward_events.append((start_event, end_event))
                    return result
                started = time.perf_counter()
                result = self.model(**kwargs)
                dit_forward_cpu_seconds += time.perf_counter() - started
                return result

            def forward_cond(latent_model_input, timestep, kv_context=None):
                return timed_model_forward(
                    x=latent_model_input,
                    t=timestep,
                    vace_context=z,
                    vace_context_scale=context_scale,
                    token_kv_context=kv_context,
                    **arg_c,
                )[0]

            def forward_uncond(latent_model_input, timestep, kv_context=None):
                return timed_model_forward(
                    x=latent_model_input,
                    t=timestep,
                    vace_context=z,
                    vace_context_scale=context_scale,
                    token_kv_context=kv_context,
                    **arg_null,
                )[0]

            def combine_guided(noise_pred_cond, noise_pred_uncond):
                return noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            def forward_guided(latent_model_input, timestep):
                return combine_guided(
                    forward_cond(latent_model_input, timestep),
                    forward_uncond(latent_model_input, timestep),
                )

            denoising_cpu_started = time.perf_counter()
            denoising_start_event = None
            denoising_end_event = None
            if torch.cuda.is_available():
                denoising_start_event = torch.cuda.Event(enable_timing=True)
                denoising_end_event = torch.cuda.Event(enable_timing=True)
                denoising_start_event.record()

            for step_index, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = torch.stack([t])

                if controller is None:
                    noise_pred = forward_guided(latent_model_input, timestep)
                else:
                    step_hat, edit_hat = controller.make_drafts(
                        step_index=step_index,
                        device=self.device,
                        dtype=latents[0].dtype,
                        latent_input=latent_model_input[0],
                        guide_scale=guide_scale,
                    )
                    decision = controller.decide(
                        step_index=step_index,
                        step_hat=step_hat,
                        edit_hat=edit_hat,
                    )
                    exact = None
                    token_kv_ready = False
                    # Budget anchors must remain full exact forwards because they
                    # provide the counterfactual error used by the calibrator.
                    token_kv_anchor = False
                    if (
                        token_kv_bank is not None
                        and controller.region_revision == token_kv_region_revision
                        and controller.turn_id > 0
                        and not token_kv_refresh_turn
                        and decision.needs_exact
                        and (
                            decision.reason == "calibrated_exact"
                            or token_kv_anchor
                        )
                        and decision.edited_source == "exact"
                        and (
                            decision.preserved_source != "exact"
                            or token_kv_anchor
                        )
                        and step_index in token_kv_steps
                        and token_kv_uses < token_kv_max_uses_per_turn
                        and controller.calibrator.score(
                            "edited",
                            "step",
                            controller.phase(step_index),
                        )
                        <= controller.region_threshold("edited")
                        * token_kv_max_score_factor
                    ):
                        layer_count = len(self.model.blocks) + len(
                            self.model.vace_blocks
                        )
                        preserved_count = token_preserved_indices.numel()
                        token_kv_ready = all(
                            token_kv_bank.has_step(
                                branch=branch,
                                step_index=step_index,
                                num_layers=layer_count,
                                full_seq_len=seq_len,
                                preserved_count=preserved_count,
                            )
                            for branch in ("cond", "uncond")
                        )

                    if token_kv_ready:
                        if token_kv_anchor:
                            # Keep the current exact forward for edited tokens,
                            # but route preserved tokens through the step draft.
                            decision.preserved_source = "step"
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        exact_cond = forward_cond(
                            latent_model_input,
                            timestep,
                            token_kv_context("reuse", "cond", step_index),
                        )
                        exact_uncond = forward_uncond(
                            latent_model_input,
                            timestep,
                            token_kv_context("reuse", "uncond", step_index),
                        )
                        exact = combine_guided(exact_cond, exact_uncond)
                        decision.reason = "token_kv"
                        decision.update_calibrator = False
                        decision.token_kv = True
                        token_kv_uses += 1
                        noise_pred = controller.compose(
                            decision=decision,
                            exact=exact,
                            step_hat=step_hat,
                            edit_hat=edit_hat,
                        )
                        token_score = controller.observe_token_exact(
                            step_index=step_index,
                            composed=noise_pred,
                            edited_exact=exact,
                            step_hat=step_hat,
                        )
                        end_event.record()
                        torch.cuda.synchronize()
                        logging.info(
                            "AlignFlow pre-V2 token-kv use turn=%d step=%d "
                            "preserved_source=%s use_index=%d/%d "
                            "edited_score=%s edited_tokens=%d "
                            "preserved_tokens=%d cuda_ms=%.3f cache_gib=%.3f",
                            controller.turn_id,
                            step_index,
                            decision.preserved_source,
                            token_kv_uses,
                            token_kv_max_uses_per_turn,
                            "none" if token_score is None else f"{token_score:.6f}",
                            token_edited_indices.numel(),
                            token_preserved_indices.numel(),
                            start_event.elapsed_time(end_event),
                            token_kv_bank.memory_bytes() / (1024**3),
                        )
                    elif decision.needs_exact:
                        record_token_kv = (
                            token_kv_bank is not None
                            and controller.region_revision == token_kv_region_revision
                            and token_kv_refresh_turn
                            and step_index in token_kv_steps
                        )
                        exact_cond = forward_cond(
                            latent_model_input,
                            timestep,
                            token_kv_context("record", "cond", step_index)
                            if record_token_kv
                            else None,
                        )
                        exact_uncond = forward_uncond(
                            latent_model_input,
                            timestep,
                            token_kv_context("record", "uncond", step_index)
                            if record_token_kv
                            else None,
                        )
                        exact = combine_guided(exact_cond, exact_uncond)
                        controller.observe_exact(
                            step_index=step_index,
                            exact=exact,
                            step_hat=step_hat,
                            edit_hat=edit_hat,
                            update_calibrator=decision.update_calibrator,
                            monitor_regions=decision.reason == "budget_anchor",
                            exact_cond=exact_cond,
                            exact_uncond=exact_uncond,
                            latent_input=latent_model_input[0],
                        )
                        noise_pred = controller.compose(
                            decision=decision,
                            exact=exact,
                            step_hat=step_hat,
                            edit_hat=edit_hat,
                        )
                        if record_token_kv:
                            controller.mark_token_kv_recorded()
                            logging.info(
                                "AlignFlow pre-V2 token-kv record turn=%d step=%d "
                                "cache_gib=%.3f",
                                controller.turn_id,
                                step_index,
                                token_kv_bank.memory_bytes() / (1024**3),
                            )
                    else:
                        noise_pred = controller.compose(
                            decision=decision,
                            exact=exact,
                            step_hat=step_hat,
                            edit_hat=edit_hat,
                        )
                    controller.finish_step(decision, step_index)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents = [temp_x0.squeeze(0)]

            if torch.cuda.is_available():
                denoising_end_event.record()
                torch.cuda.synchronize()
                self._last_denoising_loop_seconds = (
                    denoising_start_event.elapsed_time(denoising_end_event) / 1000.0
                )
                self._last_dit_forward_seconds = sum(
                    start.elapsed_time(end) for start, end in dit_forward_events
                ) / 1000.0
            else:
                self._last_denoising_loop_seconds = (
                    time.perf_counter() - denoising_cpu_started
                )
                self._last_dit_forward_seconds = dit_forward_cpu_seconds
            self._last_dit_forward_calls = len(dit_forward_events)

            x0 = latents
            if controller is not None:
                controller.finish_turn()

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
