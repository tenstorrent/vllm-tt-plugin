# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks host logprob payloads through decode submission, finalization, and packing.

The real controller and runner cooperate with a host-output model double.
The root host-suite fixture restores runtime patches and platform state.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.support.logprobs import ReportedLogprobs
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.model_input import TTModelInput, TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner


class HostOutputModel:
    """Returns host tokens and logprobs without device readback or processing hooks."""

    decode_input_update_contract = 1
    model_capabilities = {"supports_async_decode": True}

    def __init__(
        self,
        token_ids: torch.Tensor,
        logprobs: tuple[torch.Tensor, torch.Tensor] | ReportedLogprobs,
    ) -> None:
        """Retains the host output to return from each decode call."""
        self.token_ids = token_ids
        self.logprobs = logprobs

    def decode_forward(
        self, **kwargs: object
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | ReportedLogprobs]:
        """Returns the stored host output for the submitted decode."""
        return self.token_ids, self.logprobs


class TestHostDecodeOutput:
    """Checks nested host payloads through synchronous and async-read submissions."""

    @pytest.fixture
    def model_input(self) -> TTModelInput:
        """Supplies one ordinary decode row requesting its sampled token's logprob."""
        page_table = torch.tensor([[0]], dtype=torch.int32)
        return TTModelInput(
            input_tokens=torch.zeros((1, 1), dtype=torch.int32),
            input_positions=torch.zeros(1, dtype=torch.int32),
            prompt_lens=None,
            block_tables=page_table,
            block_tables_per_group=[page_table],
            block_tables_per_layer=None,
            unpadded_batch_size=1,
            tt_sampling_params=TTSamplingParams(
                temperature=torch.ones(1),
                top_k=torch.full((1,), 32),
                top_p=torch.ones(1),
                presence_penalty=torch.zeros(1),
                frequency_penalty=torch.zeros(1),
                repetition_penalty=torch.ones(1),
                seed=torch.full((1,), -1),
                num_logprobs=torch.zeros(1, dtype=torch.int64),
                enable_log_probs=torch.ones(1, dtype=torch.bool),
            ),
            multi_modal_kwargs={},
            perform_device_sampling=True,
            grammar_bitmask=[None],
            logitsprocs_list=[None],
            bad_words_token_ids_list=[{}],
            allowed_token_ids_mask_list=[None],
            generators_list=[{}],
            max_num_logprobs=[0],
            decode_layout_changed=True,
            slot_remap=torch.tensor([0], dtype=torch.int32),
        )

    @pytest.mark.parametrize("async_read", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "payload, rank",
        [
            pytest.param(
                (
                    torch.tensor([[-0.5, -1.5, -2.0]]),
                    torch.tensor([[41, 42, 43]], dtype=torch.int32),
                ),
                2,
                id="topk",
            ),
            pytest.param(
                ReportedLogprobs(torch.tensor([-1.5]), torch.tensor([7])),
                7,
                id="reported-rank",
            ),
        ],
    )
    def test_payload_rank_reaches_packing(
        self,
        model_input: TTModelInput,
        payload: tuple[torch.Tensor, torch.Tensor] | ReportedLogprobs,
        rank: int,
        async_read: bool,
    ) -> None:
        """Preserves token, score, and rank across controller-to-runner boundaries."""
        controller = TTAsyncDecodeController(
            SimpleNamespace(
                model=HostOutputModel(torch.tensor([[42]], dtype=torch.int32), payload),
                trace_mode="decode_only",
                kv_caches=object(),
                request_specific_rope=False,
                supports_stage_outputs=False,
                note_decode_layout_consumed=lambda: None,
                note_decode_state_slots_settled=lambda: None,
            )
        )

        submission = controller.submit_decode(
            model_input, read_from_device=not async_read, async_read=async_read
        )
        finalized = controller.finalize_decode(submission)
        runner = SimpleNamespace(
            _num_speculative_tokens=0,
            _output_tokens_per_step=1,
            _is_block_output_model=False,
            _is_adaptive_block_output=False,
            tt_per_lane_max_num_seqs=1,
        )
        _, logprobs_per_dp = TTModelRunner._get_output_tokens(
            runner,
            finalized.tt_out,
            finalized.tt_log_probs,
            submission.sampling_params,
            model_input,
            submission.batch_size_per_dp,
            perform_device_sampling=True,
            is_decode=True,
        )

        (logprobs,) = logprobs_per_dp
        assert logprobs.logprob_token_ids.tolist() == [[42]]
        assert logprobs.logprobs.tolist() == [[-1.5]]
        assert logprobs.selected_token_ranks.tolist() == [rank]
