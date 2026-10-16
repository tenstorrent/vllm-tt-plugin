# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request side outputs of a pipeline-stage model's decode steps.

A model that declares ``supports_stage_outputs`` implements
``pop_stage_outputs()``. The runner calls it right after every
``decode_forward``, so each step's outputs stay attached to the submission
that produced them. It returns one entry per submitted decode row: the stage
frame this forward completed for that row's request, or ``None``. The runner
tags each stage frame with its request and ``frame_index`` and publishes them on
the step's ``ModelRunnerOutput``.

"Stage frame" is the model's per-request payload, distinct from the output
frames ``TTScheduler`` counts per published step.

``frame_index`` comes from the row's decode position: the step that consumes
output token ``k`` (position ``num_prompt_tokens + k``) completes stage frame
``k``. Positions are current because async decode is refused for these models.
Prefill completes no stage frame and is never asked.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput

    from vllm_tt_plugin.model_input import TTModelInput

_TT_STAGE_OUTPUTS_ATTR = "_tt_stage_outputs"


@dataclass(frozen=True)
class TTStageFrame:
    """One completed frame of one request, opaque to the plugin."""

    frame_index: int
    data: Any


def pop_tt_stage_outputs(
    model: Any, model_input: "TTModelInput", requests: Mapping[str, Any]
) -> dict[str, TTStageFrame]:
    """Collect the frames the decode just submitted completed, by request ID.

    Raises:
        ValueError: the build carries no row request IDs, the model returned
            fewer entries than live rows, or a padding row carries a frame.
    """
    row_req_ids = model_input.row_req_ids
    if row_req_ids is None:
        raise ValueError(
            "stage outputs need the decode build's row request IDs; lane-DP "
            "builds do not carry them"
        )
    rows = list(model.pop_stage_outputs())
    if len(rows) < len(row_req_ids):
        raise ValueError(
            f"{type(model).__name__}.pop_stage_outputs() returned {len(rows)} "
            f"entries for {len(row_req_ids)} decode rows {row_req_ids}"
        )
    padding = [
        row for row in range(len(row_req_ids), len(rows)) if rows[row] is not None
    ]
    if padding:
        raise ValueError(
            f"{type(model).__name__}.pop_stage_outputs() returned frames for "
            f"padding rows {padding}; live rows are 0..{len(row_req_ids) - 1}"
        )
    positions = model_input.input_positions
    frames: dict[str, TTStageFrame] = {}
    for row, req_id in enumerate(row_req_ids):
        if rows[row] is None:
            continue
        frame_index = int(positions[row]) - requests[req_id].num_prompt_tokens
        if frame_index < 0:
            raise ValueError(
                f"request {req_id} decodes at position {int(positions[row])}, "
                f"inside its {requests[req_id].num_prompt_tokens}-token prompt"
            )
        frames[req_id] = TTStageFrame(frame_index=frame_index, data=rows[row])
    return frames


def set_tt_stage_outputs(
    runner_output: "ModelRunnerOutput", frames: Mapping[str, TTStageFrame] | None
) -> None:
    if frames:
        setattr(runner_output, _TT_STAGE_OUTPUTS_ATTR, dict(frames))


def get_tt_stage_outputs(runner_output: "ModelRunnerOutput") -> dict[str, TTStageFrame]:
    """The frames a step published, by request ID; empty when it published none."""
    return dict(getattr(runner_output, _TT_STAGE_OUTPUTS_ATTR, {}))
