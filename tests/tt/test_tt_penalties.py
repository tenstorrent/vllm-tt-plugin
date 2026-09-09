# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from tests.tt.utils import (
    RequestConfig,
    assert_deterministic_allow_near_tie,
    assert_varied,
    count_prompts_changed_by,
    run_concurrent_batch,
)


class TestRepetitionPenalty:
    """
    Different repetition penalties per request in same batch.
    """

    def test_different_repetition_penalties(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Each request has different repetition penalty.
        """
        prompt = "a a a a a a a a a"
        penalties = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0][:max_batch_size]

        # Requests otherwise reproducible
        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=10,
                temperature=0,
                repetition_penalty=penalty,
            )
            for penalty in penalties
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_varied(
            results, 2, "Varying penalty should at least influence the output somewhat."
        )

    # Caught https://github.com/tenstorrent/vllm/issues/286
    def test_repetition_penalty_mixed_batch(
        self, tt_server, tt_model_name, max_batch_size
    ):
        prompt = "a a a a a a a a a"

        configs = []
        for i in range(max_batch_size):
            if i % 2 == 0:
                # No penalty
                configs.append(
                    RequestConfig(
                        prompt=prompt,
                        max_tokens=10,
                        temperature=0,
                        repetition_penalty=1.0,
                    )
                )
            else:
                # High penalty
                configs.append(
                    RequestConfig(
                        prompt=prompt,
                        max_tokens=10,
                        temperature=0,
                        repetition_penalty=2.0,
                    )
                )

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = [results[i] for i in range(0, max_batch_size, 2)]
        with_penalty = [results[i] for i in range(1, max_batch_size, 2)]

        assert_deterministic_allow_near_tie(
            no_penalty, "No penalty requests should be identical."
        )
        assert_deterministic_allow_near_tie(
            with_penalty, "With penalty requests should be identical."
        )
        assert_varied(
            [no_penalty[0], with_penalty[0]], 2, "Penalty should change output."
        )


class TestPresencePenalty:
    """
    Different presence penalties per request.
    """

    # Presence contributes a FIXED -P to any token already emitted, so it only
    # changes the sampled text when P exceeds the model's top-2 logit gap at a
    # repeat. vLLM hard-clamps presence_penalty to +/-2.0, so a SINGLE-prompt text
    # assertion is really a bet on the model being under 2.0 nats confident on that
    # one prompt -- a property of the model, not of the penalty. On Qwen3-32B the
    # old "a b c a b c" prompt drives a high-confidence loop (gap ~2.4) and the test
    # could never pass; on Llama the same prompt sits lower and it did. Frequency
    # has no such problem because count*P grows without bound.
    #
    # Asserting on logprobs instead does not work either: this server reports
    # logprobs computed from PRE-penalty logits, so a correctly applied penalty
    # leaves every reported logprob unchanged (see utils.count_prompts_changed_by).
    #
    # So: assert over a SET of prompts. A working penalty moves most of them; a
    # broken one moves none; no single prompt's confidence decides the outcome.

    # Deliberately mixed: several genuinely ambiguous continuations, plus two
    # high-confidence degenerate loops that presence cannot be expected to break.
    PRESENCE_PROMPTS = [
        "She opened the door and",
        "The reason is that the",
        "He said that the",
        "I think the answer is maybe",
        "After a while, she",
        "It was a",
        "The book was",
        "a b c a b c a b c",
    ]

    def test_presence_penalty_changes_output(self, tt_server, tt_model_name):
        """Presence must alter greedy output on most of a diverse prompt set.

        Threshold is a bare majority, well under the observed rate, so ordinary
        model-to-model variation in confidence cannot fail it -- but a penalty that
        is not applied at all scores 0 and fails loudly.

        STOPGAP. Presence is structurally the wrong penalty for a "the penalty
        reached the sampler" claim: it adds a FIXED -P once, so any prompt whose
        top-2 logit gap exceeds P is immune no matter how correct the plumbing is,
        which is why this has to be a majority vote over a prompt set rather than
        an assertion. test_frequency_penalty_changes_output below carries that
        claim properly -- count * F is unbounded and clears any gap. Keep this one
        only as long as it is cheap; the real presence math is asserted against
        controlled logits in tt-metal models/common/tests/test_tt_sampling.py
        (TestPresencePenaltyPerRequest), where the gap is fixed at 1.0 nat.
        """
        changed, slot_noise, detail = count_prompts_changed_by(
            tt_server,
            tt_model_name,
            self.PRESENCE_PROMPTS,
            presence_penalty=2.0,
        )
        total = len(self.PRESENCE_PROMPTS)
        minimum = total // 2
        # The control run measures how many prompts disagree with THEMSELVES across
        # the two slots of the same batch. That is the floor the penalty has to clear
        # before "changed" means the penalty did anything.
        assert changed >= minimum, (
            f"presence_penalty=2.0 changed greedy output on only {changed}/{total} "
            f"prompts (expected at least {minimum}).\n"
            f"A penalty that is never applied scores 0. A low-but-nonzero score means "
            f"the model is unusually confident and the prompt set needs refreshing, "
            f"not that sampling is broken.\n" + "\n".join(detail)
        )
        assert changed > slot_noise, (
            f"presence_penalty=2.0 changed {changed}/{total} prompts, but the "
            f"unpenalised control disagreed with itself on {slot_noise}/{total}. "
            f"The 'changed' count is not distinguishable from slot noise, so this "
            f"test is not evidence the penalty reached the sampler.\n"
            + "\n".join(detail)
        )

    def test_presence_penalty_is_per_request(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """Penalised and unpenalised requests in one batch must not affect each other.

        This is the isolation half of the old mixed-batch test: it still catches a
        penalty leaking across slots, without requiring the penalty to be strong
        enough to flip an argmax.
        """
        prompt = "a b c a b c a b c"
        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=20,
                temperature=0,
                presence_penalty=0.0 if i % 2 == 0 else 2.0,
            )
            for i in range(max_batch_size)
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = [results[i] for i in range(0, max_batch_size, 2)]
        with_penalty = [results[i] for i in range(1, max_batch_size, 2)]

        assert_deterministic_allow_near_tie(
            no_penalty, "Unpenalised requests in a mixed batch should agree."
        )
        assert_deterministic_allow_near_tie(
            with_penalty, "Penalised requests in a mixed batch should agree."
        )

        # Cross-check by varying ONLY the neighbours. Comparing against the same
        # request run alone would vary batch WIDTH too, and width changes the
        # arithmetic on its own (batch composition decides near-ties), so a
        # difference there would be uninterpretable. Instead: same width, same
        # prompt, same slot 0 -- one batch fully unpenalised, one with slot 0
        # unpenalised and every neighbour penalised. Any difference in slot 0 is a
        # leak.
        #
        # This also exercises the batch-wide no_penalties gate in the plugin
        # (model_runner.py, an .all() reduction): the first batch skips the penalty
        # kernel entirely, the second runs it with a 0.0 scalar on row 0. A failure
        # means either a real cross-slot leak or a penalty kernel that is not
        # exactly identity at P=0 in bf8. Both are findings.
        all_clean = run_concurrent_batch(
            tt_server,
            tt_model_name,
            [
                RequestConfig(prompt=prompt, max_tokens=20, temperature=0)
                for _ in range(max_batch_size)
            ],
        )
        neighbours_penalised = run_concurrent_batch(
            tt_server,
            tt_model_name,
            [
                RequestConfig(
                    prompt=prompt,
                    max_tokens=20,
                    temperature=0,
                    presence_penalty=0.0 if i == 0 else 2.0,
                )
                for i in range(max_batch_size)
            ],
        )
        assert_deterministic_allow_near_tie(
            [all_clean[0], neighbours_penalised[0]],
            "Slot 0 is unpenalised in both batches, which have the same width and "
            "prompt and differ only in whether its NEIGHBOURS are penalised. Its "
            "output must not change.",
        )


class TestFrequencyPenalty:
    """
    Different frequency penalties per request.
    """

    def test_frequency_penalty_changes_output(self, tt_server, tt_model_name):
        """Frequency penalty must alter greedy output on the whole prompt set.

        This is the "the penalty reaches the sampler" assertion. Unlike presence,
        frequency subtracts count * F, which grows without bound as a token repeats,
        so no top-2 gap can hold out over a 24-token greedy continuation. That lets
        this assert on EVERY prompt rather than voting, and it fails loudly if the
        penalty never reaches the device.
        """
        prompts = TestPresencePenalty.PRESENCE_PROMPTS
        changed, slot_noise, detail = count_prompts_changed_by(
            tt_server,
            tt_model_name,
            prompts,
            frequency_penalty=2.0,
        )
        total = len(prompts)
        assert changed > slot_noise, (
            f"frequency_penalty=2.0 changed {changed}/{total} prompts but the "
            f"unpenalised control disagreed with itself on {slot_noise}/{total}; "
            f"not distinguishable from slot noise.\n" + "\n".join(detail)
        )
        assert changed >= total - slot_noise, (
            f"frequency_penalty=2.0 changed greedy output on only {changed}/{total} "
            f"prompts. count*F is unbounded, so every prompt should move unless the "
            f"penalty is not reaching the sampler "
            f"(control slot noise: {slot_noise}/{total}).\n" + "\n".join(detail)
        )

    def test_different_frequency_penalties(
        self, tt_server, tt_model_name, max_batch_size
    ):
        prompt = "5 5 5 5 5 5 5 5"

        penalties = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0][:max_batch_size]

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=20,
                temperature=0,
                frequency_penalty=penalty,
            )
            for penalty in penalties
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert_varied(
            results,
            2,
            "Different frequency penalties should produce different outputs.",
        )

    def test_frequency_penalty_mixed_batch(
        self, tt_server, tt_model_name, max_batch_size
    ):
        prompt = "a a a a a a a a a"

        configs = []
        for i in range(max_batch_size):
            configs.append(
                RequestConfig(
                    prompt=prompt,
                    max_tokens=15,
                    temperature=0,
                    frequency_penalty=0.0 if i % 2 == 0 else 2.0,
                )
            )

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = [results[i] for i in range(0, max_batch_size, 2)]
        with_penalty = [results[i] for i in range(1, max_batch_size, 2)]

        # Count "a"s in each output
        no_penalty_a_count = no_penalty[0].count("a")
        with_penalty_a_count = with_penalty[0].count("a")

        assert no_penalty_a_count > with_penalty_a_count, (
            f"Frequency penalty should reduce 'a' repetitions: "
            f"no_penalty={no_penalty_a_count},"
            f"with_penalty={with_penalty_a_count}"
        )
        assert_deterministic_allow_near_tie(
            no_penalty, "No penalty requests should be identical."
        )
        assert_deterministic_allow_near_tie(
            with_penalty, "With penalty requests should be identical."
        )
