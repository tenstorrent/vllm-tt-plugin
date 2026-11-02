# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TT_DETOK_PRIME_TOKENS: the streaming detokenizer's DecodeStream primed with the
prompt's tail (detok_prime)."""

import random

import pytest
import tokenizers.decoders
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from vllm_tt_plugin import detok_prime
from vllm_tt_plugin.detok_prime import BoundedDecodeStream


@pytest.fixture(autouse=True)
def stock_after():
    yield
    detok_prime.uninstall()
    assert tokenizers.decoders.DecodeStream is not BoundedDecodeStream


def test_value():
    assert detok_prime.value({}) == 0 and not detok_prime.enabled({})
    assert detok_prime.value({"TT_DETOK_PRIME_TOKENS": "0"}) == 0
    assert detok_prime.value(
        {"TT_DETOK_PRIME_TOKENS": " 16 "}
    ) == 16 and detok_prime.enabled({"TT_DETOK_PRIME_TOKENS": "16"})
    for bad in ("-1", "x", "1.5", "3"):  # 3: below a character's ids
        with pytest.raises(ValueError, match="TT_DETOK_PRIME_TOKENS"):
            detok_prime.value({"TT_DETOK_PRIME_TOKENS": bad})
    assert detok_prime.value({"TT_DETOK_PRIME_TOKENS": "4"}) == 4


def test_install_is_idempotent_and_reversible():
    stock = tokenizers.decoders.DecodeStream
    assert detok_prime.install({}) == 0 and tokenizers.decoders.DecodeStream is stock
    assert detok_prime.install({"TT_DETOK_PRIME_TOKENS": "8"}) == 8
    assert (
        tokenizers.decoders.DecodeStream is BoundedDecodeStream
        and BoundedDecodeStream.bound == 8
    )
    assert BoundedDecodeStream.stock is stock
    assert (
        detok_prime.install({"TT_DETOK_PRIME_TOKENS": "16"}) == 16
    )  # again: the bound moves, the stock is kept
    assert BoundedDecodeStream.stock is stock and BoundedDecodeStream.bound == 16
    assert detok_prime.install({}) == 0 and tokenizers.decoders.DecodeStream is stock


def _tokenizer() -> Tokenizer:
    """A byte-level BPE over a small vocabulary, the decoder the model's tokenizer uses
    (ByteLevel), with a
    special token; enough for multi-byte characters to span several ids."""
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    merges: list[tuple[str, str]] = []
    for word in ("Ġthe", "Ġfox", "Ġdog", "Ġjumps", "Ġover", "ĠHello", "Ġworld"):
        pieces = list(word)
        while len(pieces) > 1:
            a, b = pieces[0], pieces[1]
            merges.append((a, b))
            pieces = [a + b] + pieces[2:]
            vocab.setdefault(a + b, len(vocab))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|eot|>", "<|user|>"])
    return tok


def _emit(stream, tok: Tokenizer, ids: list[int]) -> list[str]:
    return [stream.step(tok, i) or "" for i in ids]


@pytest.mark.parametrize("skip_special", [True, False])
@pytest.mark.parametrize("bound", [4, 7, 16])
def test_bounded_stream_emits_the_same_text(bound: int, skip_special: bool):
    tok = _tokenizer()
    eot = tok.token_to_id("<|eot|>")
    user = tok.token_to_id("<|user|>")
    rng = random.Random(bound)
    vocab = tok.get_vocab_size(with_added_tokens=False)
    text = "the quick brown fox jumps over the lazy dog. 日本語のテキスト。 Hello world"
    body = tok.encode(text, add_special_tokens=False).ids
    prompts = {
        "ascii": body,
        "multibyte-split": tok.encode(text + " 日本", add_special_tokens=False).ids,
        "special": body + [eot],
        "special-run": body + [user, eot, user],
        "random": [rng.randrange(vocab) for _ in range(500)],
    }
    outputs = [
        tok.encode(
            " Hello world. 日本語も大丈夫です。 the end", add_special_tokens=False
        ).ids,
        [rng.randrange(vocab) for _ in range(64)],
        [eot if rng.random() < 0.2 else rng.randrange(vocab) for _ in range(64)],
    ]
    detok_prime.install({"TT_DETOK_PRIME_TOKENS": str(bound)})
    for name, prompt in prompts.items():
        for out in outputs:
            stock = BoundedDecodeStream.stock(
                ids=prompt, skip_special_tokens=skip_special
            )
            bounded = tokenizers.decoders.DecodeStream(
                ids=prompt, skip_special_tokens=skip_special
            )
            assert isinstance(bounded, BoundedDecodeStream) and bounded.primed == min(
                bound, len(prompt)
            )
            a = _emit(stock, tok, out)
            b = _emit(bounded, tok, out)
            if a == b:
                continue
            # the stream's own degenerate case (the decoded prompt ends with U+FFFD: the
            # first step never sets
            # its prefix and echoes its whole window as the first delta); the bounded
            # stream echoes at most the
            # tail's text, and the rest is identical
            i = next(i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y)
            assert (
                name == "random" and a[i].endswith(b[i]) and a[i + 1 :] == b[i + 1 :]
            ), (name, bound, i)


def test_a_bound_below_a_character_is_the_degenerate_case():
    """Why the minimum is 4: a one-id tail inside a multi-byte character decodes to
    U+FFFD, the stream never
    sets its prefix and the first delta echoes the window; the stock priming of the
    whole prompt does not."""
    tok = _tokenizer()
    prompt = tok.encode("the fox 日本", add_special_tokens=False).ids
    out = tok.encode(" Hello", add_special_tokens=False).ids
    stock = _emit(
        tokenizers.decoders.DecodeStream(ids=prompt, skip_special_tokens=True), tok, out
    )
    BoundedDecodeStream.bound = (
        1  # below the minimum value() allows; set directly to show the failure mode
    )
    try:
        one = _emit(BoundedDecodeStream(ids=prompt, skip_special_tokens=True), tok, out)
    finally:
        BoundedDecodeStream.bound = detok_prime.DEFAULT
    assert stock == [" Hello"] and one == ["\ufffd Hello"]
    four = _emit(
        BoundedDecodeStream(ids=prompt[-4:], skip_special_tokens=True), tok, out
    )
    assert four == stock


def test_bounded_stream_without_ids_and_delegation():
    tok = _tokenizer()
    detok_prime.install({"TT_DETOK_PRIME_TOKENS": "4"})
    s = tokenizers.decoders.DecodeStream(
        skip_special_tokens=True
    )  # vLLM's reset path: no ids
    assert isinstance(s, BoundedDecodeStream) and s.primed == 0
    ids = tok.encode(" Hello world", add_special_tokens=False).ids
    assert "".join(_emit(s, tok, ids)) == " Hello world"
    s2 = tokenizers.decoders.DecodeStream(ids=ids, skip_special_tokens=True)
    assert s2.primed == min(4, len(ids))
    with pytest.raises(AttributeError):
        _ = s2.no_such_attribute
