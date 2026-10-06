# SPDX-License-Identifier: Apache-2.0
"""Which block-aligned prefixes have a saved recurrent state, and what holds it.

Attention keeps its prefix in paged KV that the block manager can hand to any request. A
recurrent or convolutional layer does not: its state is a running summary of the tokens one
request has seen, so reusing a shared prefix means putting that summary back explicitly. The
model owns the bytes and returns an opaque handle; this maps the content hash of a prefix to
that handle.

The question this answers is *how much of a prefix is servable*, which is not the same as how
much of its KV is cached. A hybrid model can have every attention page for a prefix and still be
unable to start from it. Answering it before the scheduler commits is the whole point: once a
request is admitted with a cache hit, the skipped tokens are never sent to the model, so there is
no way to recompute what the state should have been.

Deliberately free of vLLM imports: block hashes are opaque keys and handles are opaque values.
"""

from collections import OrderedDict


class RecurrentPrefixCache:
    """A bounded, least-recently-used index from prefix hash to snapshot handle."""

    def __init__(self, block_size: int, capacity: int):
        if block_size <= 0 or capacity < 0:
            raise ValueError("Recurrent prefix cache needs a positive block size and capacity")
        self.block_size = block_size
        self.capacity = capacity
        # hash -> (handle, tokens), ordered least-recently-used first
        self._entries: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def servable_tokens(self, block_hashes, limit: int) -> int:
        """The longest held prefix of these blocks, in tokens, at most ``limit``.

        Only whole blocks count. A held prefix is usable just when every token before it is also
        the same, which the block hash already encodes, so the longest match is the last held
        block within the limit rather than the first gap.
        """
        usable = min(len(block_hashes), limit // self.block_size)
        for count in range(usable, 0, -1):
            entry = self._entries.get(block_hashes[count - 1])
            if entry is not None:
                self._entries.move_to_end(block_hashes[count - 1])
                return count * self.block_size
        return 0

    def handle_for(self, block_hashes, tokens: int):
        """The handle holding exactly ``tokens`` of these blocks, or None."""
        if tokens <= 0 or tokens % self.block_size:
            return None
        index = tokens // self.block_size - 1
        if index >= len(block_hashes):
            return None
        entry = self._entries.get(block_hashes[index])
        return None if entry is None else entry[0]

    def remember(self, block_hash, handle, tokens: int):
        """Record a saved prefix. Returns a handle the caller must free, if one was displaced.

        An existing entry for the same hash is replaced rather than duplicated, and its handle is
        returned for freeing: two handles for one prefix would leave the model holding state that
        nothing can ever ask for.
        """
        if tokens <= 0 or tokens % self.block_size:
            raise ValueError("A saved prefix must be a whole number of blocks")
        displaced = None
        existing = self._entries.pop(block_hash, None)
        if existing is not None:
            displaced = existing[0]
        elif self.capacity and len(self._entries) >= self.capacity:
            _, evicted = self._entries.popitem(last=False)
            displaced = evicted[0]
        if self.capacity:
            self._entries[block_hash] = (handle, tokens)
        return displaced

    def forget_handle(self, handle) -> None:
        """Drop whatever entry points at this handle, once its state is gone."""
        for key, (held, _) in list(self._entries.items()):
            if held == handle:
                del self._entries[key]
                return

    def clear(self) -> None:
        self._entries.clear()
