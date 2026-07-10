"""Sampler classes for distributed training."""

import bisect
import math
import random

from torch.utils.data import Sampler


class DistributedSequentialSampler(Sampler):
    """A memory-efficient sampler that shards a dataset across DDP ranks sequentially.

    This avoids materializing a massive index tensor (as PyTorch's DistributedSampler does)
    by yielding indices lazily using Python's range. It supports an optional global
    start_offset to randomize the initial position deterministically across ranks.

    Args:
        dataset_len: Total length of the dataset
        num_replicas: Number of distributed replicas (GPUs)
        rank: Rank of the current process
        drop_last: Whether to drop the last incomplete batch
        start_offset: Starting offset in the dataset (allows randomization)
    """

    def __init__(self, dataset_len: int, num_replicas: int, rank: int, block_size: int, drop_last: bool = False, start_offset: int = 0) -> None:
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.drop_last = bool(drop_last)
        # Ensure offset within bounds
        self.start_offset = int(start_offset % self.dataset_len) if self.dataset_len > 0 else 0

        # Stride by num_replicas * block_size to ensure non-overlapping sequences across ranks
        # Each rank gets sequences spaced block_size apart from each other
        self.stride = self.num_replicas * self.block_size

        # Compute how many samples this rank will yield
        remaining = max(0, self.dataset_len - (self.start_offset + self.rank * self.block_size))
        if self.drop_last:
            self._length = remaining // self.stride
        else:
            self._length = (remaining + self.stride - 1) // self.stride

    def __iter__(self):
        if self.dataset_len == 0:
            return iter(())
        # Each rank starts at its offset + rank * block_size, then strides by num_replicas * block_size
        start = self.start_offset + self.rank * self.block_size
        return iter(range(start, self.dataset_len, self.stride))

    def __len__(self) -> int:
        return self._length


class SequentialOffsetSampler(Sampler):
    """Sequential sampler that starts from a specific offset."""

    def __init__(self, dataset_len: int, start_offset: int = 0) -> None:
        self.dataset_len = int(dataset_len)
        self.start_offset = int(start_offset % self.dataset_len) if self.dataset_len > 0 else 0
        self._length = max(0, self.dataset_len - self.start_offset)

    def __iter__(self):
        if self.dataset_len == 0:
            return iter(())
        return iter(range(self.start_offset, self.dataset_len))

    def __len__(self) -> int:
        return self._length


class FixedRandomChunkDistributedSampler(Sampler):
    """Distributed sampler with a fixed random chunk order.

    The sampler:
    - advances over sequence starts spaced by ``block_size`` (non-overlapping sequence starts),
    - groups those starts into fixed-size chunks,
    - shuffles chunk order once deterministically via ``seed``,
    - assigns shuffled chunks to DDP ranks by striding over shuffled chunk order.

    This gives reproducible, memory-efficient shuffling without materializing all indices.
    """

    def __init__(
        self,
        dataset_len: int,
        num_replicas: int,
        rank: int,
        block_size: int,
        chunk_size_units: int = 262_144,
        seed: int = 1337,
        start_offset: int = 0,
        resume_samples_seen_per_rank: int = 0,
    ) -> None:
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.chunk_size_units = max(1, int(chunk_size_units))
        self.seed = int(seed)
        self.start_offset = int(start_offset % self.dataset_len) if self.dataset_len > 0 else 0

        if self.dataset_len <= self.start_offset:
            self._num_units = 0
        else:
            self._num_units = math.ceil((self.dataset_len - self.start_offset) / self.block_size)

        num_chunks = math.ceil(self._num_units / self.chunk_size_units) if self._num_units > 0 else 0
        shuffled_chunk_ids = list(range(num_chunks))
        rng = random.Random(self.seed)
        rng.shuffle(shuffled_chunk_ids)

        # Each rank takes every N-th chunk from the same shuffled order.
        rank_chunk_ids = shuffled_chunk_ids[self.rank::self.num_replicas]
        self._rank_chunks = []
        self._prefix_chunk_lengths = [0]
        for chunk_id in rank_chunk_ids:
            unit_start = chunk_id * self.chunk_size_units
            unit_end = min(unit_start + self.chunk_size_units, self._num_units)
            if unit_start >= unit_end:
                continue
            self._rank_chunks.append((unit_start, unit_end))
            self._prefix_chunk_lengths.append(self._prefix_chunk_lengths[-1] + (unit_end - unit_start))

        total_rank_units = self._prefix_chunk_lengths[-1]
        self.resume_samples_seen_per_rank = max(0, min(int(resume_samples_seen_per_rank), total_rank_units))
        self._length = total_rank_units - self.resume_samples_seen_per_rank

    def __iter__(self):
        if self._length <= 0:
            return iter(())

        skip = self.resume_samples_seen_per_rank
        chunk_idx = bisect.bisect_right(self._prefix_chunk_lengths, skip) - 1
        in_chunk_offset = skip - self._prefix_chunk_lengths[chunk_idx]

        def _iter_positions():
            for i in range(chunk_idx, len(self._rank_chunks)):
                unit_start, unit_end = self._rank_chunks[i]
                start_u = unit_start + (in_chunk_offset if i == chunk_idx else 0)
                for unit in range(start_u, unit_end):
                    yield self.start_offset + unit * self.block_size

        return _iter_positions()

    def __len__(self) -> int:
        return self._length
