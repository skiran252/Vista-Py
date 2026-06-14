import random
from typing import List, Tuple

from vista.core.example import Example


class Dataset:
    """Manages train/val split and minibatch sampling as described in the paper."""

    def __init__(
        self,
        examples: List[Example],
        train_size: int = 50,
        val_size: int = 50,
        seed: int = 0,
    ):
        self._rng = random.Random(seed)
        shuffled = list(examples)
        self._rng.shuffle(shuffled)

        # Clamp sizes to available data
        total_needed = train_size + val_size
        if len(shuffled) < total_needed:
            # Use a proportional split when data is scarce
            ratio = train_size / total_needed
            train_size = max(1, int(len(shuffled) * ratio))
            val_size = max(1, len(shuffled) - train_size)

        self.train: List[Example] = shuffled[:train_size]
        self.val: List[Example] = shuffled[train_size : train_size + val_size]

    def sample_minibatch(self, size: int) -> List[Example]:
        """Sample a minibatch M ~ D_train with |M| = size."""
        k = min(size, len(self.train))
        return self._rng.sample(self.train, k)

    def split_from_list(
        self, examples: List[Example]
    ) -> Tuple[List[Example], List[Example]]:
        """Utility: split an arbitrary list into two halves."""
        mid = len(examples) // 2
        return examples[:mid], examples[mid:]
