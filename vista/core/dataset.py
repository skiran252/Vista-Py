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

        # Clamp sizes to available data. If total_needed exceeds available examples,
        # we allow the train and val sets to overlap rather than shrinking them proportionally.
        self.train: List[Example] = shuffled[:min(train_size, len(shuffled))]
        self.val: List[Example] = shuffled[-min(val_size, len(shuffled)):]

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
