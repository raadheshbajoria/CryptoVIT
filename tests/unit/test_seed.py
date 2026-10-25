from __future__ import annotations

import random

import numpy as np
import pytest
import torch


class TestSetSeed:
    def test_python_random_reproducible(self):
        from cryptovit.utils import set_seed

        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_numpy_reproducible(self):
        from cryptovit.utils import set_seed

        set_seed(7)
        a = np.random.rand(5)
        set_seed(7)
        b = np.random.rand(5)
        assert np.allclose(a, b)

    def test_torch_reproducible(self):
        from cryptovit.utils import set_seed

        set_seed(99)
        a = torch.rand(5)
        set_seed(99)
        b = torch.rand(5)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        from cryptovit.utils import set_seed

        set_seed(1)
        a = torch.rand(10)
        set_seed(2)
        b = torch.rand(10)
        assert not torch.allclose(a, b)

    def test_negative_seed_raises(self):
        from cryptovit.utils import set_seed

        with pytest.raises(ValueError, match="non-negative"):
            set_seed(-1)
