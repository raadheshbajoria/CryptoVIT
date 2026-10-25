"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def base_cfg():
    from cryptovit.utils import load_config

    return load_config()


@pytest.fixture(scope="session")
def cifar100_cfg():
    from cryptovit.utils import load_config

    return load_config(extra_files=["cifar100.yaml"])
