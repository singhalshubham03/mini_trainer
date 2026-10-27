"""Tests for accelerator selection in mini_trainer.device_utils."""

import pytest
import torch

from mini_trainer import device_utils


@pytest.fixture(autouse=True)
def _unpinned(monkeypatch):
    # conftest pins unit tests to the CUDA path; these tests exercise selection.
    monkeypatch.delenv("MINI_TRAINER_DEVICE", raising=False)


def _fake(monkeypatch, *, cuda: bool, xpu: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)


@pytest.mark.parametrize(
    "cuda,xpu,expected_type,expected_backend",
    [
        (True, False, "cuda", "nccl"),
        (True, True, "cuda", "nccl"),  # CUDA wins when both are present
        (False, True, "xpu", "xccl"),
        (False, False, "cuda", "nccl"),  # CPU-only hosts keep the original CUDA path
    ],
)
def test_selection(monkeypatch, cuda, xpu, expected_type, expected_backend):
    _fake(monkeypatch, cuda=cuda, xpu=xpu)
    assert device_utils.get_device_type() == expected_type
    assert device_utils.get_dist_backend() == expected_backend
    assert device_utils.get_device_module() is getattr(torch, expected_type)
    assert device_utils.get_device(3) == torch.device(expected_type, 3)


def test_env_override(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    monkeypatch.setenv("MINI_TRAINER_DEVICE", "CUDA")
    assert device_utils.get_device_type() == "cuda"


def test_get_device_defaults_to_local_rank(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert device_utils.get_device() == torch.device("xpu", 2)


def test_helpers_dispatch_to_xpu_module(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    calls = []
    for name in ("empty_cache", "reset_peak_memory_stats", "set_device", "manual_seed_all"):
        monkeypatch.setattr(torch.xpu, name, lambda *a, _n=name: calls.append((_n, a)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("CUDA used on XPU host"))
    device_utils.empty_cache()
    device_utils.reset_peak_memory_stats()
    device_utils.set_device(1)
    device_utils.manual_seed_all(7)
    assert calls == [
        ("empty_cache", ()),
        ("reset_peak_memory_stats", ()),
        ("set_device", (1,)),
        ("manual_seed_all", (7,)),
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="Intel XPU not available")
def test_real_xpu_selected_when_no_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; CUDA is preferred")
    assert device_utils.get_device_type() == "xpu"
    t = torch.ones(4, device=device_utils.get_device(0))
    assert t.device.type == "xpu"
    assert device_utils.memory_allocated() > 0
