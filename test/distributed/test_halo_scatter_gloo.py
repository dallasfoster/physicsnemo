# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Local gloo/CPU tests for the halo scatter-correction primitive.

A gloo + ``torch.multiprocessing.spawn`` companion to the ``multigpu_static``
suite in ``test/domain_parallel/test_halo_scatter_compile.py``. It runs
non-degenerate 2- and 4-rank halos on CPU without GPUs, for local iteration where
a multi-GPU box is unavailable. The compiled ``halo_scatter_correct`` op is checked
against an independent plain-``funcol`` reference (forward and backward), plus a
recompile-survival check (routing rides as a graph-input tensor).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _synthetic_routing(rank: int, alt: bool = False):
    r"""A fixed, non-degenerate 2-rank halo layout (no partitioner).

    Rank 0 lends 2 owned rows to rank 1; rank 1 lends 3 to rank 0. With ``alt`` the
    lent-row INDICES differ but the counts (hence the packed-routing shape) are
    identical -- used to prove a routing value change does not recompile.
    """
    world_size, n_owned, feat = 2, 5, 4
    send_sizes = [[0, 2], [3, 0]]
    if rank == 0:
        send_indices = [[], [2, 3] if alt else [0, 1]]
    else:
        send_indices = [[0, 1, 2] if alt else [2, 3, 4], []]
    n_ghost = sum(send_sizes[i][rank] for i in range(world_size))
    return world_size, n_owned, n_owned + n_ghost, feat, send_indices, send_sizes


def _plain_correct_fn(halo_forward_exchange, halo_reverse_exchange):
    r"""Build an independent forward(reverse(.)) reference over ``send_idx_t``."""

    def _make(n_owned, send_idx_t, send_sizes, rank, ws, group=None):
        def _plain(p):
            return halo_forward_exchange(
                halo_reverse_exchange(
                    p, n_owned, send_idx_t, send_sizes, rank, ws, group
                ),
                send_idx_t,
                send_sizes,
                rank,
                ws,
                group,
            )

        return _plain

    return _make


def _worker(rank: int, world_size: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29691"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
            halo_forward_exchange,
            halo_reverse_exchange,
            halo_scatter_correct,
            pack_halo_routing,
        )

        ws, n_owned, n_padded, feat, send_indices, send_sizes = _synthetic_routing(rank)
        assert n_padded > n_owned, "degenerate partition: no halo rows to exercise"
        send_idx_t = [torch.tensor(s, dtype=torch.int64) for s in send_indices]
        routing = pack_halo_routing(send_indices, send_sizes, n_owned, rank, ws)

        torch.manual_seed(100 + rank)
        padded0 = torch.randn(n_padded, feat, dtype=torch.float64)

        def fn(p, r):
            return halo_scatter_correct(p, r)

        # Independent ground truth: forward(reverse(.)) via plain funcol. The map
        # M = forward.reverse is self-adjoint, so the gradient of sum(M @ p) is
        # M @ ones (== the op applied to a ones tensor).
        plain = _plain_correct_fn(halo_forward_exchange, halo_reverse_exchange)(
            n_owned, send_idx_t, send_sizes, rank, ws
        )
        ref_fwd = plain(padded0)
        ref_grad = plain(torch.ones_like(padded0))
        assert not torch.allclose(ref_fwd, padded0), "halo correction is a no-op here"

        # Closed-form check independent of the exchange kernels: correct(ones) on
        # owned row i == 1 (itself) + the number of ranks that borrowed it.
        lent_count = torch.zeros(n_owned, dtype=torch.float64)
        for j in range(ws):
            for i in send_indices[j]:
                lent_count[i] += 1.0
        expected_owned = (1.0 + lent_count).unsqueeze(-1).expand(-1, feat)
        torch.testing.assert_close(
            ref_grad[:n_owned], expected_owned, rtol=1e-12, atol=1e-12
        )

        # Eager custom_op matches the independent reference (forward and backward).
        pe = padded0.clone().requires_grad_(True)
        out_e = fn(pe, routing)
        (grad_e,) = torch.autograd.grad(out_e.sum(), pe)
        torch.testing.assert_close(out_e, ref_fwd, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grad_e, ref_grad, rtol=1e-12, atol=1e-12)

        # Compiled (fullgraph) matches the same reference.
        torch._dynamo.reset()
        pc = padded0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend=backend, fullgraph=True)
        out_c = cf(pc, routing)
        (grad_c,) = torch.autograd.grad(out_c.sum(), pc)
        torch.testing.assert_close(out_c, ref_fwd, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grad_c, ref_grad, rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


def _worker_inductor_lowering(rank: int, world_size: int) -> None:
    """Forward-only: the correction op must LOWER under inductor with the routing
    riding as a graph-input tensor. ``backend="eager"`` does not lower, so this is
    the guard for the tensor-routing op node."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29692"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
            halo_forward_exchange,
            halo_reverse_exchange,
            halo_scatter_correct,
            pack_halo_routing,
        )

        ws, n_owned, n_padded, feat, send_indices, send_sizes = _synthetic_routing(rank)
        send_idx_t = [torch.tensor(s, dtype=torch.int64) for s in send_indices]
        routing = pack_halo_routing(send_indices, send_sizes, n_owned, rank, ws)
        torch.manual_seed(100 + rank)
        padded0 = torch.randn(n_padded, feat, dtype=torch.float64)

        ref = _plain_correct_fn(halo_forward_exchange, halo_reverse_exchange)(
            n_owned, send_idx_t, send_sizes, rank, ws
        )(padded0)
        torch._dynamo.reset()
        cf = torch.compile(halo_scatter_correct, backend="inductor", fullgraph=True)
        out_c = cf(padded0, routing)  # must lower without raising
        torch.testing.assert_close(out_c, ref, rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


def _worker_recompile(rank: int, world_size: int) -> None:
    """A routing value change (same shape) must not recompile: routing rides as a graph-input
    tensor, so only its shape is guarded, not its values."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29694"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from torch._dynamo.testing import CompileCounterWithBackend

        from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
            halo_scatter_correct,
            pack_halo_routing,
        )

        ws, n_owned, n_padded, feat, si, ss = _synthetic_routing(rank)
        _, _, _, _, si_alt, _ = _synthetic_routing(rank, alt=True)
        r1 = pack_halo_routing(si, ss, n_owned, rank, ws)
        r2 = pack_halo_routing(si_alt, ss, n_owned, rank, ws)
        assert r1.shape == r2.shape, "alt routing must keep the packed shape fixed"

        torch.manual_seed(100 + rank)
        padded0 = torch.randn(n_padded, feat, dtype=torch.float64)

        def fn(p, r):
            return halo_scatter_correct(p, r)

        torch._dynamo.reset()
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(fn, backend=cnt, fullgraph=True)

        cf(padded0, r1)
        assert cnt.frame_count == 1
        out2 = cf(padded0, r2)
        assert cnt.frame_count == 1, (
            f"routing value change recompiled ({cnt.frame_count} frames)"
        )
        # The compiled op used r2's values (matches eager on r2), not a stale r1.
        torch.testing.assert_close(out2, fn(padded0, r2), rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


def _worker_subgroup(rank: int, world_size: int) -> None:
    """Two isolated 2-rank neighbourhoods ({0,1} and {2,3}) over the SAME world.

    Each rank corrects only within its neighbour sub-group, so the collective span
    is O(neighbours), not O(world). Exercises the group threaded through the op as
    its c10d name AND proves the two sub-groups do not cross-talk.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29693"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
            halo_forward_exchange,
            halo_reverse_exchange,
            halo_scatter_correct,
            pack_halo_routing,
        )

        # new_group is collective over the whole world: every rank calls both.
        g_lo = dist.new_group([0, 1])
        g_hi = dist.new_group([2, 3])
        sg = g_lo if rank < 2 else g_hi
        sub_rank, sub_ws = rank % 2, 2

        _, n_owned, n_padded, feat, send_indices, send_sizes = _synthetic_routing(
            sub_rank
        )
        send_idx_t = [torch.tensor(s, dtype=torch.int64) for s in send_indices]
        routing = pack_halo_routing(send_indices, send_sizes, n_owned, sub_rank, sub_ws)

        torch.manual_seed(100 + rank)
        padded0 = torch.randn(n_padded, feat, dtype=torch.float64)

        def fn(p, r):
            return halo_scatter_correct(p, r, group=sg)

        plain = _plain_correct_fn(halo_forward_exchange, halo_reverse_exchange)(
            n_owned, send_idx_t, send_sizes, sub_rank, sub_ws, sg
        )
        ref_fwd = plain(padded0)
        assert not torch.allclose(ref_fwd, padded0), "halo correction is a no-op here"

        torch._dynamo.reset()
        pc = padded0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend="aot_eager", fullgraph=True)
        out_c = cf(pc, routing)
        (grad_c,) = torch.autograd.grad(out_c.sum(), pc)
        ref_grad = plain(torch.ones_like(padded0))
        torch.testing.assert_close(out_c, ref_fwd, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grad_c, ref_grad, rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_halo_scatter_correct_compile_2ranks(backend: str) -> None:
    """Compiled halo scatter-correction matches an independent plain-funcol
    reference in forward AND backward, on a non-degenerate 2-rank partition."""
    mp.spawn(_worker, args=(2, backend), nprocs=2)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_halo_scatter_correct_inductor_lowering_2ranks() -> None:
    """The tensor-routing op lowers under inductor (forward-only)."""
    mp.spawn(_worker_inductor_lowering, args=(2,), nprocs=2)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_halo_scatter_correct_no_recompile_on_routing_change_2ranks() -> None:
    """A routing value change (same shape) does not recompile."""
    mp.spawn(_worker_recompile, args=(2,), nprocs=2)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_halo_scatter_correct_subgroup_4ranks() -> None:
    """Halo correction over a neighbour sub-group (2 isolated {0,1}/{2,3} pairs on
    a 4-rank world): compiled fwd+bwd match the plain-funcol reference over the
    same sub-group, proving the group plumbing and O(neighbours) span."""
    mp.spawn(_worker_subgroup, args=(4,), nprocs=4)


def _ring_routing(rank: int, n_owned: int = 6, lend: int = 2):
    r"""Uniform 2-rank ring halo: each rank lends its first ``lend`` owned rows to
    the other and borrows ``lend`` ghosts, so ``n_padded`` is identical on both
    ranks (needed for the no-comm ``from_local`` chunk path)."""
    send_sizes = [[0, lend], [lend, 0]]
    send_indices = [[], []]
    send_indices[(rank + 1) % 2] = list(range(lend))
    return n_owned, n_owned + lend, send_indices, send_sizes


def _make_halo_shard_tensor(padded, mesh, routing, world_size):
    from torch.distributed.tensor._dtensor_spec import TensorMeta
    from torch.distributed.tensor.placement_types import Replicate

    from physicsnemo.domain_parallel import ShardTensor
    from physicsnemo.domain_parallel.shard_tensor import ShardTensorSpec

    # A halo-padded local ([owned | ghost]) does not tile the mesh global, so use a
    # local-honest spec (Replicate, global == local) to keep autograd purely local.
    spec = ShardTensorSpec(
        mesh=mesh,
        placements=(Replicate(),) * mesh.ndim,
        tensor_meta=TensorMeta(
            shape=padded.shape, stride=padded.stride(), dtype=padded.dtype
        ),
        _local_shape=padded.shape,
        _sharding_shapes=None,
    )

    class _HaloShardTensor(ShardTensor):
        _extra_inner_tensors = ("_halo_meta_packed",)
        _halo_meta_packed_v = None
        _halo_meta_packed_c = None

        @property
        def _halo_meta_packed(self):
            v = self._halo_meta_packed_v
            return v if v is not None else self._stable_inner_sentinel("_halo_c")

        @_halo_meta_packed.setter
        def _halo_meta_packed(self, value):
            self._halo_meta_packed_v = value
            self._halo_meta_packed_c = None

    st = _HaloShardTensor.__new__(
        _HaloShardTensor,
        local_tensor=padded,
        spec=spec,
        requires_grad=padded.requires_grad,
    )
    st._halo_meta_packed = routing
    return st


def _worker_dispatch(rank: int, world_size: int, backend: str) -> None:
    """A halo ShardTensor's ``scatter_add`` is corrected: eager via the function
    handler, compiled via the dispatch handler, both matching a plain-funcol
    reference (fwd+bwd), with routing riding as an inner tensor (no recompile)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29695"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import DeviceMesh

        from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
            halo_forward_exchange,
            halo_reverse_exchange,
            pack_halo_routing,
            register_halo_scatter_handlers,
        )

        register_halo_scatter_handlers()
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dom",))
        n_owned, n_padded, send_indices, send_sizes = _ring_routing(rank)
        feat = 4
        send_idx_t = [torch.tensor(s, dtype=torch.int64) for s in send_indices]
        routing = pack_halo_routing(send_indices, send_sizes, n_owned, rank, world_size)
        idx = torch.arange(n_padded).unsqueeze(-1).expand(-1, feat)

        torch.manual_seed(100 + rank)
        src0 = torch.randn(n_padded, feat, dtype=torch.float64)

        # Reference: identity scatter (agg is zeros) then the plain halo correction.
        plain = _plain_correct_fn(halo_forward_exchange, halo_reverse_exchange)(
            n_owned, send_idx_t, send_sizes, rank, world_size, mesh
        )
        ref_fwd = plain(src0)
        ref_grad = plain(torch.ones_like(src0))
        assert not torch.allclose(ref_fwd, src0), "correction is a no-op here"

        def fn(agg, index, source):
            return agg.scatter_add(0, index, source)

        # Eager: the function handler applies the correction.
        src_e = src0.clone().requires_grad_(True)
        agg_e = _make_halo_shard_tensor(
            torch.zeros(n_padded, feat, dtype=torch.float64), mesh, routing, world_size
        )
        out_e = fn(agg_e, idx, src_e)
        (grad_e,) = torch.autograd.grad(out_e.to_local().sum(), src_e)
        torch.testing.assert_close(out_e._local_tensor, ref_fwd, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(grad_e, ref_grad, rtol=1e-9, atol=1e-9)

        # Compiled: the correction survives the traced graph as a differentiable
        # node. Differentiating ``out_c.to_local()`` back to the plain source must
        # thread the halo op's (self-adjoint) backward, and the gradient for a
        # plain input comes back plain (not re-wrapped as a ShardTensor).
        torch._dynamo.reset()
        src_c = src0.clone().requires_grad_(True)
        agg_c = _make_halo_shard_tensor(
            torch.zeros(n_padded, feat, dtype=torch.float64), mesh, routing, world_size
        )
        cf = torch.compile(fn, backend=backend, fullgraph=True)
        out_c = cf(agg_c, idx, src_c)
        assert torch.allclose(out_c.to_local().detach(), ref_fwd, rtol=1e-6, atol=1e-6)
        (grad_c,) = torch.autograd.grad(out_c.to_local().sum(), src_c)
        assert type(grad_c) is torch.Tensor, f"compiled grad is {type(grad_c)}"
        torch.testing.assert_close(grad_c, ref_grad, rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_halo_shard_tensor_scatter_add_2ranks(backend: str) -> None:
    """A halo ShardTensor's scatter_add is corrected eagerly and under compile,
    matching a plain-funcol reference in forward and backward."""
    mp.spawn(_worker_dispatch, args=(2, backend), nprocs=2)


def _worker_inplace_dispatch(
    rank: int, world_size: int, opname: str, backend: str
) -> None:
    """A halo ShardTensor's in-place ``scatter_add_`` / ``index_add_`` is corrected, eager and
    under compile: the returned value matches the plain-funcol reference (fwd+bwd) and the
    accumulator is mutated in place to the corrected values."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29696"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import DeviceMesh

        from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
            halo_forward_exchange,
            halo_reverse_exchange,
            pack_halo_routing,
            register_halo_scatter_handlers,
        )

        register_halo_scatter_handlers()
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dom",))
        n_owned, n_padded, send_indices, send_sizes = _ring_routing(rank)
        feat = 4
        send_idx_t = [torch.tensor(s, dtype=torch.int64) for s in send_indices]
        routing = pack_halo_routing(send_indices, send_sizes, n_owned, rank, world_size)

        torch.manual_seed(100 + rank)
        src0 = torch.randn(n_padded, feat, dtype=torch.float64)

        # Reference: identity accumulate into a zero accumulator, then the halo correction.
        # ``scatter_add_`` (full-shape arange index) and ``index_add_`` (1-D arange index)
        # both reduce to identity-into-zeros here, so they share the reference.
        plain = _plain_correct_fn(halo_forward_exchange, halo_reverse_exchange)(
            n_owned, send_idx_t, send_sizes, rank, world_size, mesh
        )
        ref_fwd = plain(src0)
        ref_grad = plain(torch.ones_like(src0))
        assert not torch.allclose(ref_fwd, src0), "correction is a no-op here"

        idx = (
            torch.arange(n_padded).unsqueeze(-1).expand(-1, feat)
            if opname == "scatter_add_"
            else torch.arange(n_padded)
        )

        def fn(agg, index, source):
            return getattr(agg, opname)(0, index, source)

        src_e = src0.clone().requires_grad_(True)
        agg_e = _make_halo_shard_tensor(
            torch.zeros(n_padded, feat, dtype=torch.float64), mesh, routing, world_size
        )
        out_e = fn(agg_e, idx, src_e)
        # Returned value: corrected + autograd-connected back to the plain source.
        (grad_e,) = torch.autograd.grad(out_e.to_local().sum(), src_e)
        torch.testing.assert_close(out_e._local_tensor, ref_fwd, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(grad_e, ref_grad, rtol=1e-9, atol=1e-9)
        # In-place semantics: the accumulator itself now holds the corrected values.
        torch.testing.assert_close(agg_e._local_tensor, ref_fwd, rtol=1e-9, atol=1e-9)

        # Compiled: the in-place correction must survive the traced graph -- both the
        # returned (differentiable) value and the in-place mutation of the accumulator.
        torch._dynamo.reset()
        src_c = src0.clone().requires_grad_(True)
        agg_c = _make_halo_shard_tensor(
            torch.zeros(n_padded, feat, dtype=torch.float64), mesh, routing, world_size
        )
        cf = torch.compile(fn, backend=backend, fullgraph=True)
        out_c = cf(agg_c, idx, src_c)
        assert torch.allclose(out_c.to_local().detach(), ref_fwd, rtol=1e-6, atol=1e-6)
        (grad_c,) = torch.autograd.grad(out_c.to_local().sum(), src_c)
        assert type(grad_c) is torch.Tensor, f"compiled grad is {type(grad_c)}"
        torch.testing.assert_close(grad_c, ref_grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            agg_c.to_local().detach(), ref_fwd, rtol=1e-6, atol=1e-6
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("opname", ["scatter_add_", "index_add_"])
def test_halo_shard_tensor_scatter_add_inplace_2ranks(
    opname: str, backend: str
) -> None:
    """The in-place ``scatter_add_`` / ``index_add_`` on a halo ShardTensor is corrected --
    eager and under ``torch.compile`` (aot_eager + inductor) -- with the return value matching
    the funcol reference (fwd+bwd) and the accumulator mutated in place to the corrected
    values."""
    mp.spawn(_worker_inplace_dispatch, args=(2, opname, backend), nprocs=2)


def test_select_halo_backend_env_override(monkeypatch) -> None:
    """Backend selection: funcol by default, honours the env override, and forcing
    the symm-mem backend on a CPU tensor surfaces a clear error at first use."""
    from physicsnemo.domain_parallel.shard_utils import halo_scatter as hs

    monkeypatch.delenv("PHYSICSNEMO_HALO_BACKEND", raising=False)
    assert hs.select_halo_backend().name == "funcol"

    # Regression: auto-selection must route CPU work (is_cuda=False) to funcol, never a
    # CUDA-only backend. A CUDA process registers a mixed "cpu:gloo,cuda:nccl" backend whose
    # NCCL substring used to let a CPU exchange pass the symm-mem capability gate and then
    # crash on the CPU tensor; the is_cuda guard prevents that for every auto branch.
    assert hs.select_halo_backend(is_cuda=False).name == "funcol"

    monkeypatch.setenv("PHYSICSNEMO_HALO_BACKEND", "funcol")
    assert hs.select_halo_backend().name == "funcol"

    monkeypatch.setenv("PHYSICSNEMO_HALO_BACKEND", "symm_mem")
    be = hs.select_halo_backend()
    assert be.name == "symm_mem"
    # CPU tensor -> the symm-mem backend refuses with a clear error (points at funcol).
    with pytest.raises(RuntimeError, match="funcol"):
        be.reverse(torch.zeros(1, 1), 1, [], [[0]], 0, 1, None)

    monkeypatch.setenv("PHYSICSNEMO_HALO_BACKEND", "bogus")
    with pytest.raises(ValueError):
        hs.select_halo_backend()


def test_pack_halo_routing_fixed_cap() -> None:
    """``cap`` pads the packed routing to a constant length regardless of how many rows a
    rank lends (fixed-shape routing for a compiled ``dynamic=False`` MD loop), and
    ``_unpack_halo_routing`` recovers it exactly, ignoring the trailing pad."""
    from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
        _unpack_halo_routing,
        pack_halo_routing,
    )

    ws, n_owned, rank, cap = 3, 10, 1, 8
    # rank 1 lends rows [0, 1] to rank 0 and [2] to rank 2 (total 3 <= cap).
    send_indices = [[0, 1], [], [2]]
    send_sizes = [[0, 0, 0], [2, 0, 1], [0, 0, 0]]

    r = pack_halo_routing(send_indices, send_sizes, n_owned, rank, ws, cap=cap)
    assert r.numel() == 4 + ws * ws + ws + cap  # fixed length

    # A different lend pattern (different total) keeps the SAME packed shape -> no recompile.
    r2 = pack_halo_routing(
        [[0], [], []], [[0, 0, 0], [1, 0, 0], [0, 0, 0]], n_owned, rank, ws, cap=cap
    )
    assert r2.shape == r.shape

    # Unpack recovers the routing exactly; the trailing pad (index-0 zeros) is not read.
    si, ssz, no, rk, w = _unpack_halo_routing(r)
    assert (no, rk, w) == (n_owned, rank, ws)
    assert ssz == send_sizes
    assert [t.tolist() for t in si] == send_indices

    # cap smaller than the real index count is a clear error, not silent truncation.
    with pytest.raises(ValueError, match="cap"):
        pack_halo_routing(send_indices, send_sizes, n_owned, rank, ws, cap=2)
