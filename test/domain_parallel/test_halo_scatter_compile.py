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

r"""``torch.compile`` tests for the halo scatter-correction primitive.

The compiled ``halo_scatter_correct`` op is checked against an independent
plain-``funcol`` reference (forward and backward), so a bug in the op cannot hide
behind an equally-buggy reference. Runs under the standard ``multigpu_static``
harness (``torchrun`` + ``distributed_mesh``); the correction needs at least two
ranks to have any halo, so single-rank runs are skipped.
"""

import contextlib
import os

# Per-rank, node-local Triton compile cache. The nvshmem_triton backend JIT-compiles its
# device kernels on every rank at once; the default shared (networked-home) cache races on
# the same artifact files -> ``OSError: [Errno 116] Stale file handle``. Isolate per rank on
# /tmp before importing anything that pulls in triton (``RANK`` is set by torchrun).
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/triton_cache_{os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))}",  # noqa: S108
)

import pytest
import torch
import torch.distributed as dist

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
    halo_forward_exchange,
    halo_reverse_exchange,
    halo_scatter_correct,
    pack_halo_routing,
)


def _ring_routing(rank: int, world_size: int, n_owned: int, lend: int):
    r"""Directed-ring halo: each rank lends its first ``lend`` owned rows to the
    next rank and borrows ``lend`` ghost rows from the previous rank. Non-degenerate
    for any ``world_size >= 2``."""
    send_indices = [[] for _ in range(world_size)]
    send_indices[(rank + 1) % world_size] = list(range(lend))
    send_sizes = [[0] * world_size for _ in range(world_size)]
    for i in range(world_size):
        send_sizes[i][(i + 1) % world_size] = lend
    n_ghost = sum(send_sizes[i][rank] for i in range(world_size))
    return n_owned, n_owned + n_ghost, send_indices, send_sizes


def _dense_routing(rank: int, world_size: int, n_owned: int, lend: int):
    r"""Dense all-to-all halo: each rank lends its first ``lend`` owned rows to *every*
    other rank and borrows ``lend`` ghost rows from each. Unlike the directed ring (one
    neighbour), this gives ``world_size - 1`` sources per exchange -- so it exercises the
    multi-neighbour single-kernel overlapped pull (``n_src > 1``), the regime that path
    targets. Non-degenerate for any ``world_size >= 2``."""
    send_indices = [list(range(lend)) if j != rank else [] for j in range(world_size)]
    send_sizes = [
        [lend if j != i else 0 for j in range(world_size)] for i in range(world_size)
    ]
    n_ghost = sum(send_sizes[i][rank] for i in range(world_size))
    return n_owned, n_owned + n_ghost, send_indices, send_sizes


def run_halo_scatter_correct(mesh, backend, n_owned=6, lend=2, feat=4):
    device = DistributedManager().device
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    n_owned, n_padded, send_indices, send_sizes = _ring_routing(
        rank, world_size, n_owned, lend
    )
    send_idx_t = [
        torch.tensor(s, dtype=torch.int64, device=device) for s in send_indices
    ]
    routing = pack_halo_routing(
        send_indices, send_sizes, n_owned, rank, world_size, device=device
    )

    torch.manual_seed(100 + rank)
    padded0 = torch.randn(n_padded, feat, dtype=torch.float64, device=device)

    def fn(p, r):
        return halo_scatter_correct(p, r, group=mesh)

    # Independent ground truth: forward(reverse(.)) via plain funcol (no custom_op).
    # The map M = forward.reverse is self-adjoint, so the gradient of sum(M @ p) is
    # M @ ones (== the op applied to a ones tensor).
    def _plain_correct(p):
        return halo_forward_exchange(
            halo_reverse_exchange(
                p, n_owned, send_idx_t, send_sizes, rank, world_size, mesh
            ),
            send_idx_t,
            send_sizes,
            rank,
            world_size,
            mesh,
        )

    ref_fwd = _plain_correct(padded0)
    ref_grad = _plain_correct(torch.ones_like(padded0))
    assert not torch.allclose(ref_fwd, padded0), "halo correction is a no-op here"

    # Closed-form check independent of the exchange kernels: correct(ones) on owned
    # row i == 1 (itself) + the number of ranks that borrowed it.
    lent_count = torch.zeros(n_owned, dtype=torch.float64, device=device)
    for j in range(world_size):
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


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_halo_scatter_correct_1d(distributed_mesh, backend):
    if distributed_mesh.size() < 2:
        pytest.skip("halo correction needs >= 2 ranks")
    run_halo_scatter_correct(distributed_mesh, backend)


def _make_halo_shard_tensor(padded, mesh, routing):
    r"""A ShardTensor whose local is a ``[owned | ghost]`` halo, carrying the packed
    routing as an extra inner tensor. Uses a local-honest ``Replicate`` spec
    (global == local): the overlapping halo locals do not tile a mesh global, so an
    honest ``Shard(0)`` would make AOT insert cross-rank redistributes for the local
    scatter. Cross-rank movement lives in the halo op instead."""
    from torch.distributed.tensor._dtensor_spec import TensorMeta
    from torch.distributed.tensor.placement_types import Replicate

    from physicsnemo.domain_parallel import ShardTensor
    from physicsnemo.domain_parallel.shard_tensor import ShardTensorSpec

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


def run_halo_shard_tensor_scatter_add(mesh, backend, n_owned=6, lend=2, feat=4):
    from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
        register_halo_scatter_handlers,
    )

    register_halo_scatter_handlers()
    device = DistributedManager().device
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    n_owned, n_padded, send_indices, send_sizes = _ring_routing(
        rank, world_size, n_owned, lend
    )
    send_idx_t = [
        torch.tensor(s, dtype=torch.int64, device=device) for s in send_indices
    ]
    routing = pack_halo_routing(
        send_indices, send_sizes, n_owned, rank, world_size, device=device
    )
    idx = torch.arange(n_padded, device=device).unsqueeze(-1).expand(-1, feat)

    torch.manual_seed(100 + rank)
    src0 = torch.randn(n_padded, feat, dtype=torch.float64, device=device)

    # Reference: identity scatter (agg is zeros) then the plain halo correction.
    def _plain_correct(p):
        return halo_forward_exchange(
            halo_reverse_exchange(
                p, n_owned, send_idx_t, send_sizes, rank, world_size, mesh
            ),
            send_idx_t,
            send_sizes,
            rank,
            world_size,
            mesh,
        )

    ref_fwd = _plain_correct(src0)
    ref_grad = _plain_correct(torch.ones_like(src0))
    assert not torch.allclose(ref_fwd, src0), "correction is a no-op here"

    def fn(agg, index, source):
        return agg.scatter_add(0, index, source)

    # Eager: the scatter_add function handler applies the halo correction, with the
    # gradient threaded through the wrapper (``to_local``) back to the plain source.
    src_e = src0.clone().requires_grad_(True)
    agg_e = _make_halo_shard_tensor(
        torch.zeros(n_padded, feat, dtype=torch.float64, device=device), mesh, routing
    )
    out_e = fn(agg_e, idx, src_e)
    (grad_e,) = torch.autograd.grad(out_e.to_local().sum(), src_e)
    torch.testing.assert_close(out_e._local_tensor, ref_fwd, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(grad_e, ref_grad, rtol=1e-9, atol=1e-9)

    # Compiled: the correction survives as a differentiable graph node; the gradient
    # for the plain source comes back plain (not re-wrapped as a ShardTensor).
    torch._dynamo.reset()
    src_c = src0.clone().requires_grad_(True)
    agg_c = _make_halo_shard_tensor(
        torch.zeros(n_padded, feat, dtype=torch.float64, device=device), mesh, routing
    )
    cf = torch.compile(fn, backend=backend, fullgraph=True)
    out_c = cf(agg_c, idx, src_c)
    (grad_c,) = torch.autograd.grad(out_c.to_local().sum(), src_c)
    assert type(grad_c) is torch.Tensor, f"compiled grad is {type(grad_c)}"
    torch.testing.assert_close(out_c.to_local().detach(), ref_fwd, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grad_c, ref_grad, rtol=1e-6, atol=1e-6)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_halo_shard_tensor_scatter_add_1d(distributed_mesh, backend):
    if distributed_mesh.size() < 2:
        pytest.skip("halo correction needs >= 2 ranks")
    run_halo_shard_tensor_scatter_add(distributed_mesh, backend)


def _force_halo_backend(name):
    if name is None:
        os.environ.pop("PHYSICSNEMO_HALO_BACKEND", None)
    else:
        os.environ["PHYSICSNEMO_HALO_BACKEND"] = name


def _symm_mem_capable(mesh):
    r"""True when the symmetric-memory backend can serve this mesh (CUDA, >=2 ranks,
    and a workspace rendezvous succeeds). Collective: every rank runs it, so all
    return the same verdict on homogeneous hardware."""
    if not torch.cuda.is_available() or mesh.size() < 2:
        return False
    try:
        import torch.distributed._symmetric_memory as sm

        with torch.cuda.device(DistributedManager().device):
            sm.get_symm_mem_workspace(mesh.get_group().group_name, 1024)
        return True
    except Exception:
        return False


def run_symm_mem_equivalence(mesh, backend, n_owned=6, lend=2, feat=4):
    r"""The symm-mem transport must match the funcol oracle bitwise (fwd+bwd), eager
    and compiled -- pins the self-adjoint / linear-map equivalence across backends."""
    device = DistributedManager().device
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    n_owned, n_padded, send_indices, send_sizes = _ring_routing(
        rank, world_size, n_owned, lend
    )
    routing = pack_halo_routing(
        send_indices, send_sizes, n_owned, rank, world_size, device=device
    )
    torch.manual_seed(100 + rank)
    padded0 = torch.randn(n_padded, feat, dtype=torch.float64, device=device)

    def fn(p, r):
        return halo_scatter_correct(p, r, group=mesh)

    # funcol oracle (eager fwd+bwd).
    _force_halo_backend("funcol")
    pf = padded0.clone().requires_grad_(True)
    ref_fwd = fn(pf, routing)
    (ref_grad,) = torch.autograd.grad(ref_fwd.sum(), pf)

    # symm-mem, eager: identical float64 arithmetic, only the transport differs, so equality is
    # exact. Repeat many times to surface any nondeterministic ordering race in the fences.
    _force_halo_backend("symm_mem")
    try:
        for _ in range(50):
            ps = padded0.clone().requires_grad_(True)
            sm_fwd = fn(ps, routing)
            (sm_grad,) = torch.autograd.grad(sm_fwd.sum(), ps)
            torch.testing.assert_close(sm_fwd, ref_fwd, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(sm_grad, ref_grad, rtol=1e-12, atol=1e-12)
    finally:
        _force_halo_backend(None)

    # Auto-selection (no env override) also picks symm-mem here.
    from physicsnemo.domain_parallel.shard_utils.halo_scatter import select_halo_backend

    assert select_halo_backend(mesh).name == "symm_mem"

    # symm-mem, compiled: the op body reads the backend env at runtime, so it survives tracing.
    _force_halo_backend("symm_mem")
    try:
        torch._dynamo.reset()
        pc = padded0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend=backend, fullgraph=True)
        out_c = cf(pc, routing)
        (grad_c,) = torch.autograd.grad(out_c.sum(), pc)
    finally:
        _force_halo_backend(None)
    torch.testing.assert_close(out_c, ref_fwd, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_c, ref_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_halo_scatter_symm_mem_equivalence_1d(distributed_mesh, backend):
    if not _symm_mem_capable(distributed_mesh):
        pytest.skip("symmetric memory (>=2 P2P/NVSHMEM GPUs) not available")
    run_symm_mem_equivalence(distributed_mesh, backend)


def _nvshmem_triton_capable(mesh):
    r"""True when the cross-node NVSHMEM-Triton backend can serve this mesh: CUDA, >=2
    ranks, NVSHMEM available, and the ``@requires_nvshmem`` getmem kernel builds (device
    ``.bc`` locatable). Local (non-collective) so it cannot hang; homogeneous hardware ->
    same verdict on every rank. Works intra-node too (NVSHMEM spans a single node), so the
    equivalence can be exercised on a multi-GPU box, not only cross-node."""
    if not torch.cuda.is_available() or mesh.size() < 2:
        return False
    from physicsnemo.domain_parallel.shard_utils import halo_scatter as hs

    if not hs._HAS_NVSHMEM_TRITON:
        return False
    try:
        import torch.distributed._symmetric_memory as sm

        if not sm.is_nvshmem_available():
            return False
        with torch.cuda.device(DistributedManager().device):
            return hs._nvshmem_get_launchable() is not None
    except Exception:
        return False


def run_nvshmem_triton_equivalence(
    mesh, backend, routing_fn=_ring_routing, n_owned=6, lend=2, feat=4
):
    r"""The cross-node NVSHMEM-Triton transport (device-initiated getmem) must match the
    funcol oracle bitwise (fwd+bwd), eager and compiled -- same float64 arithmetic, only
    the transport differs. Mirrors :func:`run_symm_mem_equivalence`. ``routing_fn`` selects
    the halo topology (ring = 1 neighbour; dense = ``world_size - 1``, which exercises the
    multi-neighbour single-kernel overlapped pull)."""
    import torch.distributed._symmetric_memory as sm

    from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
        _group_is_multinode,
        select_halo_backend,
    )

    device = DistributedManager().device
    group = mesh.get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    n_owned, n_padded, send_indices, send_sizes = routing_fn(
        rank, world_size, n_owned, lend
    )
    routing = pack_halo_routing(
        send_indices, send_sizes, n_owned, rank, world_size, device=device
    )
    torch.manual_seed(100 + rank)
    padded0 = torch.randn(n_padded, feat, dtype=torch.float64, device=device)

    def fn(p, r):
        return halo_scatter_correct(p, r, group=mesh)

    # funcol oracle (eager fwd+bwd).
    _force_halo_backend("funcol")
    pf = padded0.clone().requires_grad_(True)
    ref_fwd = fn(pf, routing)
    (ref_grad,) = torch.autograd.grad(ref_fwd.sum(), pf)

    # nvshmem-triton, eager: identical arithmetic, only the transport (device getmem +
    # host-barrier readiness) differs, so equality is exact. Loop to surface any
    # buffer-reuse / barrier-ordering bug across the cached symmetric buffers. Restore the
    # default (CUDA/IPC) symm-mem backend afterwards so a later symm-mem test is unaffected
    # by the process-wide set_backend("NVSHMEM") this path performs.
    _force_halo_backend("nvshmem_triton")
    try:
        for _ in range(50):
            ps = padded0.clone().requires_grad_(True)
            nt_fwd = fn(ps, routing)
            (nt_grad,) = torch.autograd.grad(nt_fwd.sum(), ps)
            torch.testing.assert_close(nt_fwd, ref_fwd, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(nt_grad, ref_grad, rtol=1e-12, atol=1e-12)
    finally:
        _force_halo_backend(None)

    # Auto-selection: a multi-node group picks nvshmem-triton (single-node prefers symm-mem).
    if _group_is_multinode(mesh):
        assert select_halo_backend(mesh).name == "nvshmem_triton"

    # nvshmem-triton, compiled (backend read at runtime inside the opaque op).
    _force_halo_backend("nvshmem_triton")
    try:
        torch._dynamo.reset()
        pc = padded0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend=backend, fullgraph=True)
        out_c = cf(pc, routing)
        (grad_c,) = torch.autograd.grad(out_c.sum(), pc)
    finally:
        _force_halo_backend(None)
        with contextlib.suppress(Exception):
            sm.set_backend(
                "CUDA"
            )  # best-effort restore of the default symm-mem backend
    torch.testing.assert_close(out_c, ref_fwd, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_c, ref_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
# ring (1 neighbour) + dense (world-1 neighbours, feat 256) covers both the single-source
# path and the multi-neighbour single-kernel overlapped pull at a bandwidth-relevant width.
@pytest.mark.parametrize(
    "topo,feat", [("ring", 4), ("dense", 256)], ids=["ring", "dense256"]
)
def test_halo_scatter_nvshmem_triton_equivalence_1d(
    distributed_mesh, backend, topo, feat
):
    if not _nvshmem_triton_capable(distributed_mesh):
        pytest.skip("NVSHMEM + Triton device API (>=2 GPUs) not available")
    # The symm-mem backend is process-wide, so if the symm_mem test locked CUDA/IPC first
    # (intra-node) NVSHMEM cannot be selected in this process; skip cleanly in that case.
    from physicsnemo.domain_parallel.shard_utils.halo_scatter import (
        reset_nvshmem_halo_state,
    )

    routing_fn = _ring_routing if topo == "ring" else _dense_routing
    try:
        run_nvshmem_triton_equivalence(
            distributed_mesh, backend, routing_fn=routing_fn, feat=feat
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "locked to a different backend" in msg or "can not be changed" in msg:
            pytest.skip(f"symm-mem backend already locked to non-NVSHMEM: {msg}")
        raise
    finally:
        # Coordinated free of the cached symmetric buffers while the runtime is still live
        # (all ranks reach this together) -- avoids the NVSHMEM teardown-order segfault at
        # uncoordinated interpreter exit.
        reset_nvshmem_halo_state()


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
def test_halo_scatter_nvshmem_triton_overlap_dense_1d(distributed_mesh):
    r"""The opt-in single-kernel overlapped pull (``PHYSICSNEMO_HALO_NVSHMEM_OVERLAP=1``:
    one kernel posts all ``getmem_nbi`` then one ``quiet``) must also match the funcol oracle
    bitwise. Dense routing (``world_size - 1`` sources) exercises the multi-neighbour loop --
    the case the overlap targets. This guards correctness of the overlap path even though it
    is off by default (it is slower than blocking on the ``ibrc`` fabric, see the backend's
    ``_nvshmem_overlap_enabled``)."""
    if not _nvshmem_triton_capable(distributed_mesh):
        pytest.skip("NVSHMEM + Triton device API (>=2 GPUs) not available")
    from physicsnemo.domain_parallel.shard_utils import halo_scatter as hs

    if not hs._HAS_NVSHMEM_OVERLAP:
        pytest.skip("getmem_nbi_block / quiet not available in this torch")

    prev = os.environ.get("PHYSICSNEMO_HALO_NVSHMEM_OVERLAP")
    os.environ["PHYSICSNEMO_HALO_NVSHMEM_OVERLAP"] = "1"
    try:
        run_nvshmem_triton_equivalence(
            distributed_mesh, "aot_eager", routing_fn=_dense_routing, feat=64
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "locked to a different backend" in msg or "can not be changed" in msg:
            pytest.skip(f"symm-mem backend already locked to non-NVSHMEM: {msg}")
        raise
    finally:
        if prev is None:
            os.environ.pop("PHYSICSNEMO_HALO_NVSHMEM_OVERLAP", None)
        else:
            os.environ["PHYSICSNEMO_HALO_NVSHMEM_OVERLAP"] = prev
        hs.reset_nvshmem_halo_state()


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("topo", ["ring", "dense"])
def test_halo_scatter_nvshmem_triton_flags_1d(distributed_mesh, topo):
    r"""The opt-in barrier-free readiness path (``PHYSICSNEMO_HALO_NVSHMEM_FLAGS=1``: device
    flags -- READY producer->consumer, DONE consumer->producer, monotonic seq -- replace the
    host ``dist.barrier``s) must match the funcol oracle bitwise, fwd+bwd. Runs ring (1
    neighbour: simplest READY/DONE) and dense (``world_size - 1`` readers+sources: the full
    multi-peer handshake, and the reuse-safety DONE waits). The equivalence's 50x eager loop
    is the barrier-free repeated-exchange stress test."""
    if not _nvshmem_triton_capable(distributed_mesh):
        pytest.skip("NVSHMEM + Triton device API (>=2 GPUs) not available")
    from physicsnemo.domain_parallel.shard_utils import halo_scatter as hs

    if not hs._HAS_NVSHMEM_FLAGS:
        pytest.skip("putmem_block / signal_wait_until not available in this torch")

    routing_fn = _ring_routing if topo == "ring" else _dense_routing
    feat = 4 if topo == "ring" else 64
    prev = os.environ.get("PHYSICSNEMO_HALO_NVSHMEM_FLAGS")
    os.environ["PHYSICSNEMO_HALO_NVSHMEM_FLAGS"] = "1"
    try:
        run_nvshmem_triton_equivalence(
            distributed_mesh, "aot_eager", routing_fn=routing_fn, feat=feat
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "locked to a different backend" in msg or "can not be changed" in msg:
            pytest.skip(f"symm-mem backend already locked to non-NVSHMEM: {msg}")
        raise
    finally:
        if prev is None:
            os.environ.pop("PHYSICSNEMO_HALO_NVSHMEM_FLAGS", None)
        else:
            os.environ["PHYSICSNEMO_HALO_NVSHMEM_FLAGS"] = prev
        hs.reset_nvshmem_halo_state()
