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

r"""``torch.compile``-safe halo scatter-correction for ShardTensor.

A ``Shard(0)`` ShardTensor with a ``[owned | borrowed-ghost]`` row layout must,
after an in-place row scatter that writes into ghost rows, fold those
contributions back into their owners and refresh the ghost rows from the corrected
owners. This module exposes that correction as an AOT-traceable primitive:

* :func:`halo_reverse_exchange` / :func:`halo_forward_exchange` -- the
  fold-to-owner and refresh-ghost halves.
* :func:`halo_scatter_correct` -- the fused ``forward(reverse(padded))`` as a single
  ``torch.library.custom_op`` (opaque to fake mode, a packed-tensor routing arg, a
  self-adjoint backward) that survives both ``aot_eager`` and inductor.

Routing is passed as a packed tensor (from :func:`pack_halo_routing`), so the
primitive makes no partitioner or spatial assumptions and the values may change
across steps without recompiling. The data movement is a pluggable backend
(:class:`_HaloBackend`) chosen by :func:`select_halo_backend`
(``PHYSICSNEMO_HALO_BACKEND`` override), and all entry points accept a neighbour
sub-``group`` to bound the coordination span. :func:`register_halo_scatter_handlers`
wires the correction onto ``ShardTensor.scatter_add`` / ``index_add``.
"""

from __future__ import annotations

import os
from typing import Protocol

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh

__all__ = [
    "funcol_all_to_all_v_rows",
    "halo_forward_exchange",
    "halo_reverse_exchange",
    "halo_scatter_correct",
    "pack_halo_routing",
    "register_halo_scatter_handlers",
    "reset_nvshmem_halo_state",
    "select_halo_backend",
]


def _funcol_group_arg(group: object) -> object:
    r"""Return *group* in the form functional collectives accept: ``(DeviceMesh,
    0)``, a ``ProcessGroup`` / group-name ``str`` unchanged, or the default world
    group for ``None`` (funcol rejects ``None``)."""
    if isinstance(group, DeviceMesh):
        return (group, 0)
    if group is None:
        return dist.distributed_c10d._get_default_group()
    return group


def _halo_group_name(group: object) -> str:
    r"""Resolve *group* to its c10d group-name string (``""`` for the default world
    group) -- the traceable token a ``custom_op`` can carry, since a
    ``ProcessGroup`` is not a valid op argument."""
    if group is None:
        return ""
    if isinstance(group, str):
        return group
    if isinstance(group, DeviceMesh):
        return group._dim_group_names[0]
    return group.group_name


def funcol_all_to_all_v_rows(
    send_rows: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: object = None,
) -> torch.Tensor:
    r"""AOT-traceable variable-sized ``all_to_all`` over the rows (dim 0) of a
    tensor, via ``funcol.all_to_all_single``.

    Parameters
    ----------
    send_rows : torch.Tensor
        ``(sum(send_counts), *F)`` send buffer, rows ordered by destination rank.
    send_counts : list[int]
        Rows sent to each rank (plain ``int`` -- graph constants under compile).
    recv_counts : list[int]
        Rows received from each rank.
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group; ``None`` resolves to the default world group.

    Returns
    -------
    torch.Tensor
        ``(sum(recv_counts), *F)`` received rows, ordered by source rank.
    """
    trailing = tuple(send_rows.shape[1:])
    row_size = 1
    for d in trailing:
        row_size *= d
    flat_send = send_rows.contiguous().reshape(-1)
    send_flat = [c * row_size for c in send_counts]
    recv_flat = [c * row_size for c in recv_counts]
    total_recv = sum(recv_counts)
    flat_recv = funcol.wait_tensor(
        funcol.all_to_all_single(
            flat_send, recv_flat, send_flat, _funcol_group_arg(group)
        )
    )
    return flat_recv.reshape((total_recv,) + trailing)


# Backend seam: a transport owns the whole reverse/forward exchange, since the data-movement
# structure -- not just the collective call -- is transport-specific.


class _HaloBackend(Protocol):
    name: str

    def reverse(
        self,
        padded: torch.Tensor,
        n_owned: int,
        send_indices: list[torch.Tensor],
        send_sizes: list[list[int]],
        rank: int,
        world_size: int,
        group: object,
    ) -> torch.Tensor: ...

    def forward(
        self,
        owned: torch.Tensor,
        send_indices: list[torch.Tensor],
        send_sizes: list[list[int]],
        rank: int,
        world_size: int,
        group: object,
    ) -> torch.Tensor: ...


class _FuncolHaloBackend:
    r"""Functional-collective transport: dense ``all_to_all_single`` over the whole
    group. Portable everywhere (incl. gloo/CPU) and the default fallback."""

    name = "funcol"

    def reverse(
        self, padded, n_owned, send_indices, send_sizes, rank, world_size, group
    ):
        r"""Fold ghost rows back into owners via a reverse row all-to-all-v."""
        ghost = padded[n_owned:].contiguous()

        # Send each ghost block back to the owner it was borrowed from.
        rev_indices: list[torch.Tensor] = []
        offset = 0
        for r in range(world_size):
            n = int(send_sizes[r][rank])
            rev_indices.append(
                torch.arange(
                    offset, offset + n, device=padded.device, dtype=torch.int64
                )
            )
            offset += n
        send_rows = torch.cat(
            [ghost.index_select(0, rev_indices[j]) for j in range(world_size)], dim=0
        )
        send_counts = [int(send_sizes[r][rank]) for r in range(world_size)]
        recv_counts = [int(send_sizes[rank][j]) for j in range(world_size)]
        received_back = funcol_all_to_all_v_rows(
            send_rows, send_counts, recv_counts, group
        )

        # Fold the returned contributions into the lent owned rows (float64
        # accumulator keeps the sum well-conditioned for float32 inputs).
        acc_dtype = torch.float64 if padded.dtype == torch.float32 else padded.dtype
        owned = padded[:n_owned].to(acc_dtype)
        offset = 0
        for j in range(world_size):
            n = int(send_sizes[rank][j])
            if n == 0:
                continue
            owned = owned.index_add(
                0, send_indices[j], received_back[offset : offset + n].to(acc_dtype)
            )
            offset += n
        return owned.to(padded.dtype)

    def forward(self, owned, send_indices, send_sizes, rank, world_size, group):
        r"""Refresh ghost rows from owners via a forward row all-to-all-v."""
        send_rows = torch.cat(
            [owned.index_select(0, send_indices[j]) for j in range(world_size)], dim=0
        )
        send_counts = [int(send_sizes[rank][j]) for j in range(world_size)]
        recv_counts = [int(send_sizes[i][rank]) for i in range(world_size)]
        ghost_new = funcol_all_to_all_v_rows(send_rows, send_counts, recv_counts, group)
        return torch.cat([owned, ghost_new], dim=0)


def _symm_group_name(group: object) -> str:
    r"""Resolve *group* to a c10d group-name string usable with
    ``get_symm_mem_workspace`` (unlike :func:`_halo_group_name`, ``None`` resolves to
    the *named* default world group, not ``""``)."""
    if group is None:
        return dist.distributed_c10d._get_default_group().group_name
    if isinstance(group, str):
        return group
    if isinstance(group, DeviceMesh):
        return group._dim_group_names[0]
    return group.group_name


def _global_max_staged_rows(send_sizes: list[list[int]], world_size: int) -> int:
    r"""Rows the symmetric workspace must hold on every rank: the group-wide max over
    ranks of ``max(ghost rows, lent rows)``. Identical on all ranks (all hold the full
    ``send_sizes``), so the symmetric allocation stays uniform."""
    m = 0
    for r in range(world_size):
        ghost = sum(int(send_sizes[i][r]) for i in range(world_size))
        lent = sum(int(send_sizes[r][j]) for j in range(world_size))
        m = max(m, ghost, lent)
    return m


def _require_symm_mem(tensor: torch.Tensor):
    r"""Return the ``_symmetric_memory`` module, or raise a clear error when the
    symmetric-memory backend cannot serve *tensor* (CPU, or torch without it)."""
    if not tensor.is_cuda:
        raise RuntimeError(
            "the symmetric-memory halo backend requires CUDA tensors; "
            "set PHYSICSNEMO_HALO_BACKEND=funcol for CPU/gloo."
        )
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except Exception as exc:  # pragma: no cover - torch build without symm-mem
        raise RuntimeError(
            "torch.distributed._symmetric_memory is unavailable; "
            "set PHYSICSNEMO_HALO_BACKEND=funcol."
        ) from exc
    return symm_mem


# Signal channels for the per-neighbour readiness / completion fences. Reverse and
# forward use disjoint channels so a straggler's reverse signal is never mistaken for a
# forward one.
_REV_READY, _REV_DONE, _FWD_READY, _FWD_DONE = 0, 1, 2, 3


class _SymmMemHaloBackend:
    r"""Symmetric-memory one-sided transport (NVSHMEM device-initiated across nodes,
    CUDA-IPC ``get_buffer`` within a node).

    Each rank stages its exchange block into a symmetric workspace, then *pulls* each
    neighbour's block with ``get_buffer``. Only real ghost/lent data moves, and the
    coordination is neighbour-local: ``put_signal`` / ``wait_signal`` fence each
    exchange peer-to-peer (readiness before a pull, completion before a buffer is
    reused) instead of a group-wide ``barrier``, so a rank synchronizes with
    O(neighbours) peers rather than O(world). Numerically identical to
    :class:`_FuncolHaloBackend` (the correctness oracle); the region offsets mirror that
    backend's dense destination-ordered layout.
    """

    name = "symm_mem"

    @staticmethod
    def _row_numel(feat_shape: tuple[int, ...]) -> int:
        n = 1
        for d in feat_shape:
            n *= int(d)
        return n

    def reverse(
        self, padded, n_owned, send_indices, send_sizes, rank, world_size, group
    ):
        r"""Fold ghost rows back into owners via a one-sided reverse exchange."""
        symm_mem = _require_symm_mem(padded)
        feat = tuple(padded.shape[1:])
        row_numel = self._row_numel(feat)
        dtype = padded.dtype
        group_name = _symm_group_name(group)
        max_rows = _global_max_staged_rows(send_sizes, world_size)

        # Readers pull FROM my buffer (peers I borrowed from); sources are the peers I
        # pull from (peers I lent to). Reverse sends ghost contributions back to owners.
        readers = [s for s in range(world_size) if int(send_sizes[s][rank]) > 0]
        sources = [j for j in range(world_size) if int(send_sizes[rank][j]) > 0]

        acc_dtype = torch.float64 if dtype == torch.float32 else dtype
        owned = padded[:n_owned].to(acc_dtype)
        with torch.cuda.device(padded.device):
            handle = symm_mem.get_symm_mem_workspace(
                group_name, max(1, max_rows * row_numel * padded.element_size())
            )
            # Stage this rank's whole ghost region ([from_0 | from_1 | ...]); the block
            # borrowed from owner o already sits at o's pull offset. Signal each reader
            # its data is staged.
            ghost = padded[n_owned:].contiguous()
            if ghost.shape[0]:
                handle.get_buffer(rank, tuple(ghost.shape), dtype).copy_(ghost)
            for r in readers:
                handle.put_signal(r, channel=_REV_READY)
            # Pull each lent-to peer's staged contributions and fold them into the rows
            # this rank lent; signal that peer its buffer is free once the read is done.
            for j in sources:
                n = int(send_sizes[rank][j])
                off = sum(int(send_sizes[d][j]) for d in range(rank)) * row_numel
                handle.wait_signal(j, channel=_REV_READY)
                recv = handle.get_buffer(j, (n, *feat), dtype, storage_offset=off)
                owned = owned.index_add(0, send_indices[j], recv.to(acc_dtype))
                handle.put_signal(j, channel=_REV_DONE)
            # Hold until every reader has finished pulling, so the next phase's staging
            # cannot overwrite this buffer mid-read.
            for r in readers:
                handle.wait_signal(r, channel=_REV_DONE)
        return owned.to(dtype)

    def forward(self, owned, send_indices, send_sizes, rank, world_size, group):
        r"""Refresh ghost rows from the corrected owners via a one-sided exchange."""
        symm_mem = _require_symm_mem(owned)
        feat = tuple(owned.shape[1:])
        row_numel = self._row_numel(feat)
        dtype = owned.dtype
        group_name = _symm_group_name(group)
        max_rows = _global_max_staged_rows(send_sizes, world_size)

        # Forward broadcasts owners to ghosts, so the roles swap: readers are the peers
        # I lent to; sources are the peers I borrowed from.
        readers = [j for j in range(world_size) if int(send_sizes[rank][j]) > 0]
        sources = [i for i in range(world_size) if int(send_sizes[i][rank]) > 0]

        with torch.cuda.device(owned.device):
            handle = symm_mem.get_symm_mem_workspace(
                group_name, max(1, max_rows * row_numel * owned.element_size())
            )
            # Stage the rows lent to each peer, destination-ordered; signal each reader.
            send_rows = torch.cat(
                [owned.index_select(0, send_indices[j]) for j in range(world_size)],
                dim=0,
            )
            if send_rows.shape[0]:
                handle.get_buffer(rank, tuple(send_rows.shape), dtype).copy_(send_rows)
            for r in readers:
                handle.put_signal(r, channel=_FWD_READY)
            # Pull each refreshed ghost block from its owner (source-rank order); signal
            # that owner its buffer is free once the block is copied out.
            ghost_blocks = {}
            for i in sources:
                n = int(send_sizes[i][rank])
                off = sum(int(send_sizes[i][d]) for d in range(rank)) * row_numel
                handle.wait_signal(i, channel=_FWD_READY)
                gb = handle.get_buffer(i, (n, *feat), dtype, storage_offset=off)
                ghost_blocks[i] = gb.clone()
                handle.put_signal(i, channel=_FWD_DONE)
            for r in readers:
                handle.wait_signal(r, channel=_FWD_DONE)
        if not ghost_blocks:
            return owned
        ghost_new = torch.cat([ghost_blocks[i] for i in sources], dim=0)
        return torch.cat([owned, ghost_new], dim=0)


# Cross-node NVSHMEM (Triton) transport: device-initiated one-sided GET over IB for groups
# that span nodes, where the CUDA-IPC ``get_buffer`` path of :class:`_SymmMemHaloBackend` is
# node-local only. The exchange structure and offsets mirror that backend, but each neighbour
# block is pulled by a Triton kernel instead of an IPC copy. All NIC-touched buffers live in
# the NVSHMEM symmetric heap (a plain tensor's ``data_ptr()`` is not IB-registered), and
# readiness is fenced with a host barrier since ``ibrc`` has no atomic device signal.

# The Triton device bindings are optional; without the ``_nvshmem_triton`` device API or
# triton, ``_HAS_NVSHMEM_TRITON`` is False and this backend is never selected. Bind by name
# with fallbacks since the API differs across versions.
try:
    import triton as _triton
    import triton.language as _tl
    from torch.distributed._symmetric_memory import (
        _nvshmem_triton as _nvshmem_triton_mod,
    )

    _nvshmem_getmem_fn = getattr(_nvshmem_triton_mod, "getmem_block", None) or getattr(
        _nvshmem_triton_mod, "getmem_block_extern_wrapper", None
    )
    _requires_nvshmem_fn = getattr(_nvshmem_triton_mod, "requires_nvshmem", None)
    _HAS_NVSHMEM_TRITON = (
        _nvshmem_getmem_fn is not None and _requires_nvshmem_fn is not None
    )
    # Non-blocking getmem + quiet, for the overlapped-pull path; absent on older torch, in
    # which case that path is skipped and the blocking getmem below is used.
    _nvshmem_getmem_nbi_fn = getattr(
        _nvshmem_triton_mod, "getmem_nbi_block", None
    ) or getattr(_nvshmem_triton_mod, "getmem_nbi_block_extern_wrapper", None)
    _nvshmem_quiet_fn = getattr(_nvshmem_triton_mod, "quiet", None)
    _HAS_NVSHMEM_OVERLAP = (
        _HAS_NVSHMEM_TRITON
        and _nvshmem_getmem_nbi_fn is not None
        and _nvshmem_quiet_fn is not None
    )
    # Plain (non-atomic) put + device wait, for the flag-based readiness path: a producer
    # plain-puts a sequence number into a consumer's symmetric slot and the consumer spins on
    # it. Absent on older torch, in which case that path is unavailable.
    _nvshmem_putmem_fn = getattr(_nvshmem_triton_mod, "putmem_block", None) or getattr(
        _nvshmem_triton_mod, "putmem_block_extern_wrapper", None
    )
    _nvshmem_wait_fn = getattr(_nvshmem_triton_mod, "signal_wait_until", None)
    _HAS_NVSHMEM_FLAGS = (
        _HAS_NVSHMEM_TRITON
        and _nvshmem_putmem_fn is not None
        and _nvshmem_wait_fn is not None
        and _nvshmem_quiet_fn is not None
    )
except Exception:  # pragma: no cover - torch/triton without the nvshmem device API
    _HAS_NVSHMEM_TRITON = False
    _HAS_NVSHMEM_OVERLAP = False
    _HAS_NVSHMEM_FLAGS = False
    _nvshmem_getmem_fn = None
    _requires_nvshmem_fn = None
    _nvshmem_getmem_nbi_fn = None
    _nvshmem_quiet_fn = None
    _nvshmem_putmem_fn = None
    _nvshmem_wait_fn = None

# NVSHMEM comparison op for signal_wait_until: wait until the slot value is >= the operand.
_NVSHMEM_CMP_GE = 5

if _HAS_NVSHMEM_TRITON:

    @_triton.jit
    def _nvshmem_getmem_kernel(dst, src, nbytes, pe):  # pragma: no cover - GPU/IB only
        r"""Blocking pull of ``nbytes`` from symmetric ``src`` on PE ``pe`` into local ``dst``.
        ``src`` is this rank's symmetric base plus a byte offset, resolved to the same offset
        on ``pe``."""
        _nvshmem_getmem_fn(
            dst.to(_tl.int64), src.to(_tl.int64), nbytes.to(_tl.int64), pe
        )


if _HAS_NVSHMEM_OVERLAP:

    @_triton.jit
    def _nvshmem_pull_all_kernel(  # pragma: no cover - GPU/IB only
        recv_base, stage_base, pe_ptr, src_off_ptr, dst_off_ptr, nb_ptr, n_src
    ):
        r"""Pull ``n_src`` neighbour blocks in one launch: issue a non-blocking ``getmem_nbi``
        for each, then a single ``quiet()`` that completes them all (the posts and the
        ``quiet`` must share one kernel). Routing is four length-``n_src`` device arrays: ``pe``
        (raw) and byte offsets/counts (int64)."""
        for k in range(n_src):
            pe = _tl.load(pe_ptr + k)
            so = _tl.load(src_off_ptr + k)
            do = _tl.load(dst_off_ptr + k)
            nb = _tl.load(nb_ptr + k)
            _nvshmem_getmem_nbi_fn(
                (recv_base + do).to(_tl.int64),
                (stage_base + so).to(_tl.int64),
                nb.to(_tl.int64),
                pe.to(_tl.int32),
            )
        _nvshmem_quiet_fn()


if _HAS_NVSHMEM_FLAGS:

    @_triton.jit
    def _nvshmem_putflag_kernel(dst, src, nbytes, pe):  # pragma: no cover - GPU/IB only
        r"""Plain-put an ``nbytes`` flag from local symmetric ``src`` into symmetric ``dst`` on
        PE ``pe``, then ``quiet()`` so it is remotely complete on return."""
        _nvshmem_putmem_fn(
            dst.to(_tl.int64), src.to(_tl.int64), nbytes.to(_tl.int64), pe
        )
        _nvshmem_quiet_fn()

    @_triton.jit
    def _nvshmem_wait_kernel(sig, cmp_op, cmp_val):  # pragma: no cover - GPU/IB only
        r"""Device-spin until the symmetric slot ``sig`` satisfies ``cmp_op`` vs ``cmp_val``."""
        _nvshmem_wait_fn(sig, cmp_op, cmp_val)


# Caches: ``@requires_nvshmem`` launchables per kernel, and the symmetric staging/receive
# buffer pair per (group, shape) so the collective rendezvous happens once, not every step.
_nvshmem_launchables: dict = {}
_nvshmem_bufs: dict = {}
# Per-(group, world) symmetric flag buffers + a monotonic per-group exchange counter for
# the flag-based readiness path.
_nvshmem_flag_bufs: dict = {}
_nvshmem_epochs: dict = {}


def reset_nvshmem_halo_state() -> None:
    r"""Free the cached NVSHMEM symmetric buffers and reset the exchange counters. This is a
    coordinated cleanup -- every rank must call it together, since the symmetric frees are
    collective -- and should be called while the process group is still alive (e.g. before
    tearing it down) rather than left to uncoordinated interpreter exit."""
    _nvshmem_bufs.clear()
    _nvshmem_flag_bufs.clear()
    _nvshmem_epochs.clear()


def _nvshmem_launch(kernel):
    r"""Return the ``@requires_nvshmem``-decorated launchable for *kernel* (built + cached
    per kernel on first call). Raises if the NVSHMEM device library cannot be located."""
    launchable = _nvshmem_launchables.get(kernel)
    if launchable is None:
        if not _HAS_NVSHMEM_TRITON:
            raise RuntimeError("nvshmem-triton halo backend is unavailable")
        launchable = _requires_nvshmem_fn(kernel)
        _nvshmem_launchables[kernel] = launchable
    return launchable


def _nvshmem_get_launchable():
    r"""The blocking getmem launchable -- also the capability gate (its build proves the
    device .bc is locatable). See :func:`_nvshmem_launch`."""
    return _nvshmem_launch(_nvshmem_getmem_kernel)


def _nvshmem_overlap_enabled() -> bool:
    r"""Whether to use the overlapped-pull path (``getmem_nbi`` + one ``quiet``) instead of the
    default blocking per-neighbour ``getmem``. Opt-in via ``PHYSICSNEMO_HALO_NVSHMEM_OVERLAP=1``
    and off by default, as it is slower than blocking on fabrics without device atomics."""
    return (
        _HAS_NVSHMEM_OVERLAP
        and os.environ.get("PHYSICSNEMO_HALO_NVSHMEM_OVERLAP") == "1"
    )


def _nvshmem_flags_enabled() -> bool:
    r"""Whether to fence readiness/reuse with device flags instead of host ``dist.barrier``s.
    Opt-in via ``PHYSICSNEMO_HALO_NVSHMEM_FLAGS=1`` and off by default, as the per-neighbour
    flag traffic is slower than a single collective barrier on fabrics without device atomics."""
    return (
        _HAS_NVSHMEM_FLAGS and os.environ.get("PHYSICSNEMO_HALO_NVSHMEM_FLAGS") == "1"
    )


# Flag channels: one uint64 slot per (channel, peer). Reverse and forward use disjoint
# READY/DONE channels so a reverse flag is never read as a forward one.
_NV_REV_READY, _NV_REV_DONE, _NV_FWD_READY, _NV_FWD_DONE = 0, 1, 2, 3
_NV_N_CHANNELS = 4


def _nvshmem_flags(symm_mem, group_name, world_size, device):
    r"""Return the cached ``(flags, flags_h, seq, seq_h)`` for this (group, world), allocated
    once: a symmetric ``uint64[4 * world]`` flag buffer (slot ``channel * world + peer``) and a
    symmetric ``uint64[1]`` holding the value a producer puts into a consumer's slot."""
    key = (group_name, int(world_size))
    quad = _nvshmem_flag_bufs.get(key)
    if quad is None:
        _ensure_nvshmem_backend(symm_mem, device)
        flags = symm_mem.empty(
            _NV_N_CHANNELS * world_size, dtype=torch.uint64, device=device
        )
        flags.zero_()
        flags_h = symm_mem.rendezvous(flags, group_name)
        seq = symm_mem.empty(1, dtype=torch.uint64, device=device)
        seq_h = symm_mem.rendezvous(seq, group_name)
        quad = (flags, flags_h, seq, seq_h)
        _nvshmem_flag_bufs[key] = quad
    return quad


def _nvshmem_next_epoch(group_name) -> int:
    r"""Monotonic per-group exchange counter (incremented once per reverse/forward call). All
    ranks call in lockstep so the counters stay aligned, and a ``>=`` wait needs no reset."""
    e = _nvshmem_epochs.get(group_name, 0) + 1
    _nvshmem_epochs[group_name] = e
    return e


def _halo_dbg(msg: str) -> None:
    r"""Emit a per-rank progress line when ``PHYSICSNEMO_HALO_DEBUG`` is set -- localizes a
    hang in the cross-node exchange (the last line printed before a stall points at it)."""
    if os.environ.get("PHYSICSNEMO_HALO_DEBUG"):
        try:
            r = dist.get_rank()
        except Exception:
            r = "?"
        print(f"[halo dbg r{r}] {msg}", flush=True)


def _ensure_nvshmem_backend(symm_mem, device) -> None:
    r"""Select the NVSHMEM symmetric-memory backend (needed for cross-node RMA), unless it is
    already active. The backend is process-wide and cannot be changed once any symmetric tensor
    is allocated, so raise a clear error if a different one is already in use."""
    try:
        current = str(symm_mem.get_backend(device))
    except Exception:  # pragma: no cover - get_backend unavailable on older torch
        current = ""
    if "NVSHMEM" in current:
        return
    try:
        symm_mem.set_backend("NVSHMEM")
    except Exception as exc:
        raise RuntimeError(
            "the nvshmem-triton halo backend requires the NVSHMEM symmetric-memory "
            "backend, but it is locked to a different backend in this process (a symmetric "
            "tensor was already allocated under CUDA/NCCL -- e.g. the intra-node symm_mem "
            "halo backend). A process can use only one symm-mem backend: use "
            "PHYSICSNEMO_HALO_BACKEND=nvshmem_triton exclusively, or select the NVSHMEM "
            "backend before any symmetric allocation."
        ) from exc


def _nvshmem_symm_pair(symm_mem, group_name, max_rows, feat, dtype, device):
    r"""Return the cached ``(stage, recv)`` symmetric-heap buffers for this
    (group, shape), allocating + rendezvousing on first use. ``max_rows`` is the group-wide
    max staged rows, so the symmetric allocation is uniform across ranks."""
    key = (group_name, int(max_rows), tuple(int(d) for d in feat), str(dtype))
    pair = _nvshmem_bufs.get(key)
    if pair is None:
        _ensure_nvshmem_backend(symm_mem, device)  # cross-node heap; one-shot global
        stage = symm_mem.empty(max_rows, *feat, dtype=dtype, device=device)
        stage_h = symm_mem.rendezvous(stage, group_name)
        recv = symm_mem.empty(max_rows, *feat, dtype=dtype, device=device)
        recv_h = symm_mem.rendezvous(recv, group_name)
        pair = (stage, stage_h, recv, recv_h)
        _nvshmem_bufs[key] = pair
    return pair


class _NvshmemTritonHaloBackend:
    r"""Cross-node symmetric-memory transport: device-initiated NVSHMEM one-sided GET.

    Each rank stages its exchange block into a symmetric buffer, and after a host barrier for
    readiness every rank pulls the neighbour blocks it needs with a Triton ``getmem`` kernel
    and folds/appends them as :class:`_SymmMemHaloBackend` does. Numerically identical to
    :class:`_FuncolHaloBackend`; the region offsets mirror its destination-ordered layout.
    """

    name = "nvshmem_triton"

    @staticmethod
    def _row_numel(feat: tuple[int, ...]) -> int:
        n = 1
        for d in feat:
            n *= int(d)
        return n

    def _fence(self, group) -> None:
        r"""Host readiness/reuse fence: complete this rank's device work, then a group
        barrier so every rank observes the staged (and prior-read) symmetric buffers."""
        torch.cuda.synchronize()
        dist.barrier(group=_resolve_pg(group))

    @staticmethod
    def _flag_put(put_kern, flags_base, seq_base, channel, sender_rank, world, peer):
        r"""Plain-put this rank's current sequence value into ``peer``'s flag slot for
        (``channel``, this rank). The slot is indexed by the *sender*, so it has a single
        writer -- no atomic needed."""
        off = (channel * world + sender_rank) * 8  # uint64 slots
        put_kern[(1,)](flags_base + off, seq_base, 8, peer, num_warps=8)

    @staticmethod
    def _flag_wait(wait_kern, flags_base, channel, world, sender, epoch):
        r"""Device-spin until this rank's (``channel``, ``sender``) flag slot is ``>= epoch``
        (i.e. ``sender`` has reached this exchange). Monotonic seq -> no per-step reset."""
        off = (channel * world + sender) * 8
        wait_kern[(1,)](flags_base + off, _NVSHMEM_CMP_GE, epoch, num_warps=8)

    def _pull(self, recv_base, stage_base, plan, row_numel, elem, device):
        r"""Pull every neighbour block in *plan* (``[(pe, src_off_rows, n_rows), ...]``) into
        disjoint cumulative receive regions and return ``[(dst_row, n_rows), ...]`` for the
        caller's reduction. Uses one overlapped ``getmem_nbi`` kernel where available, else a
        blocking ``getmem`` per neighbour into the same layout."""
        regions, cum = [], 0
        for _pe, _so, n in plan:
            regions.append((cum, n))
            cum += n
        if not plan:
            return regions
        if _HAS_NVSHMEM_OVERLAP:
            pe_t = torch.tensor(
                [p for p, _s, _n in plan], dtype=torch.int64, device=device
            )
            src_off_t = torch.tensor(
                [s * row_numel * elem for _p, s, _n in plan],
                dtype=torch.int64,
                device=device,
            )
            dst_off_t = torch.tensor(
                [d * row_numel * elem for d, _n in regions],
                dtype=torch.int64,
                device=device,
            )
            nb_t = torch.tensor(
                [n * row_numel * elem for _d, n in regions],
                dtype=torch.int64,
                device=device,
            )
            _nvshmem_launch(_nvshmem_pull_all_kernel)[(1,)](
                recv_base,
                stage_base,
                pe_t,
                src_off_t,
                dst_off_t,
                nb_t,
                len(plan),
                num_warps=8,
            )
        else:
            kern = _nvshmem_get_launchable()
            for (p, s, _n), (d, n) in zip(plan, regions):
                kern[(1,)](
                    recv_base + d * row_numel * elem,
                    stage_base + s * row_numel * elem,
                    n * row_numel * elem,
                    p,
                    num_warps=8,
                )
        return regions

    def reverse(
        self, padded, n_owned, send_indices, send_sizes, rank, world_size, group
    ):
        r"""Fold ghost rows back into owners via a one-sided cross-node reverse exchange."""
        symm_mem = _require_symm_mem(padded)
        feat = tuple(padded.shape[1:])
        row_numel = self._row_numel(feat)
        elem = padded.element_size()
        dtype = padded.dtype
        device = padded.device
        group_name = _symm_group_name(group)
        max_rows = max(1, _global_max_staged_rows(send_sizes, world_size))
        stage, stage_h, recv, recv_h = _nvshmem_symm_pair(
            symm_mem, group_name, max_rows, feat, dtype, device
        )
        stage_base = int(stage_h.buffer_ptrs[rank])
        recv_base = int(recv_h.buffer_ptrs[rank])

        # Sources = peers I lent to (I pull back what they scattered into my lent rows).
        # Each source's staged block sits at this rank's pull offset within that source's
        # ghost region (destination-ordered, mirroring the funcol/symm-mem layout).
        sources = [j for j in range(world_size) if int(send_sizes[rank][j]) > 0]
        plan = [
            (
                j,
                sum(int(send_sizes[d][j]) for d in range(rank)),  # src offset (rows)
                int(send_sizes[rank][j]),  # rows
            )
            for j in sources
        ]
        acc_dtype = torch.float64 if dtype == torch.float32 else dtype
        owned = padded[:n_owned].to(acc_dtype)
        ghost = padded[n_owned:].contiguous()
        _halo_dbg(f"reverse: sources={sources} max_rows={max_rows} feat={feat}")

        if _nvshmem_flags_enabled():
            # Device flags replace the two host barriers: signal READY to peers that pull from
            # me, pull each source (waiting its READY first) and signal DONE, then wait all
            # readers' DONE before returning so the next exchange can safely restage.
            readers = [s for s in range(world_size) if int(send_sizes[s][rank]) > 0]
            epoch = _nvshmem_next_epoch(group_name)
            _flags, flags_h, _seq, seq_h = _nvshmem_flags(
                symm_mem, group_name, world_size, device
            )
            flags_base = int(flags_h.buffer_ptrs[rank])
            seq_base = int(seq_h.buffer_ptrs[rank])
            put_kern = _nvshmem_launch(_nvshmem_putflag_kernel)
            wait_kern = _nvshmem_launch(_nvshmem_wait_kernel)
            get_kern = _nvshmem_get_launchable()
            with torch.cuda.device(device):
                if ghost.shape[0]:
                    stage[: ghost.shape[0]].copy_(ghost)
                _seq.fill_(epoch)
                torch.cuda.synchronize()  # local: stage + seq visible before signaling
                for r in readers:
                    self._flag_put(
                        put_kern,
                        flags_base,
                        seq_base,
                        _NV_REV_READY,
                        rank,
                        world_size,
                        r,
                    )
                for j, off_rows, n in plan:
                    self._flag_wait(
                        wait_kern, flags_base, _NV_REV_READY, world_size, j, epoch
                    )
                    get_kern[(1,)](
                        recv_base,
                        stage_base + off_rows * row_numel * elem,
                        n * row_numel * elem,
                        j,
                        num_warps=8,
                    )
                    owned = owned.index_add(0, send_indices[j], recv[:n].to(acc_dtype))
                    self._flag_put(
                        put_kern,
                        flags_base,
                        seq_base,
                        _NV_REV_DONE,
                        rank,
                        world_size,
                        j,
                    )
                for r in readers:
                    self._flag_wait(
                        wait_kern, flags_base, _NV_REV_DONE, world_size, r, epoch
                    )
            _halo_dbg("reverse: done (flags)")
            return owned.to(dtype)

        with torch.cuda.device(device):
            self._fence(group)  # prior exchange's pulls complete before restaging
            # Stage this rank's whole ghost region; the block borrowed from each owner already
            # sits at that owner's pull offset.
            if ghost.shape[0]:
                stage[: ghost.shape[0]].copy_(ghost)
            self._fence(group)  # staged + visible to the NIC before any peer pulls
            _halo_dbg("reverse: staged+fenced, pulling")
            if _nvshmem_overlap_enabled():
                # Pull all sources into disjoint recv regions, then fold each into the rows
                # this rank lent it.
                regions = self._pull(
                    recv_base, stage_base, plan, row_numel, elem, device
                )
                for (j, _so, _n), (d, n) in zip(plan, regions):
                    owned = owned.index_add(
                        0, send_indices[j], recv[d : d + n].to(acc_dtype)
                    )
            else:
                # Blocking getmem per neighbour into a reused scratch, folding as we go.
                kern = _nvshmem_get_launchable()
                for j, off_rows, n in plan:
                    kern[(1,)](
                        recv_base,
                        stage_base + off_rows * row_numel * elem,
                        n * row_numel * elem,
                        j,
                        num_warps=8,
                    )
                    owned = owned.index_add(0, send_indices[j], recv[:n].to(acc_dtype))
        _halo_dbg("reverse: done")
        return owned.to(dtype)

    def forward(self, owned, send_indices, send_sizes, rank, world_size, group):
        r"""Refresh ghost rows from the corrected owners via a one-sided cross-node GET."""
        symm_mem = _require_symm_mem(owned)
        feat = tuple(owned.shape[1:])
        row_numel = self._row_numel(feat)
        elem = owned.element_size()
        dtype = owned.dtype
        device = owned.device
        group_name = _symm_group_name(group)
        max_rows = max(1, _global_max_staged_rows(send_sizes, world_size))
        stage, stage_h, recv, recv_h = _nvshmem_symm_pair(
            symm_mem, group_name, max_rows, feat, dtype, device
        )
        stage_base = int(stage_h.buffer_ptrs[rank])
        recv_base = int(recv_h.buffer_ptrs[rank])

        # Sources = peers I borrowed from (I pull my refreshed ghost block from each). Each
        # owner's block sits at this rank's offset within the rows that owner lent out.
        sources = [i for i in range(world_size) if int(send_sizes[i][rank]) > 0]
        plan = [
            (
                i,
                sum(int(send_sizes[i][d]) for d in range(rank)),  # src offset (rows)
                int(send_sizes[i][rank]),  # rows
            )
            for i in sources
        ]
        _halo_dbg(f"forward: sources={sources} max_rows={max_rows} feat={feat}")

        if _nvshmem_flags_enabled():
            # Same flag handshake as reverse with the roles swapped: readers are the peers I
            # lent to, sources the peers I borrowed from; disjoint forward channels.
            readers = [j for j in range(world_size) if int(send_sizes[rank][j]) > 0]
            epoch = _nvshmem_next_epoch(group_name)
            _flags, flags_h, _seq, seq_h = _nvshmem_flags(
                symm_mem, group_name, world_size, device
            )
            flags_base = int(flags_h.buffer_ptrs[rank])
            seq_base = int(seq_h.buffer_ptrs[rank])
            put_kern = _nvshmem_launch(_nvshmem_putflag_kernel)
            wait_kern = _nvshmem_launch(_nvshmem_wait_kernel)
            get_kern = _nvshmem_get_launchable()
            ghost_blocks = []
            with torch.cuda.device(device):
                send_rows = torch.cat(
                    [owned.index_select(0, send_indices[j]) for j in range(world_size)],
                    dim=0,
                )
                if send_rows.shape[0]:
                    stage[: send_rows.shape[0]].copy_(send_rows)
                _seq.fill_(epoch)
                torch.cuda.synchronize()  # local: stage + seq visible before signaling
                for r in readers:
                    self._flag_put(
                        put_kern,
                        flags_base,
                        seq_base,
                        _NV_FWD_READY,
                        rank,
                        world_size,
                        r,
                    )
                for i, off_rows, n in plan:
                    self._flag_wait(
                        wait_kern, flags_base, _NV_FWD_READY, world_size, i, epoch
                    )
                    get_kern[(1,)](
                        recv_base,
                        stage_base + off_rows * row_numel * elem,
                        n * row_numel * elem,
                        i,
                        num_warps=8,
                    )
                    ghost_blocks.append(recv[:n].clone())
                    self._flag_put(
                        put_kern,
                        flags_base,
                        seq_base,
                        _NV_FWD_DONE,
                        rank,
                        world_size,
                        i,
                    )
                for r in readers:
                    self._flag_wait(
                        wait_kern, flags_base, _NV_FWD_DONE, world_size, r, epoch
                    )
            _halo_dbg("forward: done (flags)")
            if not ghost_blocks:
                return owned
            return torch.cat([owned, torch.cat(ghost_blocks, dim=0)], dim=0)

        with torch.cuda.device(device):
            self._fence(group)
            # Stage the rows lent to each peer, destination-ordered.
            send_rows = torch.cat(
                [owned.index_select(0, send_indices[j]) for j in range(world_size)],
                dim=0,
            )
            if send_rows.shape[0]:
                stage[: send_rows.shape[0]].copy_(send_rows)
            self._fence(group)
            _halo_dbg("forward: staged+fenced, pulling")
            if _nvshmem_overlap_enabled():
                # Pull every ghost block into disjoint recv regions in source order, so
                # recv[:total] is the concatenated ghost region.
                regions = self._pull(
                    recv_base, stage_base, plan, row_numel, elem, device
                )
                _halo_dbg("forward: done")
                if not regions:
                    return owned
                total = regions[-1][0] + regions[-1][1]
                return torch.cat([owned, recv[:total]], dim=0)
            # Blocking getmem per owner into a reused scratch, cloning each block out before
            # the next pull overwrites it, then concatenating in source order.
            kern = _nvshmem_get_launchable()
            ghost_blocks = []
            for i, off_rows, n in plan:
                kern[(1,)](
                    recv_base,
                    stage_base + off_rows * row_numel * elem,
                    n * row_numel * elem,
                    i,
                    num_warps=8,
                )
                ghost_blocks.append(recv[:n].clone())
        _halo_dbg("forward: done")
        if not ghost_blocks:
            return owned
        ghost_new = torch.cat(ghost_blocks, dim=0)
        return torch.cat([owned, ghost_new], dim=0)


_FUNCOL_BACKEND = _FuncolHaloBackend()
_SYMM_MEM_BACKEND = _SymmMemHaloBackend()
_NVSHMEM_TRITON_BACKEND = _NvshmemTritonHaloBackend()


_symm_capability_cache: dict[str, bool] = {}
_nvshmem_triton_capability_cache: dict[str, bool] = {}
_group_multinode_cache: dict[str, bool] = {}


def _resolve_pg(group: object):
    r"""Resolve *group* to a ``ProcessGroup`` (or ``None`` if it cannot be), for the
    backend check in :func:`_symm_mem_usable`."""
    if group is None:
        return dist.distributed_c10d._get_default_group()
    if isinstance(group, DeviceMesh):
        try:
            return group.get_group() if group.ndim == 1 else group.get_group(0)
        except Exception:
            return None
    if isinstance(group, str):
        try:
            return dist.distributed_c10d._resolve_process_group(group)
        except Exception:
            return None
    return group


def _check_symm_mem_ipc(symm_mem, group_name: str) -> bool:
    r"""One-time collective check that a symmetric workspace can be rendezvoused for
    *group_name* (i.e. CUDA-IPC / P2P is available). Every rank must call this together;
    :func:`_symm_mem_usable` caches the verdict so it happens at most once per group."""
    try:
        with torch.cuda.device(torch.cuda.current_device()):
            symm_mem.get_symm_mem_workspace(group_name, 1024)
        return True
    except Exception:  # pragma: no cover - runs only on real multi-GPU hardware
        return False


def _symm_mem_usable(group: object) -> bool:
    r"""Whether the symmetric-memory transport is auto-selectable for *group*.

    Gated on a NCCL group plus a one-time, cached, per-group symmetric-workspace
    rendezvous check. That check is reliable: it succeeds for an
    intra-node group (the ``get_symm_mem_workspace`` / ``get_buffer`` path is CUDA-IPC,
    node-local) and fails cleanly for a cross-node one (the IPC handle exchange, "send
    fd", cannot cross nodes). ``is_nvshmem_available()`` is deliberately NOT trusted
    here: it reports only that NVSHMEM is compiled in, not that the default rendezvous
    uses it, so trusting it would auto-select symm-mem on a cross-node group and hang.
    ``funcol`` (no requirement) is the always-correct fallback -- including cross-node.
    """
    if not torch.cuda.is_available():
        return False
    try:
        import torch.distributed as _dist
        import torch.distributed._symmetric_memory as symm_mem
    except Exception:  # pragma: no cover - torch build without symm-mem
        return False
    if not (_dist.is_available() and _dist.is_initialized()):
        return False
    # symm-mem needs an NCCL-capable group; a substring test (not exact-match) since a CUDA
    # process registers a mixed backend like "cpu:gloo,cuda:nccl".
    pg = _resolve_pg(group)
    if pg is None:
        return False
    try:
        if "nccl" not in str(_dist.get_backend(pg)).lower():
            return False
    except Exception:
        return False
    group_name = _symm_group_name(group)
    cached = _symm_capability_cache.get(group_name)
    if cached is None:
        cached = _check_symm_mem_ipc(symm_mem, group_name)
        _symm_capability_cache[group_name] = cached
    return cached


def _nvshmem_triton_usable(group: object) -> bool:
    r"""Whether the cross-node NVSHMEM-Triton transport is auto-selectable for *group*: a NCCL
    group, NVSHMEM compiled in, the ``_nvshmem_triton`` device API and triton present, and a
    cached local check that the ``@requires_nvshmem`` launchable builds. Falls back to funcol
    otherwise."""
    if not (_HAS_NVSHMEM_TRITON and torch.cuda.is_available()):
        return False
    try:
        import torch.distributed as _dist
        import torch.distributed._symmetric_memory as symm_mem
    except Exception:  # pragma: no cover - torch build without symm-mem
        return False
    if not (_dist.is_available() and _dist.is_initialized()):
        return False
    pg = _resolve_pg(group)
    if pg is None:
        return False
    try:
        if "nccl" not in str(_dist.get_backend(pg)).lower():
            return False
        if not symm_mem.is_nvshmem_available():
            return False
    except Exception:
        return False
    group_name = _symm_group_name(group)
    cached = _nvshmem_triton_capability_cache.get(group_name)
    if cached is None:
        try:
            cached = _nvshmem_get_launchable() is not None
        except Exception:  # pragma: no cover - .bc not locatable on this build
            cached = False
        _nvshmem_triton_capability_cache[group_name] = cached
    return cached


def _group_is_multinode(group: object) -> bool:
    r"""Whether *group* spans more than one physical node, via a hostname all-gather (which
    allocates no symmetric memory, unlike the intra-node capability check). Cached per group;
    falls back to a world-size-vs-local-GPU-count heuristic if the gather cannot run."""
    import socket

    if not (dist.is_available() and dist.is_initialized()):
        return False
    pg = _resolve_pg(group)
    if pg is None:
        return False
    group_name = _symm_group_name(group)
    cached = _group_multinode_cache.get(group_name)
    if cached is None:
        try:
            world = dist.get_world_size(pg)
            gathered: list[object] = [None] * world
            dist.all_gather_object(gathered, socket.gethostname(), group=pg)
            cached = len({str(h) for h in gathered}) > 1
        except Exception:  # pragma: no cover - fall back to a local heuristic
            try:
                cached = dist.get_world_size(pg) > torch.cuda.device_count()
            except Exception:
                cached = False
        _group_multinode_cache[group_name] = cached
    return cached


def select_halo_backend(group: object = None, is_cuda: bool = True) -> _HaloBackend:
    r"""Return the halo transport backend for *group*.

    Honours ``PHYSICSNEMO_HALO_BACKEND`` (``"funcol"`` | ``"symm_mem"`` |
    ``"nvshmem_triton"``); otherwise picks the intra-node symmetric-memory backend when
    usable, then the cross-node NVSHMEM-Triton backend, and ``funcol`` (the always-correct
    fallback) otherwise.

    Parameters
    ----------
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group used for the capability check.
    is_cuda : bool, optional, default=True
        Whether the exchanged tensor is on CUDA. The symmetric-memory backends are CUDA-only,
        so CPU work always routes to ``funcol``; this is passed explicitly because it is not
        reliably inferable from *group* alone.

    Returns
    -------
    _HaloBackend
        The selected transport backend.
    """
    forced = os.environ.get("PHYSICSNEMO_HALO_BACKEND")
    if forced == "funcol":
        return _FUNCOL_BACKEND
    if forced == "symm_mem":
        return _SYMM_MEM_BACKEND
    if forced == "nvshmem_triton":
        return _NVSHMEM_TRITON_BACKEND
    if forced:
        raise ValueError(
            f"PHYSICSNEMO_HALO_BACKEND={forced!r} is not a known halo backend "
            "(expected 'funcol', 'symm_mem', or 'nvshmem_triton')."
        )
    # CPU work can only use funcol (the symm-mem backends are CUDA-only); guard before any
    # capability check so no CPU exchange reaches a CUDA backend.
    if not is_cuda:
        return _FUNCOL_BACKEND
    # Node locality selects the transport without allocating symmetric memory: single-node
    # uses the intra-node symm-mem backend, multi-node the NVSHMEM-Triton backend.
    if _group_is_multinode(group):
        if _nvshmem_triton_usable(group):
            return _NVSHMEM_TRITON_BACKEND
    elif _symm_mem_usable(group):
        return _SYMM_MEM_BACKEND
    return _FUNCOL_BACKEND


def halo_reverse_exchange(
    padded: torch.Tensor,
    n_owned: int,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    rank: int,
    world_size: int,
    group: object = None,
) -> torch.Tensor:
    r"""Fold borrowed ghost rows back into their owners (transpose of the forward
    halo gather), using the transport backend selected for *group*.

    ``padded`` is ``[owned (n_owned) | ghost]`` with ghost rows grouped by source
    rank; each ghost row is summed back into its owning row.

    Parameters
    ----------
    padded : torch.Tensor
        ``(n_owned + n_ghost, *F)`` local tensor, ``[owned | ghost]``.
    n_owned : int
        Number of owned rows (length of the returned block).
    send_indices : list[torch.Tensor]
        ``send_indices[j]`` = owned-row indices this rank lent to rank ``j``.
    send_sizes : list[list[int]]
        ``send_sizes[i][j]`` = rows rank ``i`` lent to rank ``j``.
    rank : int
        This rank (sub-group-relative when *group* is a sub-group).
    world_size : int
        Group size (sub-group size when *group* is a sub-group).
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group; ``None`` = default world group.

    Returns
    -------
    torch.Tensor
        ``(n_owned, *F)`` owned block with every borrowed contribution summed in.
    """
    return select_halo_backend(group, padded.is_cuda).reverse(
        padded, n_owned, send_indices, send_sizes, rank, world_size, group
    )


def halo_forward_exchange(
    owned: torch.Tensor,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    rank: int,
    world_size: int,
    group: object = None,
) -> torch.Tensor:
    r"""Refresh ghost rows from the owners: gather each peer's lent rows and append
    them, returning the ``[owned | ghost]`` layout (inverse of
    :func:`halo_reverse_exchange`), using the backend selected for *group*.

    Parameters
    ----------
    owned : torch.Tensor
        ``(n_owned, *F)`` owned block.
    send_indices : list[torch.Tensor]
        ``send_indices[j]`` = owned-row indices this rank lent to rank ``j``.
    send_sizes : list[list[int]]
        ``send_sizes[i][j]`` = rows rank ``i`` lent to rank ``j``.
    rank : int
        This rank (sub-group-relative when *group* is a sub-group).
    world_size : int
        Group size (sub-group size when *group* is a sub-group).
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group; ``None`` = default world group.

    Returns
    -------
    torch.Tensor
        ``(n_owned + n_ghost, *F)`` padded tensor with ghost rows refreshed.
    """
    return select_halo_backend(group, owned.is_cuda).forward(
        owned, send_indices, send_sizes, rank, world_size, group
    )


def _scatter_correct_dense(
    padded: torch.Tensor,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    n_owned: int,
    rank: int,
    world_size: int,
    group: object,
) -> torch.Tensor:
    r"""``forward(reverse(padded))`` over *group* using a single selected backend."""
    backend = select_halo_backend(group, padded.is_cuda)
    owned = backend.reverse(
        padded, n_owned, send_indices, send_sizes, rank, world_size, group
    )
    return backend.forward(owned, send_indices, send_sizes, rank, world_size, group)


def pack_halo_routing(
    send_indices: list[list[int]] | list[torch.Tensor],
    send_sizes: list[list[int]],
    n_owned: int,
    rank: int,
    world_size: int,
    device: object = None,
    cap: int | None = None,
) -> torch.Tensor:
    r"""Pack halo routing into a 1-D int64 tensor for :func:`halo_scatter_correct`.

    The packed tensor is meant to ride as a graph input (e.g. a ShardTensor extra
    inner tensor), so its values may change across steps and survive Dynamo graph
    breaks without recompiling -- unlike routing baked in as ``int[]`` constants,
    which are guarded and force a recompile on any change.

    Parameters
    ----------
    send_indices : list[list[int]] or list[torch.Tensor]
        ``send_indices[j]`` = owned-row indices this rank lent to rank ``j``.
    send_sizes : list[list[int]]
        ``send_sizes[i][j]`` = rows rank ``i`` lent to rank ``j``.
    n_owned : int
        Number of owned rows.
    rank : int
        This rank (sub-group-relative when a sub-group is used).
    world_size : int
        Group size (sub-group size when a sub-group is used).
    device : torch.device or str or None, optional, default=None
        Device for the packed tensor.
    cap : int or None, optional, default=None
        If given, pad the trailing index section so the packed tensor always has the fixed
        length ``4 + world_size**2 + world_size + cap``, keeping the routing a constant shape
        across steps (required by a compiled ``dynamic=False`` loop). Must be ``>=`` the total
        number of lent-row indices this rank holds; the pad is ignored on unpack.

    Returns
    -------
    torch.Tensor
        1-D int64 routing tensor consumed by :func:`halo_scatter_correct`, laid out
        as ``[world_size, n_owned, rank, n_flat, *send_sizes, *send_idx_lens,
        *send_idx_flat]`` (with *send_idx_flat* padded to ``cap`` when *cap* is set).

    Notes
    -----
    Index arrays are concatenated as tensors (their lengths come from shapes), so a
    device-resident ``send_indices`` is never moved to host -- the pack is free of a
    value-dependent device sync. The ``cap`` check uses only lengths (shapes), so it too
    adds no sync.
    """
    idx_tensors = [
        idx.reshape(-1).to(torch.int64)
        if isinstance(idx, torch.Tensor)
        else torch.tensor(idx, dtype=torch.int64)
        for idx in send_indices
    ]
    lens = [int(t.numel()) for t in idx_tensors]  # shapes only -- no value sync
    ss = [int(send_sizes[i][j]) for i in range(world_size) for j in range(world_size)]
    if device is None and idx_tensors:
        device = idx_tensors[0].device
    header = torch.tensor(
        [world_size, n_owned, rank, sum(lens), *ss, *lens],
        dtype=torch.int64,
        device=device,
    )
    idx_flat = (
        torch.cat([t.to(device) for t in idx_tensors])
        if idx_tensors
        else torch.zeros(0, dtype=torch.int64, device=device)
    )
    if cap is not None:
        total = int(sum(lens))
        if total > cap:
            raise ValueError(
                f"pack_halo_routing: cap={cap} is smaller than the number of lent-row "
                f"indices this rank holds ({total}); set cap to the per-rank maximum."
            )
        if idx_flat.numel() < cap:
            idx_flat = torch.cat(
                [
                    idx_flat,
                    torch.zeros(
                        cap - idx_flat.numel(), dtype=torch.int64, device=device
                    ),
                ]
            )
    elif not idx_tensors:
        return header
    return torch.cat([header, idx_flat])


def _unpack_halo_routing(routing: torch.Tensor):
    r"""Inverse of :func:`pack_halo_routing` (runs eagerly inside the op).

    Materializes only the small fixed header to host (the counts are needed as ``int[]`` split
    sizes); the index arrays stay on-device as views. It slices the index section by the
    per-rank lengths in the header, so any trailing padding from a ``cap`` pack is never read.
    """
    world_size, n_owned, rank, _n_flat = routing[:4].tolist()
    body_len = world_size * world_size + world_size
    body = routing[4 : 4 + body_len].tolist()
    ss, lens = body[: world_size * world_size], body[world_size * world_size :]
    send_sizes = [
        [ss[i * world_size + j] for j in range(world_size)] for i in range(world_size)
    ]
    send_indices, o = [], 4 + body_len
    for length in lens:
        send_indices.append(routing[o : o + length])  # device view -- no sync
        o += length
    return send_indices, send_sizes, n_owned, rank, world_size


@torch.library.custom_op("physicsnemo::halo_scatter_correct", mutates_args=())
def _halo_scatter_correct_op(
    padded: torch.Tensor, routing: torch.Tensor, group_name: str
) -> torch.Tensor:
    r"""Dispatcher-visible ``forward(reverse(padded))``, opaque to fake mode.

    ``routing`` is a 1-D int64 tensor (a graph input, not baked constants), so its
    values may change across steps and graph breaks without recompiling. The op body
    runs only at runtime on real tensors -- unpacking ``routing`` there -- while the
    trace sees only :func:`_halo_scatter_correct_fake`. Runs over the group named
    ``group_name`` (``""`` = default world group).
    """
    group = group_name or None
    send_indices, send_sizes, n_owned, rank, world_size = _unpack_halo_routing(routing)
    return _scatter_correct_dense(
        padded, send_indices, send_sizes, n_owned, rank, world_size, group
    )


@_halo_scatter_correct_op.register_fake
def _halo_scatter_correct_fake(padded, routing, group_name):
    return torch.empty_like(padded)


def _halo_correct_setup_context(ctx, inputs, output):
    _padded, routing, group_name = inputs
    ctx.routing = routing
    ctx.group_name = group_name


def _halo_correct_backward(ctx, grad):
    # forward(reverse(.)) is self-adjoint, so the VJP is the op applied to grad.
    grad_in = _halo_scatter_correct_op(grad.contiguous(), ctx.routing, ctx.group_name)
    return grad_in, None, None


_halo_scatter_correct_op.register_autograd(
    _halo_correct_backward, setup_context=_halo_correct_setup_context
)


def halo_scatter_correct(
    padded: torch.Tensor,
    routing: torch.Tensor,
    group: object = None,
) -> torch.Tensor:
    r"""``torch.compile``-safe halo scatter-correction on a ``[owned | ghost]``
    tensor.

    Folds borrowed-ghost contributions back into their owners and refreshes the
    ghost rows (``forward(reverse(padded))``) as a single AOT-traceable, inductor-
    lowerable, differentiable graph node. ``routing`` (from :func:`pack_halo_routing`)
    rides as a graph-input tensor, so it survives Dynamo graph breaks and per-step
    value changes without recompiling.

    Parameters
    ----------
    padded : torch.Tensor
        ``(n_owned + n_ghost, *F)`` local tensor, ``[owned | ghost]``.
    routing : torch.Tensor
        1-D int64 routing tensor from :func:`pack_halo_routing`.
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group; a neighbour sub-group bounds the coordination span.

    Returns
    -------
    torch.Tensor
        ``(n_owned + n_ghost, *F)`` corrected padded tensor.
    """
    return _halo_scatter_correct_op(padded, routing, _halo_group_name(group))


# ShardTensor scatter/index-add integration. A ShardTensor carrying the packed routing as an
# inner tensor (``_halo_meta_packed``) gets its scatter_add / index_add corrected via a
# ``__torch_function__`` handler that scatters on the plain local, emits
# :func:`halo_scatter_correct`, and re-wraps. The handler runs in ``__torch_function__`` rather
# than ``__torch_dispatch__`` so the correction stays in the compiled backward. Tensors without
# routing fall through unchanged, so registering is a safe opt-in.


def register_halo_scatter_handlers() -> None:
    r"""Register ``scatter_add`` / ``index_add`` (and the in-place ``scatter_add_`` /
    ``index_add_``) halo-correction handlers on ``ShardTensor`` (idempotent, opt-in).

    Handlers apply :func:`halo_scatter_correct` only to tensors carrying a non-empty
    ``_halo_meta_packed`` routing inner tensor; all other ShardTensors fall through
    to the default behavior, so registering does not change base behavior.

    For the in-place forms the correction is computed out-of-place (so its backward chains
    cleanly), the corrected values are written back into the accumulator's local storage, and
    the returned value is a fresh autograd-connected wrapper so ``y = agg.scatter_add_(...)``
    differentiates through the correction.
    """
    from physicsnemo.domain_parallel.shard_tensor import (
        ShardTensor,
        _torch_function_fallback_via_dtensor,
    )

    def _local(x):
        return x._local_tensor if isinstance(x, ShardTensor) else x

    def _routing(self):
        r = getattr(self, "_halo_meta_packed", None)
        return r if (r is not None and r.numel() > 0) else None

    def _needs_grad(*tensors):
        return torch.is_grad_enabled() and any(
            bool(getattr(t, "requires_grad", False))
            or getattr(t, "grad_fn", None) is not None
            for t in tensors
        )

    def _build(src_type, local, spec, routing, requires_grad):
        out = src_type.__new__(
            src_type, local_tensor=local, spec=spec, requires_grad=requires_grad
        )
        out._halo_meta_packed = routing
        return out

    class _WrapLocalAsShard(torch.autograd.Function):
        # Attach a grad_fn to the op-result wrapper so the tangent flows
        # wrapper -> local -> the halo/scatter graph. A wrapper built by a bare
        # ``__new__`` is an autograd leaf, so the halo correction's backward is
        # dropped. Mirrors ``_FromTorchTensor``.
        @staticmethod
        def forward(ctx, local, src_type, spec, routing):
            return _build(src_type, local, spec, routing, local.requires_grad)

        @staticmethod
        def backward(ctx, grad_out):
            g = (
                grad_out._local_tensor
                if isinstance(grad_out, ShardTensor)
                else grad_out
            )
            return g, None, None, None

    def _wrap_like(src, local, routing, requires_grad):
        if requires_grad:
            return _WrapLocalAsShard.apply(local, type(src), src._spec, routing)
        return _build(type(src), local, src._spec, routing, False)

    def _scatter_handler(f, types, args, kwargs):
        self = args[0]
        routing = _routing(self)
        if routing is None:
            return _torch_function_fallback_via_dtensor(f, args, kwargs)
        dim, index, src = args[1], args[2], args[3]
        local_result = f(_local(self), dim, _local(index), _local(src))
        corrected = halo_scatter_correct(local_result, routing, group=self._spec.mesh)
        return _wrap_like(self, corrected, routing, _needs_grad(self, index, src))

    # In-place ``scatter_add_`` / ``index_add_`` map to their out-of-place forms so the
    # correction's backward chains cleanly (an in-place op on the inner tensor would be
    # dropped from the wrapper-subclass backward).
    _inplace_to_oop = {
        torch.Tensor.scatter_add_: torch.Tensor.scatter_add,
        torch.Tensor.index_add_: torch.Tensor.index_add,
    }

    def _scatter_inplace_handler(f, types, args, kwargs):
        self = args[0]
        routing = _routing(self)
        if routing is None:
            return _torch_function_fallback_via_dtensor(f, args, kwargs)
        dim, index, src = args[1], args[2], args[3]
        local_result = _inplace_to_oop[f](_local(self), dim, _local(index), _local(src))
        corrected = halo_scatter_correct(local_result, routing, group=self._spec.mesh)
        out = _wrap_like(self, corrected, routing, _needs_grad(self, index, src))
        # Write the corrected values into this rank's local storage so a caller that reuses the
        # accumulator sees them; the autograd path is the returned wrapper. The detached
        # no_grad copy updates the shared storage without recording an in-place op on a leaf.
        with torch.no_grad():
            self._local_tensor.detach().copy_(corrected)
        return out

    for func in (torch.Tensor.scatter_add, torch.Tensor.index_add):
        ShardTensor.register_function_handler(func, _scatter_handler)
    for func in (torch.Tensor.scatter_add_, torch.Tensor.index_add_):
        ShardTensor.register_function_handler(func, _scatter_inplace_handler)
