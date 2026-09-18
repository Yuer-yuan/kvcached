# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for logical KV pools smaller than one VMM page."""

from __future__ import annotations

import importlib


PAGE_SIZE = 2 * 1024 * 1024


class _NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


class _Page:
    def __init__(self, page_id: int, page_size: int):
        self.page_id = page_id
        self.page_size = page_size
        self._all = []
        self._free = []

    @staticmethod
    def get_block_range(page_id, page_size, block_mem_size):
        start = (page_id * page_size + block_mem_size - 1) // block_mem_size
        end = ((page_id + 1) * page_size) // block_mem_size
        return start, end

    @staticmethod
    def get_num_blocks(page_size, block_mem_size):
        return page_size // block_mem_size

    def init(self, block_mem_size):
        start, end = self.get_block_range(
            self.page_id, self.page_size, block_mem_size
        )
        self._all = list(range(start, end))
        self._free = self._all.copy()

    def alloc(self, count):
        if count > len(self._free):
            raise RuntimeError("Not enough free blocks in page")
        result = self._free[:count]
        del self._free[:count]
        return result

    def free_batch(self, block_ids):
        self._free.extend(block_ids)

    def empty(self):
        return len(self._free) == len(self._all)

    def full(self):
        return not self._free

    def num_free_blocks(self):
        return len(self._free)

    def get_free_blocks(self):
        return self._free.copy()


class _PageAllocator:
    def __init__(
        self,
        num_layers,
        mem_size_per_layer,
        page_size,
        world_size,
        **kwargs,
    ):
        self.mem_size_per_layer = mem_size_per_layer
        self.page_size = page_size
        self.free_page_ids = list(range(mem_size_per_layer // page_size))
        self.inuse_page_ids = set()

    def set_use_worker_ipc(self, enabled):
        pass

    def get_resize_target(self):
        return 0

    def get_num_free_pages(self):
        return len(self.free_page_ids)

    def get_avail_physical_pages(self):
        return 32

    def get_num_reserved_pages(self):
        return 0

    def alloc_page(self):
        if not self.free_page_ids:
            raise RuntimeError("No free pages left")
        page_id = self.free_page_ids.pop(0)
        self.inuse_page_ids.add(page_id)
        return _Page(page_id, self.page_size)

    def free_pages(self, page_ids):
        for page_id in page_ids:
            self.inuse_page_ids.remove(page_id)
            self.free_page_ids.append(page_id)

    def group_indices_by_page(self, indices, block_mem_size):
        grouped = {}
        for block_id in indices:
            page_id = block_id * block_mem_size // self.page_size
            grouped.setdefault(page_id, []).append(block_id)
        return grouped

    def start_prealloc_thread(self):
        pass


def _manager(monkeypatch):
    module = importlib.import_module("kvcached.kv_cache_manager")
    monkeypatch.setattr(module, "PAGE_SIZE", PAGE_SIZE)
    monkeypatch.setattr(module, "PageAllocator", _PageAllocator)
    monkeypatch.setattr(module, "InternalPage", _Page)
    monkeypatch.setattr(module.threading, "Thread", _NoopThread)

    # Qwen3-0.6B's observed SGLang geometry at max_total_tokens=512:
    # 513 logical slots (including null), 2 KiB per K or V token, and 28
    # local layers.  The old formula produced a 1,050,624-byte per-layer
    # range, which PageAllocator truncated to zero 2 MiB pages.
    manager = module.KVCacheManager(
        num_blocks=513,
        block_size=1,
        cell_size=2048,
        num_layers=28,
        reserve_null_block=True,
        num_kv_buffers=2,
    )
    manager._post_init_done.set()
    return manager


def test_small_logical_pool_gets_one_backing_page(monkeypatch):
    manager = _manager(monkeypatch)

    assert manager.num_blocks == 513
    assert manager.mem_size == PAGE_SIZE
    assert manager.page_allocator.get_num_free_pages() == 1
    assert manager.available_size() == 513

    manager._reserve_null_block()
    assert manager.null_block == [0]
    assert manager.available_size() == 512


def test_page_alignment_padding_never_becomes_logical_capacity(monkeypatch):
    manager = _manager(monkeypatch)
    manager._reserve_null_block()

    first = manager.alloc(512)
    assert first == list(range(1, 513))
    assert manager.available_size() == 0
    assert manager.alloc(1) is None

    # Allocation/free churn on the partial page must still expose only IDs in
    # the configured logical range, not the page's 513..1023 padding IDs.
    manager.free(first)
    second = manager.alloc(512)
    assert second == list(range(1, 513))
    assert max(second) < manager.num_blocks
