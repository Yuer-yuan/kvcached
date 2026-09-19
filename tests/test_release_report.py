"""CPU-only contract tests for exact KV release receipts."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("torch", types.ModuleType("torch"))


BLOCKS_PER_PAGE = 2


class NativeReleaseReport:
    def __init__(
        self,
        *,
        logical_pages: int,
        retained_pages: int,
        unmapped_pages: int,
        released_physical_bytes: int,
        synchronized: bool,
    ) -> None:
        self.logical_pages = logical_pages
        self.retained_pages = retained_pages
        self.unmapped_pages = unmapped_pages
        self.released_physical_bytes = released_physical_bytes
        self.synchronized = synchronized


class FakePage:
    def __init__(self, page_id: int, free_indices: tuple[int, ...]) -> None:
        self.page_id = page_id
        self.free_indices = list(free_indices)

    def free_batch(self, indices: list[int]) -> None:
        self.free_indices.extend(indices)

    def empty(self) -> bool:
        return len(self.free_indices) == BLOCKS_PER_PAGE

    def num_free_blocks(self) -> int:
        return len(self.free_indices)


class FakePageAllocator:
    def __init__(self, native_report: NativeReleaseReport | None = None) -> None:
        self.native_report = native_report
        self.calls: list[list[int]] = []

    def group_indices_by_page(self, indices, block_mem_size):
        grouped = {}
        for index in indices:
            grouped.setdefault(index // BLOCKS_PER_PAGE, []).append(index)
        return grouped

    def free_pages(self, page_ids):
        self.calls.append(list(page_ids))
        return self.native_report


def _install_vmm_stub() -> None:
    stub = types.ModuleType("kvcached.vmm_ops")
    stub.PageAllocator = FakePageAllocator  # type: ignore[attr-defined]
    stub.InternalPage = FakePage  # type: ignore[attr-defined]
    stub.kv_tensors_created = lambda group_id=0: True  # type: ignore[attr-defined]
    stub.map_to_kv_tensors = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    stub.unmap_from_kv_tensors = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = stub


try:
    import kvcached.vmm_ops  # noqa: F401
except ImportError:
    _install_vmm_stub()

from kvcached.kv_cache_manager import KVCacheManager, ReleaseReport  # noqa: E402
from kvcached.locks import NoOpLock  # noqa: E402


def manager_with_page(
    page: FakePage,
    native_report: NativeReleaseReport | None = None,
) -> KVCacheManager:
    manager = object.__new__(KVCacheManager)
    manager.page_size = BLOCKS_PER_PAGE
    manager.block_mem_size = 1
    manager.page_allocator = FakePageAllocator(native_report)
    manager.num_avail_blocks = page.num_free_blocks()
    manager.avail_pages = {page.page_id: page} if page.num_free_blocks() else {}
    manager.full_pages = {} if page.num_free_blocks() else {page.page_id: page}
    manager.reserved_blocks = []
    manager.null_block = None
    manager.in_shrink = False
    manager.target_num_blocks = None
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    return manager


def test_fragmented_page_reports_logical_but_no_physical_release() -> None:
    manager = manager_with_page(FakePage(0, ()))

    report = manager.free([0])

    assert isinstance(report, ReleaseReport)
    assert report.logical_blocks == 1
    assert report.emptied_pages == 0
    assert report.retained_pages == 0
    assert report.unmapped_pages == 0
    assert report.released_physical_bytes == 0
    assert not report.backing_left_slot_ownership
    assert report.mechanism == "fragmented-kv-page"
    assert manager.page_allocator.calls == []


def test_reserved_empty_page_does_not_claim_backing_release() -> None:
    native = NativeReleaseReport(
        logical_pages=1,
        retained_pages=1,
        unmapped_pages=0,
        released_physical_bytes=0,
        synchronized=False,
    )
    manager = manager_with_page(FakePage(0, (1,)), native)

    report = manager.free([0])

    assert report.emptied_pages == 1
    assert report.retained_pages == 1
    assert report.unmapped_pages == 0
    assert not report.backing_left_slot_ownership
    assert report.mechanism == "kvcached-reserved-page"


def test_synchronized_vmm_unmap_and_handle_release_emits_evidence() -> None:
    native = NativeReleaseReport(
        logical_pages=1,
        retained_pages=0,
        unmapped_pages=1,
        released_physical_bytes=8 * 1024 * 1024,
        synchronized=True,
    )
    manager = manager_with_page(FakePage(0, (1,)), native)

    report = manager.free([0])

    assert report.logical_blocks == 1
    assert report.emptied_pages == 1
    assert report.unmapped_pages == 1
    assert report.released_physical_bytes == 8 * 1024 * 1024
    assert report.backing_left_slot_ownership
    assert report.mechanism == "cuda-vmm-unmap-release"


def test_unmapped_without_synchronization_is_not_ownership_proof() -> None:
    native = NativeReleaseReport(
        logical_pages=1,
        retained_pages=0,
        unmapped_pages=1,
        released_physical_bytes=8 * 1024 * 1024,
        synchronized=False,
    )
    manager = manager_with_page(FakePage(0, (1,)), native)

    report = manager.free([0])

    assert report.unmapped_pages == 1
    assert not report.backing_left_slot_ownership
    assert report.mechanism == "vmm-release-unverified"


def test_inconsistent_native_page_totals_are_rejected() -> None:
    native = NativeReleaseReport(
        logical_pages=2,
        retained_pages=0,
        unmapped_pages=1,
        released_physical_bytes=8 * 1024 * 1024,
        synchronized=True,
    )
    manager = manager_with_page(FakePage(0, (1,)), native)

    try:
        manager.free([0])
    except RuntimeError as exc:
        assert "inconsistent logical pages" in str(exc)
    else:
        raise AssertionError("malformed native release receipt was accepted")


def test_empty_free_returns_zero_receipt() -> None:
    manager = manager_with_page(FakePage(0, ()))

    assert manager.free([]) == ReleaseReport.empty()
