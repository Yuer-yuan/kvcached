"""CUDA integration test for native VMM release receipts.

Run this test in a CUDA-enabled kvcached build.  CPU-only development
environments skip it during collection.
"""

from __future__ import annotations

import os

import pytest


torch = pytest.importorskip("torch")
if not hasattr(torch, "cuda") or not torch.cuda.is_available():
    pytest.skip("CUDA is required for native VMM release testing", allow_module_level=True)

# These values are read when the native extension is loaded, so configure a
# no-reserve allocator before importing vmm_ops.  A released empty page must be
# unmapped rather than retained in kvcached's warm-page pool.
os.environ["KVCACHED_PAGE_SIZE_MB"] = "2"
os.environ["KVCACHED_MIN_RESERVED_PAGES"] = "0"
os.environ["KVCACHED_MAX_RESERVED_PAGES"] = "0"

from kvcached import vmm_ops  # noqa: E402


PAGE_BYTES = 2 * 1024 * 1024
NUM_LAYERS = 2
NUM_KV_BUFFERS = 2


def test_native_free_reports_synchronized_physical_release() -> None:
    vmm_ops.init_kvcached("cuda:0", PAGE_BYTES, True)
    try:
        # Keep the virtual tensors alive until after the allocator has mapped
        # and released one compound physical page.
        pools = vmm_ops.create_kv_tensors(
            PAGE_BYTES * NUM_KV_BUFFERS,
            2,
            "cuda:0",
            NUM_LAYERS,
            num_kv_buffers=NUM_KV_BUFFERS,
        )
        allocator = vmm_ops.PageAllocator(
            NUM_LAYERS,
            PAGE_BYTES,
            PAGE_BYTES,
            world_size=1,
            pp_rank=0,
            async_sched=False,
            contiguous_layout=True,
            enable_page_prealloc=False,
            num_kv_buffers=NUM_KV_BUFFERS,
        )

        page = allocator.alloc_page()
        report = allocator.free_page(page.page_id)

        assert pools
        assert report.logical_pages == 1
        assert report.retained_pages == 0
        assert report.unmapped_pages == 1
        assert report.released_physical_bytes == (
            PAGE_BYTES * NUM_LAYERS * NUM_KV_BUFFERS
        )
        assert report.synchronized is True
    finally:
        vmm_ops.shutdown_kvcached()
