# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict

from wrapt.importer import when_imported

from kvcached.integration.patch_base import PatchManager, log_patch_results
from kvcached.integration.sglang.patches import (
    SGLANG_ALL_RANGE,
    ElasticAllocatorPatch,
    ElasticHybridLinearKVPoolPatch,
    ElasticMambaPoolPatch,
    ElasticMemoryPoolPatch,
    ElasticMLAMemoryPoolPatch,
    ElasticSWAAllocatorPatch,
    RadixCacheLimitPatch,
    SchedulerMemoryLeakPatch,
    SGLangVirtualKVCapacityPatch,
)
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


def _register_target_module_hooks() -> None:
    """Patch each SGLang module after it is initialized, before consumers run.

    Hooking the top-level ``sglang`` package is too early for SGLang 0.4.x:
    its ``__init__`` eagerly imports the serving stack, so importing allocator
    targets from that callback encounters partially initialized attention
    modules.  Target-module hooks preserve the required class-capture order
    without importing any SGLang module prematurely.
    """
    patch_entries = [
        (ElasticAllocatorPatch(), SGLANG_ALL_RANGE),
        # SWATokenToKVPoolAllocator captures allocator classes from its
        # implementation modules, not from the package aliases above.
        (ElasticSWAAllocatorPatch(), ">=0.5.13"),
        (ElasticMemoryPoolPatch(), SGLANG_ALL_RANGE),
        (ElasticMLAMemoryPoolPatch(), SGLANG_ALL_RANGE),
        (ElasticMambaPoolPatch(), SGLANG_ALL_RANGE),
        (ElasticHybridLinearKVPoolPatch(), SGLANG_ALL_RANGE),
        # Importing ModelRunner captures memory-pool classes in module globals,
        # so the memory_pool hook above must run before model_runner finishes.
        (SGLangVirtualKVCapacityPatch(), SGLANG_ALL_RANGE),
        (SchedulerMemoryLeakPatch(), SGLANG_ALL_RANGE),
        (RadixCacheLimitPatch(), SGLANG_ALL_RANGE),
    ]

    entries_by_module = defaultdict(list)
    for patch, version_range in patch_entries:
        entries_by_module[patch.target_module].append((patch, version_range))

    for target_module, entries in entries_by_module.items():

        @when_imported(target_module)
        def _patch_target(_module, entries=tuple(entries)) -> None:
            if not _env_enabled():
                logger.debug("Disabled by KVCACHED_AUTOPATCH")
                return

            patch_manager = PatchManager("sglang")
            patch_manager.register_patches_with_versions(list(entries))
            results = patch_manager.apply_all_patches()
            log_patch_results("sglang", results)


_register_target_module_hooks()
