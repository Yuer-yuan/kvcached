# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
from importlib import import_module


def autopatch_all() -> None:
    """Register import hooks only for the selected serving engine.

    Importing both hook sets is expensive on tight UMA systems: SGLang imports
    small pieces of vLLM for compatibility, which otherwise activates the full
    vLLM patch stack and leaves unused runtime modules resident.  The empty
    selector preserves upstream auto-detection behavior.
    """
    selected_engine = os.getenv("KVCACHED_ENGINE", "").strip().lower()
    if selected_engine not in ("", "sglang", "vllm"):
        raise ValueError(
            "KVCACHED_ENGINE must be empty, 'sglang', or 'vllm'; "
            f"got {selected_engine!r}"
        )

    # Importing these modules registers their when_imported hooks.
    if selected_engine in ("", "vllm"):
        try:
            import_module("kvcached.integration.vllm.autopatch")
        except Exception:
            pass
    if selected_engine in ("", "sglang"):
        try:
            import_module("kvcached.integration.sglang.autopatch")
        except Exception:
            pass


autopatch_all()
