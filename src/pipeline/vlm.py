"""Optional VLM confirmation/description of candidate events (Qwen2.5-VL via vLLM).

No-op when the VLM endpoint is unavailable or --no-vlm is given: the rule template
description is used instead (graceful degradation, never a failure).
"""
