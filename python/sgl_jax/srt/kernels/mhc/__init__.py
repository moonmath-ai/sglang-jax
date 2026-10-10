"""Multi-stream hyper-connection (mHC) kernels (DeepSeek-V4, also used by GLM-5.3-Flash)."""

from sgl_jax.srt.kernels.mhc.mhc import (
    mhc_gates,
    mhc_head_collapse_fused,
    mhc_post_fused,
    mhc_pre_fused,
)

__all__ = [
    "mhc_gates",
    "mhc_head_collapse_fused",
    "mhc_post_fused",
    "mhc_pre_fused",
]
