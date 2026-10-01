"""Multi-stream hyper-connection (mHC) kernels: DeepSeek-V4 (mhc.py) and GLM-5.3 (glm5_mhc.py)."""

from sgl_jax.srt.kernels.mhc.glm5_mhc import glm5_mhc_post, glm5_mhc_pre
from sgl_jax.srt.kernels.mhc.mhc import (
    mhc_gates,
    mhc_head_collapse_fused,
    mhc_post_fused,
    mhc_pre_fused,
)

__all__ = [
    "glm5_mhc_post",
    "glm5_mhc_pre",
    "mhc_gates",
    "mhc_head_collapse_fused",
    "mhc_post_fused",
    "mhc_pre_fused",
]
