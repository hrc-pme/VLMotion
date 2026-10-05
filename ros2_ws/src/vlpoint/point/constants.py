CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

import os
LOGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log")

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"


def vlmotion_gpu_profile_is_5060() -> bool:
    """True only when docker/5060.compose.yaml sets VLMOTION_GPU_PROFILE=5060 on vlmotion."""
    return os.environ.get("VLMOTION_GPU_PROFILE", "").strip() == "5060"
