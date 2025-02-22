import base64
import logging
import shutil
import threading
from functools import wraps
from io import BytesIO
from .pxtool import NODE_CLASS_MAPPINGS,NODE_DISPLAY_NAME_MAPPINGS
from .azw_nodes import NODE_CLASS_MAPPINGS2, NODE_DISPLAY_NAME_MAPPINGS2
from .model_loader import JanusModelLoader
from .image_understanding import JanusImageUnderstanding
from .Loader import NODE_CLASS_loaders, NODE_DISPLAY_NAME_MAPPINGS4
NODE_CLASS_MAPPINGS3 = {
    "JanusModelLoader": JanusModelLoader,
    "JanusImageUnderstanding": JanusImageUnderstanding,
}
NODE_DISPLAY_NAME_MAPPINGS3 = {
    "JanusModelLoader": "Janus模型加载器",
    "JanusImageUnderstanding": "Janus图像理解器",
}


NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **NODE_CLASS_MAPPINGS2, **NODE_CLASS_MAPPINGS3, **NODE_CLASS_loaders}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **NODE_DISPLAY_NAME_MAPPINGS2, **NODE_DISPLAY_NAME_MAPPINGS3, **NODE_DISPLAY_NAME_MAPPINGS4}

__all__ = ["NODE_CLASS_MAPPINGS"]