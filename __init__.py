import base64
import logging
import shutil
import threading
from functools import wraps
from io import BytesIO
from .pxtool import NODE_CLASS_MAPPINGS
from .azw_nodes import NODE_CLASS_MAPPINGS2
from .model_loader import JanusModelLoader
from .image_understanding import JanusImageUnderstanding
from .Loader import NODE_CLASS_loaders
NODE_CLASS_MAPPINGS3 = {
    "JanusModelLoader": JanusModelLoader,
    "JanusImageUnderstanding": JanusImageUnderstanding,
}


NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **NODE_CLASS_MAPPINGS2, **NODE_CLASS_MAPPINGS3, **NODE_CLASS_loaders}

__all__ = ["NODE_CLASS_MAPPINGS"]