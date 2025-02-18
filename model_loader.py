import os
from folder_paths import models_dir
from pathlib import Path
model_folder_path = Path(models_dir)

class JanusModelLoader:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"],),
                "model_dtype": (["default", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"],),
            },
        }
    
    RETURN_TYPES = ("JANUS_MODEL", "JANUS_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "ComfyUI-pxtool"

    def load_model(self, model_name, model_dtype):
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_map = {
            "default": torch.float16,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        try:
            dtype = dtype_map[model_dtype]
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16
        # 构建模型路径
        model_dir = os.path.join( model_folder_path, 
                               "Janus-Pro",
                               os.path.basename(model_name))
        if not os.path.exists(model_dir):
            raise ValueError(f"Local model not found at {model_dir}. Please download the model and place it in the ComfyUI/models/Janus-Pro folder.")
            
        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
        
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        vl_gpt = vl_gpt.to(dtype).to(device).eval()
        
        return (vl_gpt, vl_chat_processor) 