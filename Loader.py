import comfy.utils
import folder_paths
from comfy import model_management
from comfy.sd import VAE, CLIP
import model_detection
from comfy import clip_vision
import yaml
import logging
import torch

def load_checkpoint(config_path=None, ckpt_path=None, output_vae=True, output_clip=True, embedding_directory=None, state_dict=None, config=None, model_options={}):
    logging.warning("Warning: The load checkpoint with config function is deprecated and will eventually be removed, please use the other one.")
    model, clip, vae, _ = load_checkpoint_guess_config(
        ckpt_path,
        output_vae=output_vae,
        output_clip=output_clip,
        output_clipvision=False,
        embedding_directory=embedding_directory,
        output_model=True,
        model_options=model_options
    )
    # TODO: this function is a mess and should be removed eventually
    if config is None:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']

    if "parameterization" in model_config_params:
        if model_config_params["parameterization"] == "v":
            m = model.clone()
            class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.V_PREDICTION):
                pass
            m.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config))
            model = m

    layer_idx = clip_config.get("params", {}).get("layer_idx", None)
    if layer_idx is not None:
        clip.clip_layer(layer_idx)

    return (model, clip, vae)

def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
    sd = comfy.utils.load_torch_file(ckpt_path)
    out = load_state_dict_guess_config(
        sd,
        output_vae,
        output_clip,
        output_clipvision,
        embedding_directory,
        output_model,
        model_options,
        te_model_options=te_model_options
    )
    if out is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))
    return out

def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None

    # 分别提取各模块的 dtype 参数
    # 对 unet 优先使用 unet_dtype（否则回退到原来的 "dtype" 或 "weight_dtype"）
    unet_dtype = model_options.get("unet_dtype", model_options.get("dtype", model_options.get("weight_dtype", None)))
    vae_dtype = model_options.get("vae_dtype", None)
    clip_dtype = model_options.get("clip_dtype", None)

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix)
    if model_config is None:
        return None

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if weight_dtype is not None and model_config.scaled_fp8 is None:
        unet_weight_dtype.append(weight_dtype)

    model_config.custom_operations = model_options.get("custom_operations", None)
    # 使用已提取的 unet_dtype
    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype)

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
        model.load_model_weights(sd, diffusion_model_prefix)

    if output_vae:
        vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        # 如果提供了 vae_dtype 则传入 VAE 构造函数（假设 VAE 类支持 dtype 参数）
        if vae_dtype is not None:
            vae = VAE(sd=vae_sd, dtype=vae_dtype)
        else:
            vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = comfy.utils.calculate_parameters(clip_sd)
                # 将 clip_dtype 注入到 te_model_options 中（若已有则覆盖）
                te_options = te_model_options.copy() if te_model_options is not None else {}
                if clip_dtype is not None:
                    te_options["dtype"] = clip_dtype
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, model_options=te_options)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded diffusion model directly to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

    return (model_patcher, clip, vae, clipvision)

class CheckpointLoaderSimplePX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
                "unet_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "clip_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                "vae_dtype": (["default", "fp16", "fp32"],),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("模型", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "ComfyUI-pxtool"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name, unet_dtype, clip_dtype, vae_dtype):
        dtype_map = {
            "default": None,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e4m3fn_fast": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        model_options = {}
        if unet_dtype == "fp8_e4m3fn_fast":
            model_options["fp8_optimizations"] = True
        model_options["unet_dtype"] = dtype_map[unet_dtype]
        model_options["clip_dtype"] = dtype_map[clip_dtype]
        model_options["vae_dtype"] = dtype_map[vae_dtype]
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options = model_options)
        return out[:3]

import os
from comfy.controlnet import load_controlnet_state_dict
def load_controlnet(ckpt_path, model=None, model_options={}):
    
    if "global_average_pooling" not in model_options:
        filename = os.path.splitext(ckpt_path)[0]
        if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"): #TODO: smarter way of enabling global_average_pooling
            model_options["global_average_pooling"] = True

    cnet = load_controlnet_state_dict(comfy.utils.load_torch_file(ckpt_path, safe_load=True), model=model, model_options=model_options)
    if cnet is None:
        logging.error("error checkpoint does not contain controlnet or t2i adapter data {}".format(ckpt_path))
    return cnet


class ControlNetLoaderPX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            "control_net_dtype": (["default", "fp8_e4m3fn",  "fp8_e5m2"],),
                             }
                             
            }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("ControlNet",)
    FUNCTION = "load_controlnet_px"

    CATEGORY = "ComfyUI-pxtool"

    def load_controlnet_px(self, control_net_name, control_net_dtype):
        dtype_map = {
            "default": None,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        model_options = {}
        model_options["dtype"] = dtype_map[control_net_dtype]
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path,model_options=model_options)
        return (controlnet,)
import safetensors.torch
ALWAYS_SAFE_LOAD = False
def load_torch_file(ckpt, safe_load=False, device=None, dtype_option="default"):
    """加载 PyTorch 模型文件并支持 FP8 格式转换
    
    参数:
        ckpt:        模型文件路径
        safe_load:   是否启用安全加载 (防止恶意代码执行)
        device:      目标设备 (默认自动设为 CPU)
        dtype_option: 数据类型选项 ["default", "fp8_e4m3fn", "fp8_e5m2"]
    
    返回:
        模型状态字典 (state_dict)
    """
    # 初始化数据类型映射
    dtype_map = {
        "default": None,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    
    # 验证 dtype 选项有效性
    if dtype_option not in dtype_map:
        raise ValueError(f"Invalid dtype_option: {dtype_option}. Valid options: {list(dtype_map.keys())}")
    dtype = dtype_map[dtype_option]

    # 初始化设备
    if device is None:
        device = torch.device("cpu")

    # 加载 safetensors 格式
    if ckpt.lower().endswith((".safetensors", ".sft")):
        try:
            sd = safetensors.torch.load_file(ckpt, device=device.type)
        except Exception as e:
            error_handlers = {
                "HeaderTooLarge": "文件可能已损坏或非 safetensors 格式",
                "MetadataIncompleteBuffer": "文件不完整或下载错误"
            }
            for err_key, err_msg in error_handlers.items():
                if err_key in str(e.args):
                    raise ValueError(f"{err_msg}\n文件路径: {ckpt}") from e
            raise
    # 加载普通 PyTorch 格式
    else:
        load_args = {"map_location": device}
        if safe_load or getattr(comfy.utils, "ALWAYS_SAFE_LOAD", False):
            load_args["weights_only"] = True
        else:
            load_args["pickle_module"] = comfy.checkpoint_pickle
        
        pl_sd = torch.load(ckpt, **load_args)
        
        # 提取状态字典
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd.get(list(pl_sd.keys())[0], pl_sd) if len(pl_sd) == 1 else pl_sd
        
        # 记录训练步数
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")

    # 执行数据类型转换
    if dtype is not None:
        for key in sd:
            tensor = sd[key]
            if isinstance(tensor, torch.Tensor):
                # 保留设备信息同时转换类型
                sd[key] = tensor.to(device=device, dtype=dtype)

    return sd


def ipadapter_model_loader(file, dtype_option="default"):
    # 定义数据类型映射
    dtype_map = {
        "default": None,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    
    # 验证 dtype 选项有效性
    if dtype_option not in dtype_map:
        raise ValueError(f"Invalid dtype_option: {dtype_option}. Valid options are: {list(dtype_map.keys())}")
    dtype = dtype_map[dtype_option]

    # 加载原始模型
    model = comfy.utils.load_torch_file(file, safe_load=True)

    # 处理 safetensors 格式
    if file.lower().endswith(".safetensors"):
        st_model = {"image_proj": {}, "ip_adapter": {}}
        for key in model.keys():
            if key.startswith("image_proj."):
                st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
            elif key.startswith("ip_adapter."):
                st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            elif key.startswith("adapter_modules."):
                st_model["ip_adapter"][key.replace("adapter_modules.", "")] = model[key]
        model = st_model
        del st_model
    # 处理普通权重格式
    elif "adapter_modules" in model.keys():
        model["ip_adapter"] = model.pop("adapter_modules")

    # 有效性检查
    if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
        raise Exception("invalid IPAdapter model {}".format(file))

    # 添加版本标识
    if 'plusv2' in file.lower():
        model["faceidplusv2"] = True
    if 'unnorm' in file.lower():
        model["portraitunnorm"] = True

    # 执行数据类型转换
    if dtype is not None:
        for component in ["image_proj", "ip_adapter"]:
            if component in model:
                for key in model[component]:
                    tensor = model[component][key]
                    model[component][key] = tensor.to(dtype)

    return model

class IPAdapterModelLoaderPX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ipadapter_file": (folder_paths.get_filename_list("ipadapter"), ),
                "ipadapter_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),}
                }

    RETURN_TYPES = ("IPADAPTER",)
    RETURN_NAMES = ("IPAdapter",)
    FUNCTION = "load_ipadapter_model"
    CATEGORY = "ComfyUI-pxtool"

    def load_ipadapter_model(self, ipadapter_file, ipadapter_dtype):
        ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_file)
        return (ipadapter_model_loader(ipadapter_file, ipadapter_dtype),)

class CLIPLoaderPX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2"], ),
                              "clip_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "ComfyUI-pxtool"

    DESCRIPTION = "[Recipes]\n\nstable_diffusion: clip-l\nstable_cascade: clip-g\nsd3: t5 / clip-g / clip-l\nstable_audio: t5\nmochi: t5\ncosmos: old t5 xxl\nlumina2: gemma 2 2B"

    def load_clip(self, clip_name, type="stable_diffusion", device="default", clip_dtype="default"):
        dtype_map = {
            "default": None,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        clip_dtype = dtype_map[clip_dtype]

        if type == "stable_cascade":
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        elif type == "mochi":
            clip_type = comfy.sd.CLIPType.MOCHI
        elif type == "ltxv":
            clip_type = comfy.sd.CLIPType.LTXV
        elif type == "pixart":
            clip_type = comfy.sd.CLIPType.PIXART
        elif type == "cosmos":
            clip_type = comfy.sd.CLIPType.COSMOS
        elif type == "lumina2":
            clip_type = comfy.sd.CLIPType.LUMINA2
        else:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        model_options["dtype"] = clip_dtype

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return (clip,)

class DualCLIPLoaderPX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["sdxl", "sd3", "flux", "hunyuan_video"], ),
                              "clip_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "ComfyUI-pxtool"

    DESCRIPTION = "[Recipes]\n\nsdxl: clip-l, clip-g\nsd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\nflux: clip-l, t5"

    def load_clip(self, clip_name1, clip_name2, type, device="default", clip_dtype="default"):
        dtype_map = {
            "default": None,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        elif type == "hunyuan_video":
            clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        
        model_options["dtype"] = dtype_map[clip_dtype]

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return (clip,)


class TripleCLIPLoaderPX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                             "clip_name2": (folder_paths.get_filename_list("text_encoders"), ), 
                             "clip_name3": (folder_paths.get_filename_list("text_encoders"), ),
                             "clip_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "ComfyUI-pxtool"

    DESCRIPTION = "[Recipes]\n\nsd3: clip-l, clip-g, t5"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_dtype="default"):
        dtype_map = {
            "default": None,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        model_options = {}
        model_options["dtype"] = dtype_map[clip_dtype]
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3], embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options=model_options)
        return (clip,)




NODE_CLASS_loaders = {
    'CheckpointLoaderSimplePX': CheckpointLoaderSimplePX,
    "ControlNetLoaderPX": ControlNetLoaderPX,
    "IPAdapterModelLoaderPX": IPAdapterModelLoaderPX,
    "CLIPLoaderPX": CLIPLoaderPX,
    "DualCLIPLoaderPX": DualCLIPLoaderPX,
    "TripleCLIPLoaderPX": TripleCLIPLoaderPX,
}
NODE_DISPLAY_NAME_MAPPINGS4 = {
    'CheckpointLoaderSimplePX': 'Checkpoint模型加载器FP8',
    "ControlNetLoaderPX": "ControlNet加载器FP8",
    "IPAdapterModelLoaderPX": "IPAdapter加载器FP8",
    "CLIPLoaderPX": "CLIP加载器FP8",
    "DualCLIPLoaderPX": "双CLIP加载器FP8",
    "TripleCLIPLoaderPX": "三CLIP加载器FP8",
}
