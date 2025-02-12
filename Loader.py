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



class CheckpointLoaderPX:
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
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "ComfyUI-pxtool"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name, unet_dtype, clip_dtype, vae_dtype):
        model_options = {}
        if unet_dtype == "fp8_e4m3fn":
            model_options["unet_dtype"] = torch.float8_e4m3fn
        elif unet_dtype == "fp8_e4m3fn_fast":
            model_options["unet_dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif unet_dtype == "fp8_e5m2":
            model_options["unet_dtype"] = torch.float8_e5m2
        if clip_dtype == "fp8_e4m3fn":
            model_options["clip_dtype"] = torch.float8_e4m3fn
        elif clip_dtype == "fp8_e5m2":
            model_options["clip_dtype"] = torch.float8_e5m2
        if vae_dtype == "fp16":
            model_options["vae_dtype"] = torch.float16
        elif vae_dtype == "fp32":
            model_options["vae_dtype"] = torch.float32
        
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options = model_options)
        return out[:3]
    
NODE_CLASS_loaders = {
    'CheckpointLoaderPX': CheckpointLoaderPX,
    
}