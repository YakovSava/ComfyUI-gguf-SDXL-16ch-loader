from __future__ import annotations


import torch
from comfy import latent_formats
import folder_paths
import comfy.sd


PRECISION_OPTIONS = ["auto", "fp32", "fp16", "bf16"]

DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float64: "fp64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
}


def precision_to_dtype(precision: str) -> Optional[torch.dtype]:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(precision)


def patch_model_16ch(model):
    m = model.clone()
    m.add_object_patch("concat_keys", ())
    m.add_object_patch("latent_format", latent_formats.Flux())
    return m


class Sdxl16ChLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the checkpoint (model) to load."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name: str):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return [patch_model_16ch(model), clip, vae]


class Sdxl16ChLoaderWithPrecision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the checkpoint (model) to load."},
                ),
                "model_precision": (PRECISION_OPTIONS, {"default": "auto", "tooltip": "Precision for the diffusion model (UNet)."}),
                "clip_precision": (PRECISION_OPTIONS, {"default": "auto", "tooltip": "Precision for the CLIP text encoder."}),
                "vae_precision": (PRECISION_OPTIONS, {"default": "auto", "tooltip": "Precision for the VAE encoder/decoder."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint with individual precision control for MODEL, CLIP, and VAE components."

    def load_checkpoint(
        self,
        ckpt_name: str,
        model_precision: str = "auto",
        clip_precision: str = "auto",
        vae_precision: str = "auto",
    ):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)

        model_options: dict = {}
        te_model_options: dict = {}

        if (model_dtype := precision_to_dtype(model_precision)) is not None:
            model_options["dtype"] = model_dtype

        if (clip_dtype := precision_to_dtype(clip_precision)) is not None:
            te_model_options["dtype"] = clip_dtype

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options,
            te_model_options=te_model_options,
        )
        model, clip, vae = out[:3]

        if clip_precision != "auto" and (clip_dtype := precision_to_dtype(clip_precision)):
            clip.cond_stage_model.to(clip_dtype)

        if vae_precision != "auto" and (vae_dtype := precision_to_dtype(vae_precision)):
            vae = comfy.sd.VAE(sd=vae.get_sd(), dtype=vae_dtype)

        return [patch_model_16ch(model), clip, vae]


class Sdxl16ChGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": (
                    folder_paths.get_filename_list("unet_gguf"),
                    {"tooltip": "GGUF file with quantized UNet weights for a 16ch latent space model."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("16ch-patched quantized diffusion model.",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Loads a GGUF-quantized UNet for 16-channel latent space models. "
        "Connect CLIP and VAE separately via DualCLIPLoader and VAELoader nodes."
    )

    def load_checkpoint(self, gguf_name: str):
        from comfy_gguf.loader import gguf_sd_loader
        from comfy_gguf.ops import GGMLOps

        gguf_path = folder_paths.get_full_path_or_raise("unet_gguf", gguf_name)
        sd = gguf_sd_loader(gguf_path)

        model = comfy.sd.load_diffusion_model_state_dict(
            sd,
            model_options={"custom_operations": GGMLOps},
        )

        return (patch_model_16ch(model),)


class TensorDtypeInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model": ("MODEL", {"tooltip": "MODEL to check dtype"}),
                "clip": ("CLIP", {"tooltip": "CLIP to check dtype"}),
                "vae": ("VAE", {"tooltip": "VAE to check dtype"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("The dtype of the input (e.g., fp32, fp16, bf16)",)
    FUNCTION = "get_dtype"
    CATEGORY = "utils"
    DESCRIPTION = "Returns the dtype of MODEL, CLIP, or VAE."

    def get_dtype(self, model=None, clip=None, vae=None):
        if model is not None:
            dtype = next(model.model.parameters()).dtype
        elif clip is not None:
            dtype = next(clip.cond_stage_model.parameters()).dtype
        elif vae is not None:
            dtype = next(vae.first_stage_model.parameters()).dtype
        else:
            return ("no input provided",)

        return (DTYPE_MAP.get(dtype, str(dtype)),)


NODE_CLASS_MAPPINGS = {
    "SDXL 16ch Loader": Sdxl16ChLoader,
    "SDXL 16ch Loader (Precision)": Sdxl16ChLoaderWithPrecision,
    "SDXL 16ch GGUF Loader": Sdxl16ChGGUFLoader,
    "Tensor Dtype Info": TensorDtypeInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXL16ChLoader": "SDXL 16ch Loader",
    "SDXL16ChLoaderWithPrecision": "SDXL 16ch Loader (Precision)",
    "SDXL16ChGGUFLoader": "SDXL 16ch GGUF Loader",
    "TensorDtypeInfo": "Tensor Dtype Info",
}