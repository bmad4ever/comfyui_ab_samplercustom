import comfy
import random
import torch
from .utilities import *
import comfy_extras.nodes_mask
from comfy_extras.nodes_custom_sampler import SamplerCustom


class AB_SamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        types = SamplerCustom.INPUT_TYPES()
        types["required"].pop("positive")
        types["required"].pop("negative")
        types["required"]["cfgA"] = types["required"]["cfg"]
        types["required"]["cfgB"] = types["required"]["cfg"]
        types["required"]["sigmasA"] = types["required"]["sigmas"]
        types["required"]["sigmasB"] = types["required"]["sigmas"]
        types["required"].pop("cfg")
        types["required"].pop("model")
        types["required"].pop("sigmas")
        types["optional"] = {}

        types["required"]["modelA"] = ("MODEL",)
        types["optional"]["modelB"] = ("MODEL",)
        types["required"]["positive_A"] = ("CONDITIONING",)
        types["required"]["negative_A"] = ("CONDITIONING",)
        types["required"]["positive_B"] = ("CONDITIONING",)
        types["required"]["negative_B"] = ("CONDITIONING",)

        types["optional"]["roi_mask"] = ("MASK",)

        return types

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "Bmad/experimental"

    def sample(self, modelA, add_noise, noise_seed,
               cfgA, cfgB, positive_A, negative_A, positive_B, negative_B,
               sampler, sigmasA, sigmasB, latent_image, modelB=None, roi_mask=None):
        if modelB is None:
            modelB = modelA

        latent = latent_image
        latent_image = latent["samples"]
        latent_image_o = None if roi_mask is None else latent_image.clone()
        latent_grid = repeat_into_grid(latent, 2, 2)
        latent_grid_o = latent_grid.clone()
        _, _, height, width = latent_image.size()

        empty_noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                                  device="cpu")
        noise = empty_noise
        fake_noise_grid = torch.zeros(latent_grid.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                                      device="cpu")
        if add_noise:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, noise_seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        # callback = latent_preview.prepare_callback(modelA, sigmasA.shape[-1]+sigmasB.shape[-1] - 2, x0_output)
        total_steps = sigmasA.shape[-1] + sigmasB.shape[-1] - 2  # last sigma is zero
        pbar = comfy.utils.ProgressBar(total_steps)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        composite_node = None if roi_mask is None else comfy_extras.nodes_mask.LatentCompositeMasked()
        random.seed(noise_seed)
        # random_seeds = [random.randint(0, sys.maxsize) for _ in range(sigmasA.shape[-1]+sigmasB.shape[-1])]
        min_steps_ab = min(sigmasA.shape[-1], sigmasB.shape[-1]) - 1
        for i in range(0, min_steps_ab):
            # A step
            target_latent = comfy.sample.sample_custom(
                modelA, noise, cfgA, sampler, sigmasA[i:i + 2], positive_A, negative_A, latent_image,
                noise_mask=noise_mask, callback=None, disable_pbar=disable_pbar, seed=noise_seed)
            # noise_seed = random_seeds.pop()

            if i == 0:
                noise = empty_noise

            latent_image, latent_grid = setup_latents_ab(composite_node, width, height,
                                                         latent_image, latent_image_o,
                                                         latent_grid, latent_grid_o,
                                                         roi_mask, target_latent,
                                                         this_step=STEP_A, next_step=STEP_B)

            # B step
            target_latent = comfy.sample.sample_custom(
                modelB, fake_noise_grid, cfgB, sampler, sigmasB[i:i + 2], positive_B, negative_B, latent_grid,
                noise_mask=noise_mask, callback=None, disable_pbar=disable_pbar, seed=noise_seed)

            # check if last step, and, if so, change to STEP_B if B has more steps
            next_step = STEP_A
            if i + 1 == min_steps_ab and sigmasB.shape[-1] > sigmasA.shape[-1]:
                next_step = STEP_B
            latent_image, latent_grid = setup_latents_ab(composite_node, width, height,
                                                         latent_image, latent_image_o,
                                                         latent_grid, latent_grid_o,
                                                         roi_mask, target_latent,
                                                         this_step=STEP_B, next_step=next_step)
            # noise_seed = random_seeds.pop()
            pbar.update_absolute(i * 2, total_steps)

        # TAIL (only missing A or B steps)
        if sigmasA.shape[-1] != sigmasB.shape[-1]:  # tail, only A or B steps
            tail_step_type = STEP_A if sigmasA.shape[-1] > sigmasB.shape[-1] else STEP_B
            tail_sigmas, tail_model, tail_cfg, tail_pos, tail_neg, tail_noise = \
                (sigmasA, modelA, cfgA, positive_A, negative_A, noise) \
                    if tail_step_type is STEP_A \
                    else (sigmasB, modelB, cfgB, positive_B, negative_B, fake_noise_grid)

            for i in range(min_steps_ab, max(sigmasA.shape[-1], sigmasB.shape[-1]) - 1):
                target_latent = latent_image if tail_step_type is STEP_A else latent_grid
                target_latent = comfy.sample.sample_custom(
                    tail_model, tail_noise, tail_cfg, sampler, tail_sigmas[i:i + 2], tail_pos, tail_neg, target_latent,
                    noise_mask=noise_mask, callback=None, disable_pbar=disable_pbar, seed=noise_seed)

                latent_image, latent_grid = setup_latents_ab(composite_node, width, height,
                                                             latent_image, latent_image_o,
                                                             latent_grid, latent_grid_o,
                                                             roi_mask, target_latent,
                                                             this_step=tail_step_type, next_step=tail_step_type)

                pbar.update_absolute(min_steps_ab + i, total_steps)
                # noise_seed = random_seeds.pop()

        out = latent.copy()
        out["samples"] = latent_image
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = modelA.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)


NODE_CLASS_MAPPINGS = {
    "AB SamplerCustom (experimental)": AB_SamplerCustom,
}
