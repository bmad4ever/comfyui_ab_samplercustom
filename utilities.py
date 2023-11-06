STEP_A = True
STEP_B = False


def repeat_into_grid(samples, columns, rows):
    return samples['samples'].repeat(1, 1, rows, columns)


def setup_latents_ab(composite_node, width, height, latent_image, latent_image_o, latent_grid, latent_grid_o,
                     roi_mask, target_latent, this_step, next_step):
    # get current step result in the same size auxiliary latent
    if this_step is STEP_A:
        latent_image = target_latent
    else:
        latent_grid = target_latent
        latent_image[0, :, :, :] = latent_grid[0, :, 0:height, 0:width]

    # composite with the original using the provided mask
    if roi_mask is not None:
        latent_image = composite_node.composite(
            {"samples": latent_image_o}, {"samples": latent_image}, 0, 0, False, roi_mask)[0]["samples"]

    # NOTE: latent_image is already ready for A step

    # setup grid latent for B step
    if next_step is STEP_B:
        latent_grid[:, :, :, :] = latent_grid_o[:, :, :, :]
        latent_grid[0, :, 0:height, 0:width] = latent_image[0, :, :, :]

    return latent_image, latent_grid
