import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        "-beg", "--begin", type=str, help="begin", required=False
    )
    parser.add_argument(
        "-ed", "--end", type=str, help="end", required=False
    )
    parser.add_argument(
        "-p_t", "--prompt_tgt", type=str, help="The prompt for the desired editing", required=False
    )
    parser.add_argument(
        "-p_s", "--prompt_src", type=str, help="The prompt from the source", required=False
    )
    parser.add_argument(
        "-i", "--init_image", type=str, help="The path to the source image input", required=False
    )
    parser.add_argument(
        "-r", "--ref_image", type=str, help="The path to the reference image input", required=False
    )
    parser.add_argument("--mask", type=str,
                        help="The path to the mask to edit with", default=None)

    # Diffusion
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="100",
    )

    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=25,
    )
    parser.add_argument(
        "--local_clip_guided_diffusion",
        help="Indicator for using local CLIP guided diffusion (for baseline comparison)",
        action="store_true",
        dest="local_clip_guided_diffusion",
    )
    parser.add_argument(
        "--data",
        type=str,
        default='imagenet',
        help='imagenet;celeba;ffhq'
    )
    parser.add_argument(
        "--enc",
        type=str,
        default='imagenet',
        help='(DDIB) diffusion model for forward step'
    )
    parser.add_argument(
        "--dec",
        type=str,
        default='imagenet',
        help='(DDIB) diffusion model for reverse step'
    )

    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512],
    )

    # Augmentations
    parser.add_argument("--aug_num", type=int,
                        help="The number of augmentation", default=8)
    parser.add_argument("--aug_prob", type=float,
                        help="The probability of augmentation", default=1)
    parser.add_argument("--n_patch", type=int,
                        help="The number of patches", default=32)
    parser.add_argument("--patch_min", type=float,
                        help="Mininum patch scale", default=0.01)
    parser.add_argument("--patch_max", type=float,
                        help="Maximum patch scale", default=0.05)

    # Loss
    parser.add_argument(
        "--l_gram",
        type=float,
        help="",
        default=0,
    )
    parser.add_argument(
        "--l_clip_global",
        type=float,
        help="",
        default=0,
    )
    parser.add_argument(
        "--l_clip_global_patch",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=0,
    )
    parser.add_argument(
        "--l_clip_dir",
        type=float,
        help="",
        default=0,
    )
    parser.add_argument(
        "--l_clip_dir_patch",
        type=float,
        help="",
        default=1000,
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=50,
    )
    parser.add_argument(
        "--l_vgg",
        type=float,
        help="",
        default=0,
    )
    parser.add_argument(
        "--l_mse",
        type=float,
        help="",
        default=0,
    )
    parser.add_argument(
        "--l_zecon",
        type=float,
        help="",
        default=0,
    )
    parser.add_argument(
        "--diffusion_type",
        type=str,
        help="forward_backward",
        default="ddim_ddpm",
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="stochasticity of DDIM",
        default=0.0,
    )

    # Misc
    parser.add_argument("--seed", type=int,
                        help="The random seed", default=404)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )
    parser.add_argument("--iterations_num", type=int,
                        help="The number of iterations", default=1)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=1,
    )
    parser.add_argument(
        "--vid",
        help="Indicator for saving the video of the diffusion process",
        action="store_true",
        dest="save_video",
    )
    parser.add_argument(
        "--export_assets",
        help="Indicator for saving raw assets of the prediction",
        action="store_true",
        dest="export_assets",
    )

    args = parser.parse_args()
    return args
