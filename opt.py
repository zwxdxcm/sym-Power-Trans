import argparse
import os
import json

from util.misc import gen_cur_time


def parse_args():
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save_dir", type=str, default="./log")
    parser.add_argument("--tag", type=str, default="exp")
    parser.add_argument("--up_folder_name", type=str, default="archive")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    # training data & preprocess
    ### dataset 后续再重构吧
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/div2k/test_data/00.png",
        help="can be dataset",
    )
    parser.add_argument(
        "--signal_type",
        type=str,
        default="image",
        choices=["series", "audio", "image", "ultra_image", "sdf", "radiance_field","video"],
    )
    parser.add_argument(
        "--not_zero_mean", action="store_true", help="map [0,1] data to [-1,1]"
    )

    # model settings
    parser.add_argument(
        "--model_type",
        type=str,
        default="siren",
        choices=["siren", "finer", "wire", "gauss", "pemlp", "branch_siren"],
    )
    parser.add_argument("--hidden_layers", type=int, default=3, help="hidden_layers")
    parser.add_argument(
        "--hidden_features", type=int, default=256, help="hidden_features"
    )
    parser.add_argument(
        "--first_omega", type=float, default=30, help="(siren, wire, finer)"
    )
    parser.add_argument(
        "--hidden_omega", type=float, default=30, help="(siren, wire, finer)"
    )
    # Finer
    parser.add_argument(
        "--first_bias_scale",
        type=float,
        default=None,
        help="bias_scale of the first layer",
    )
    parser.add_argument("--scale_req_grad", action="store_true")
    # PEMLP
    parser.add_argument("--N_freqs", type=int, default=10, help="(PEMLP)")
    # Gauss, Wire
    parser.add_argument("--scale", type=float, default=30, help="simga (wire, guass)")

    # trainer
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    # loss
    parser.add_argument(
        "--lambda_l", type=float, default=0, help="laplacian_loss, 1e-5 achive best"
    )

    # log
    parser.add_argument(
        "--log_epoch", type=int, default=1000, help="log performance during training"
    )
    parser.add_argument(
        "--snap_epoch", type=int, default=5000, help="save fitted data during training"
    )
    # method
    parser.add_argument(
        "--method_type",
        type=str,
        default="default",
        choices=["default", "dev", "partition", "diner"],
    )

    # normalization
    parser.add_argument(
        "--normalization",
        type=str,
        default="min_max",
        choices=[
            "min_max",
            "instance",
            "channel",
            "power",
            "power_channel",
            "box_cox",
            "z_power",
        ],
    )
    parser.add_argument("--amp", type=float, default=1.0, help="amplitude")
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="gamma for nonlinear normlization"
    )
    parser.add_argument(
        "--pn_cum", type=float, default=0.5, help="cdf_index for power normalization"
    )
    parser.add_argument(
        "--pn_buffer",
        type=float,
        default=-1,
        help="buffer max and min val to solve long-tail | if == -1 → adpative | 0 == no work | >0 assign the value",
    )
    parser.add_argument(
        "--box_shift",
        type=float,
        default=0.1,
        help="+ shift*(max-min) to make it positive",
    )
    parser.add_argument(
        "--pn_alpha",
        type=float,
        default=0.01,
        help="hyper parmam of pn adaptive buffer",
    )
    parser.add_argument(
        "--pn_k", type=float, default=256.0, help="hyper parmam of pn adaptive buffer"
    )
    parser.add_argument(
        "--pn_beta",
        type=float,
        default=0.05,
        help="hyperparam for inverse edge calibiration",
    )
    parser.add_argument(
        "--pre_study", action="store_true", help="prestudy skewness & varaition"
    )
    parser.add_argument(
        "--skew_a", type=float, default=0.0, help="skew_a"
    )
    parser.add_argument(
        "--skew_w1", type=float, default=0.5
    )
    parser.add_argument(
        "--gamma_boundary", type=float, default=5
    )
    parser.add_argument(
        "--force_gamma", type=float, default=0, help="force setting gamma (for experiments)"
    )
    # preprocess
    parser.add_argument(
        "--inverse", action="store_true", help="inverse pixels"
    )
    parser.add_argument(
        "--rpp", action="store_true", help="random pixel permutation"
    )







    # method: dev
    # parser.add_argument('--use_coff_net', action='store_true')
    parser.add_argument("--coff_mode", type=int, default=7)
    parser.add_argument("--dct_block_size", type=int, default=8)
    parser.add_argument("--dual_low_pass", type=int, default=4)
    parser.add_argument("--permute_block", action="store_true")  # todo: 未集成
    parser.add_argument("--block_normalization", action="store_true")
    parser.add_argument(
        "--cross_ratio", type=float, default=0.6, help="cross train spatial()"
    )

    args = parser.parse_args()
    return args


def opt():
    args = parse_args()

    up_folder_name = args.up_folder_name

    if args.debug:
        up_folder_name = "debug"

    cur_time = gen_cur_time()
    args.running_name = f"{args.tag}_{cur_time}"

    args.save_folder = os.path.join(args.save_dir, up_folder_name, args.running_name)
    os.makedirs(args.save_folder, exist_ok=True)

    with open(os.path.join(args.save_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return args
