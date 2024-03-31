import argparse

import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger
from yacs.config import CfgNode as CN

import datasets.caltech101
import datasets.dtd
import datasets.eurosat
import datasets.fgvc_aircraft
import datasets.food101
import datasets.imagenet
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.oxford_flowers
import datasets.oxford_pets
import datasets.stanford_cars
import datasets.sun397
import datasets.ucf101
from trainers.constants import get_dataset_specified_config
from trainers.coprompt import CoPrompt


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if "root" in args:
        if args.root:
            cfg.DATASET.ROOT = args.root
    if "output_dir" in args:
        if args.output_dir:
            cfg.OUTPUT_DIR = args.output_dir
    if "resume" in args:
        if args.resume:
            cfg.RESUME = args.resume
    if "seed" in args:
        if args.seed:
            cfg.SEED = args.seed
    if "source_domains" in args:
        if args.source_domains:
            cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if "target_domains" in args:
        if args.target_domains:
            cfg.DATASET.TARGET_DOMAINS = args.target_domains
    if "transforms" in args:
        if args.transforms:
            cfg.INPUT.TRANSFORMS = args.transforms
    if "trainer" in args:
        if args.trainer:
            cfg.TRAINER.NAME = args.trainer
    if "backbone" in args:
        if args.backbone:
            cfg.MODEL.BACKBONE.NAME = args.backbone
    if "head" in args:
        if args.head:
            cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    # Default config for CoPrompt MaPLe
    cfg.TRAINER.CoPrompt = CN()
    cfg.TRAINER.CoPrompt.N_CTX = 2  # number of context vectors
    cfg.TRAINER.CoPrompt.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.CoPrompt.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.CoPrompt.PROMPT_DEPTH = 9
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.W = 8.0
    cfg.TRAINER.DISTILL = "cosine"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. Override dataset specific config
    cfg.merge_from_list(get_dataset_specified_config(cfg.DATASET.NAME))

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--pre_train_exp_name",
        type=str,
        default="",
        help="pre-trained experiment name; need to determine the pre-trained checkpoint",
    )
    args = parser.parse_args()
    main(args)
