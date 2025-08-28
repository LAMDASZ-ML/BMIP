import argparse
import torch
import random
import numpy as np

from dassl.utils import setup_logger, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import os
# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.bitp
import trainers.maple
import trainers.independentVL
import trainers.vpt
import trainers.proda
import trainers.upt
import trainers.promptsrc_bitp
import trainers.promptsrc_bmip
import trainers.bmip


def set_random_seed(cfg):
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if cfg.MODEL.INIT_SEED != -1:
        print("Setting fixed init seed: {}".format(cfg.MODEL.INIT_SEED))
        torch.manual_seed(cfg.MODEL.INIT_SEED)
        torch.cuda.manual_seed_all(cfg.MODEL.INIT_SEED)
    else:
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)
    print("fix cudnn")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    if args.root:
        cfg.DATASET.ROOT = args.root
        print(args.root)

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.device:
        cfg.DEVICE = args.device



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
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MAPLE
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MAPLEold (J=1)
    cfg.TRAINER.MAPLE.CONNECT_METHOD = "add"
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 1  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 1  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MAPLE (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MAPLE (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for BITP
    cfg.TRAINER.BITP = CN()
    cfg.TRAINER.BITP.N_CTX = 2  # number of context vectors
    cfg.TRAINER.BITP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.BITP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.BITP.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow BITP (J=1)
    cfg.TRAINER.BITP.CONNECT_METHOD = "add"
    # add: MLP
    # cfg.TRAINER.BITP.HIDDEN_LAYERS_NUM = 2
    # cfg.TRAINER.BITP.ACTIVATION = "sigmoid"

    # Config for BMIP    
    cfg.TRAINER.BMIP = CN()
    cfg.TRAINER.BMIP.N_CTX = 2  # number of context vectors
    cfg.TRAINER.BMIP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.BMIP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.BMIP.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow BITP (J=1)
    cfg.TRAINER.BMIP.CONNECT_METHOD = "None"
    # add: weight_threshold
    cfg.TRAINER.BMIP.WEIGHT_THRESHOLD = 0.0025
    cfg.MODEL.INIT_SEED = 48

    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX = 4  # number of context vectors
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow PROMPTSRC (J=1)
    cfg.TRAINER.PROMPTSRC.CONNECT_METHOD = "None"
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for ProDA
    cfg.TRAINER.ProDA = CN()
    cfg.TRAINER.ProDA.N_CTX = 16
    cfg.TRAINER.ProDA.N_PROMPT = 32
    cfg.TRAINER.ProDA.PREC = "fp16"  # fp16, fp32, amp
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.UPT = CN()
    cfg.TRAINER.UPT.N_CTX = 2  # number of context vectors at the vision branch
    cfg.TRAINER.UPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.UPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.UPT.PROMPT_DEPTH = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    # Config for sharing parameters
    cfg.SHARE = CN()
    cfg.SHARE.SHARE_PARAMETER_VISUAL = True
    cfg.SHARE.SHARE_PARAMETER_TEXT = True

    # Config for early stopping
    cfg.TRAIN.EARLY_STOPPING = True
    cfg.TRAIN.PATIENT_COUNT = 3
    cfg.TEST.NO_TEST_EARLY = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 2. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg)
    # 设置输出目录
    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    if args.eval_only:
        if args.best:
            trainer.load_model(args.model_dir)
        else:
            trainer.load_model(args.model_dir, epoch=args.load_epoch)
        if args.draw:
            trainer.cal_distance()
        else:
            trainer.test()
        return

    if not args.no_train:
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
    parser.add_argument("--device", default='cuda:0', type=str, help="the device you will use")
    parser.add_argument("--best", action="store_true", help="load best model")
    parser.add_argument("--draw", action="store_true", help="draw distribution")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
    os._exit(0)




