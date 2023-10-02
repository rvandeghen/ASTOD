import datetime
import os
import time
from collections import Counter

import math
import sys
import yaml

import numpy as np
import random

import torch
import torch.utils.data
from models import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, AnchorGenerator
from group_by_aspect_ratio import InfiniteSampler
from coco_utils import get_coco

from engine import evaluate

import presets
import utils


def get_dataset(dataset, root, image_set, transform, ann_file, use_score):

    if dataset == "coco":
        dataset = get_coco(root, image_set, transform, ann_file, dataset, use_score)
        num_classes = 80
    elif dataset == "dior":
        dataset = get_coco(root, image_set, transform, ann_file, dataset, use_score)
        num_classes = 20
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return dataset, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation=data_augmentation) if train else presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="ASTOD Detection Training", add_help=add_help)

    parser.add_argument("--config", help="Config file to load")
    parser.add_argument("--train-file",
                        help="training file")
    parser.add_argument("--output-dir", default="",
                        help="path where to save, leave empty if no saving")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--split", default=10, type=int)
    parser.add_argument("--resume", default="", type=str,
                        help="resume from checkpoint")
        
    # Model
    parser.add_argument("--rpn-score-thresh", default=None, type=float,
                        help="rpn score threshold for faster-rcnn")
    parser.add_argument("--trainable-backbone-layers", default=None, type=int,
                        help="number of trainable layers of backbone")


    # Misc
    parser.add_argument("--test-only", action="store_true",
                        help="Only test the model")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://",
                        help="url used to set up distributed training")

    return parser


def main(args):

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    print(args)
    print(cfg)
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    for k, v in cfg.items():
        print(f"{k}: {v}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No cuda device detected")
        return
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Using seed: {torch.initial_seed()}")

    # Data loading code
    print("Loading data")

    print("Loading training data")
    dataset, num_classes = get_dataset(cfg["dataset"], cfg["data_path"], "train", get_transform(True, cfg["data_augmentation"]), args.train_file, cfg["use_score"])
    print(f"Number of classes: {num_classes}")
    print(f"Number of images: {len(dataset)}")
    print("Loading eval data")
    dataset_test, _ = get_dataset(cfg["dataset"], cfg["data_path"], "eval", get_transform(False, cfg["data_augmentation"]), args.train_file, cfg["use_score"])

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_sampler = InfiniteSampler(train_sampler)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, cfg["batch_size"], drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=cfg["workers"],
        collate_fn=utils.collate_fn, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=cfg["workers"],
        collate_fn=utils.collate_fn, pin_memory=True)

    print("Creating model")

    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers
    }

    if cfg["data_augmentation"] in ["hard", "lsj"]:
        kwargs["_skip_resize"] = True

    if cfg["dataset"] == "dior":
        scales = tuple((x * 0.186, x * 0.281, x * 0.895) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.387, 1.0, 2.586),) * len(scales)
        kwargs["box_detections_per_img"] = 600
    elif cfg["dataset"] == "coco":
        if "rcnn" in cfg["model"]:
            scales = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(scales)
        elif "retinanet" in cfg["model"]:
            scales = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(scales)

    anchor_generator = AnchorGenerator(scales, aspect_ratios)

    thresholds = torch.tensor(dataset.dataset.thresholds)
    print(thresholds)
    kwargs["tau_l"] = thresholds.to(device)
    kwargs["tau_h"] = torch.tensor(1, dtype=torch.float32, device=device)
    if args.test_only:
        kwargs["box_score_thresh"] = 0.001
    if "rcnn" in cfg["model"]:
        kwargs["rpn_anchor_generator"] = anchor_generator
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes+1,
                                        **kwargs)
    elif "retinanet" in cfg["model"]:
        model = retinanet_resnet50_fpn(num_classes=num_classes+1,
                                        **kwargs)
    print("model created")

    model.to(device)
    if args.distributed and cfg["sync_bn"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])

    if cfg["lr_scheduler"] == "multisteplr":
        if cfg["lr_step"] is None:
            cfg["lr_step"] = [int(cfg["iterations"] * x) for x in [2/3, 8/9]]
        print(f'Number of iterations: {cfg["iterations"]}')
        print(f'Learning rate steps: {cfg["lr_step"]}')

    if cfg["lr_scheduler"] == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["lr_step"], gamma=cfg["lr_gamma"])
    elif cfg["lr_scheduler"] == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=cfg["lr_gamma"], patience=5, verbose=True)
    else:
        raise RuntimeError(f'Invalid lr scheduler "{cfg["lr_scheduler"]}". Only MultiStepLR and ReduceOnPlateau '
                           "are supported.")

    best_map = 0.
    start_iter = 0
    if args.resume:
        print("Resuming")
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f'mAP: {checkpoint["map"]}')
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if not cfg["finetune"]:
            start_iter = checkpoint["iter"]
            print(f'Resuming at iteration: {start_iter}')
        if cfg["finetune"]:
            for g in optimizer.param_groups:
                g["lr"] = cfg["lr"]
            lr_scheduler.last_epoch = -1
            lr_scheduler._step_count = 0
            lr_scheduler.base_lrs = cfg["lr"]
            lr_scheduler.milestones = Counter(cfg["lr_step"])
        lr_scheduler.best = checkpoint["map"]
        best_map = checkpoint["map"]
    
    if start_iter == 0 and not cfg["finetune"]:
        warmup_factor = 1.0 / 1000
        warmup_iters = 1000
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    else:
        warmup_scheduler = None

    if args.test_only:
        stats = evaluate(model, data_loader_test, device=device)
        print(stats[0]*100)
        checkpoint["map_1e-3"] = stats[0]*100
        utils.save_on_master(
            checkpoint,
            os.path.join("/".join(args.resume.split("/")[:-1]), "best_model.pth"))
        return

    iter_eval = cfg["eval_freq"]
    print(f"Evaluating at frequency: {iter_eval}")

    iter_loader = iter(data_loader)
    header = "Training:"
    metric_logger = utils.MetricLoggerIter(delimiter="  ", header=header, num_iter=cfg["iterations"], print_freq=50, start_iter=start_iter)
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    print("Start training")
    start_time = time.time()
    for itr in range(start_iter, cfg["iterations"]):
        model.train()
        metric_logger.before_data()
        images, targets = next(iter_loader)
        metric_logger.before_model()

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            optimizer.zero_grad()
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        if cfg["lr_scheduler"] != "plateau":
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.after_model()

        if (itr+1) % iter_eval == 0:
            print("Start evaluating")
            stats = evaluate(model, data_loader_test, device=device)
            if cfg["lr_scheduler"] == "plateau":
                lr_scheduler.step(100*stats[0])
            print(f"mAP .50:.95 = {100*stats[0]}")

            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "anchor_generator": anchor_generator,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "cfg": cfg,
                    "iter": itr,
                    "map": 100*stats[0]
                }

                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, "checkpoint.pth"))

            if 100*stats[0] > best_map:
                print(f"Improved mAP .50:.95 from {best_map} to {100*stats[0]} (delta = {100*stats[0]-best_map})")
                best_map = 100*stats[0]
                counter = 0
                if args.output_dir:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(args.output_dir, "best_model.pth"))

            if args.distributed:
                train_sampler.sampler.set_epoch(itr)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    print(f"Best mAP .50: {best_map}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
