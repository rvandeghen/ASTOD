import datetime
import json
import os
import time
import yaml

import torch
import torch.utils.data
from coco_utils import get_coco
from models import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn

from engine import label

import presets
import utils


def get_dataset(dataset, root, image_set, transform, ann_file):

    if dataset == "coco":
        dataset = get_coco(root, image_set, transform, ann_file, dataset, use_score=False)
        num_classes = 80
    elif dataset == "dior":
        dataset = get_coco(root, image_set, transform, ann_file, dataset, use_score=False)
        num_classes = 20
    elif dataset == "SNv3":
        dataset = get_coco(root, image_set, transform, ann_file, dataset, use_score=False)
        num_classes = 4
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    if image_set != "unlabeled":
        return dataset, num_classes
    else:
        return dataset


def get_transform():
    return presets.DetectionPresetLabel(scale_factor=2., scale=True, flip=True)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="ASTOD Labelling", add_help=add_help)

    # Dataset
    parser.add_argument("--config", help="Config file to load")
    parser.add_argument("--train-file",
                        help="training file")
    parser.add_argument("--unlabeled-file",
                        help="unlabeled file")
    parser.add_argument("--output-dir", default="",
                        help="path where to save")
        
    # Model
    parser.add_argument("--checkpoint",
                        help="checkpoint to load")

    # Score
    parser.add_argument("--score", default=None,
                        help="score threshold")    

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No cuda device detected")
        return

    # Data loading code
    print("Loading data")

    print("Loading training data")
    dataset, num_classes = get_dataset(cfg["dataset"], cfg["data_path"], "train", get_transform(), args.train_file)
    print(dataset.__len__())
    print("Loading unlabeled data")
    dataset_test = get_dataset(cfg["dataset"], cfg["data_path"], "unlabeled", get_transform(), args.unlabeled_file)
    print(dataset_test.__len__())

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=cfg["workers"],
        collate_fn=utils.collate_fn)

    print("Creating model")
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f'mAP: {checkpoint["map"]}')
    anchor_generator = checkpoint["anchor_generator"]
    kwargs = {}
    kwargs["rpn_anchor_generator"] = anchor_generator
    kwargs["max_size"] = 3000
    if cfg["dataset"] == "dior":
        kwargs["box_detections_per_img"] = 600

    if "rcnn" in cfg["model"]:
        kwargs["rpn_anchor_generator"] = anchor_generator
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes+1,
                                        **kwargs)
    elif "retinanet" in cfg["model"]:
        model = retinanet_resnet50_fpn(num_classes=num_classes+1,
                                        **kwargs)

    print("model created")

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Load model
    model_without_ddp.load_state_dict(checkpoint["model"])

    print("Start Labelling")
    start_time = time.time()
    results_ = label(model, data_loader_test, device)

    del model, model_without_ddp

    results_ = utils.all_gather(results_)
    results = {}
    for r in results_:
        results.update(r)

    json.dump(results, open(os.path.join(args.output_dir, "results_annotations.json"), "w"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Labelling time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
