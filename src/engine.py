import math
import sys
import time
import torch

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator, _derive_coco_results
import utils

from transforms import PseudoLabel
from torchvision.ops import boxes as box_ops

@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    with torch.inference_mode():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    classes = [data_loader.dataset.coco.cats[i]["name"] for i in data_loader.dataset.coco.cats.keys()]
    _derive_coco_results(coco_evaluator.coco_eval["bbox"], "bbox", class_names=classes)

    torch.set_num_threads(n_threads)
    return coco_evaluator.coco_eval["bbox"].stats.tolist()

@torch.inference_mode()
def label(model, data_loader, device):
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    results = {}
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Label:"

    inverse = PseudoLabel(scale_factor=0.5, scale=True, flip=True)

    for batch, image_id in metric_logger.log_every(data_loader, 100, header):
        
        batch = batch[0]
        image_id = image_id[0]

        boxes = []
        scores = []
        labels = []
        for aug, data in batch.items():

            img, _ = data

            with torch.inference_mode():
                outputs = model(img.to(device).unsqueeze(0))[0]
            outputs = {k: v.to(cpu_device) for k, v in outputs.items()}

            if aug == "normal":
                boxes.append(outputs["boxes"])
                scores.append(outputs["scores"])
                labels.append(outputs["labels"])

            if aug == "flip":
                _, outputs = inverse.flip_(img, outputs)
                boxes.append(outputs["boxes"])
                scores.append(outputs["scores"])
                labels.append(outputs["labels"])

            if aug == "scale":
                _, outputs = inverse.scale_(img, outputs)
                boxes.append(outputs["boxes"])
                scores.append(outputs["scores"])
                labels.append(outputs["labels"])

            if aug == "scale_flip":
                img, outputs = inverse.scale_(img, outputs)
                _, outputs = inverse.flip_(img, outputs)
                boxes.append(outputs["boxes"])
                scores.append(outputs["scores"])
                labels.append(outputs["labels"])

        if len(boxes) != 0:
            boxes = torch.vstack(boxes)
            scores = torch.cat(scores)
            labels = torch.cat(labels)

        keep = box_ops.batched_nms(boxes, scores, labels, 0.5)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        results[image_id] = {"boxes": boxes.tolist(), "scores": scores.tolist(), "labels": labels.tolist()}

    return results