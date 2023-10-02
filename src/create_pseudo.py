import numpy as np
import json

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="ASTOD thresholding", add_help=add_help)

    parser.add_argument("--train-file",
                        help="training file")
    parser.add_argument("--unlabeled-file",
                        help="unlabeled file")
    parser.add_argument("--pseudo-file",
                        help="unlabeled file")
    parser.add_argument("--output-file", default="",
                        help="path where to save")
        

    parser.add_argument("--bins", default=21, type=int,
                        help="number of bins")  
    parser.add_argument("--lower_bin", default=0.5, type=float) 
    parser.add_argument("--threshold", default=None, type=float,
                        help="score threshold")
    parser.add_argument("--glob", action="store_true",
                        help="use global threshold instead of per class")

    return parser

def main(args):

    print(args)

    labeled = json.load(open(args.train_file, "r"))
    unlabeled = json.load(open(args.unlabeled_file, "r"))
    pseudo = json.load(open(args.pseudo_file, "r"))

    new_labeled = {"images": [], "annotations": [], "categories": []}
    for k in labeled["images"]:
        new_labeled["images"].append({"file_name": k["file_name"],
                            "height": k["height"],
                            "width": k["width"],
                            "id": k["id"]
                            })
    ann_id = 0
    for k in labeled["annotations"]:
        ann_id += 1
        new_labeled["annotations"].append({"image_id": k["image_id"],
                                        "bbox": k["bbox"],
                                        "category_id": k["category_id"],
                                        "id": ann_id,
                                        "iscrowd": k["iscrowd"],
                                        "segmentation": k["segmentation"],
                                        "area": k["area"],
                                        "labeled": 1
                                    })
    new_labeled["categories"] = labeled["categories"]
    print(f'Number of labeled images: {len(new_labeled["images"])}')

    histo_dict = {}
    for i in range(1, len(new_labeled["categories"])+1):
        histo_dict[i] = {"scores": [], "threshold": None}
    histo_dict["all"] = {"scores": [], "threshold": None}
    for d in pseudo:
        for l, s in zip(pseudo[d]["labels"], pseudo[d]["scores"]):
            label = l
            score = s
            histo_dict[label]["scores"].append(score)
            histo_dict["all"]["scores"].append(score)


    for i in [*range(1, len(new_labeled["categories"])+1), "all"]:
        n, v = np.histogram(histo_dict[i]["scores"], np.linspace(args.lower_bin, 1, args.bins))
        histo_dict[i]["threshold"] = v[np.argmin(n)]
        if i != "all":
            print(f"Threshold for class {i}: {v[np.argmin(n)]}")

    tau = histo_dict["all"]["threshold"]
    print(f"Global threshold at: {tau}")

    unlabeled_dict = {}

    for i in unlabeled["images"]:
        unlabeled_dict[i["id"]] = {"path": i["file_name"],
                                "height": i["height"],
                                "width": i["width"],
                                "id": i["id"]
                            }

    print(f'Number of unlabeled images: {len(unlabeled["images"])}')

    pseudo_dict = {"images": [], "annotations": [], "categories": []}
    for i in unlabeled_dict:
        pseudo_dict["images"].append({"file_name": unlabeled_dict[i]["path"],
                            "height": unlabeled_dict[i]["height"],
                            "width": unlabeled_dict[i]["width"],
                            "id": unlabeled_dict[i]["id"]
                            })

    inverse_mapping = {}
    for i, c in enumerate(labeled["categories"]):
        inverse_mapping[i+1] = c["id"]

    for i in pseudo:
        for b, s, l in zip(pseudo[i]["boxes"], pseudo[i]["scores"], pseudo[i]["labels"]):
            x1 = b[0]
            y1 = b[1]
            w = b[2] - b[0]
            h = b[3] - b[1]
            if args.threshold:
                if s >= args.threshold:
                    ann_id += 1
                    pseudo_dict["annotations"].append({"image_id": int(i),
                                                    "bbox": [x1, y1, w, h],
                                                    "category_id": inverse_mapping[l],
                                                    "id": ann_id,
                                                    "iscrowd": 0,
                                                    "segmentation": [],
                                                    "area": w*h,
                                                    "score": s,
                                                    "labeled": 0
                                                    })
            else:
                if args.glob:
                    if s >= histo_dict["all"]["threshold"]:
                        ann_id += 1
                        pseudo_dict["annotations"].append({"image_id": int(i),
                                                        "bbox": [x1, y1, w, h],
                                                        "category_id": inverse_mapping[l],
                                                        "id": ann_id,
                                                        "iscrowd": 0,
                                                        "segmentation": [],
                                                        "area": w*h,
                                                        "score": s,
                                                        "labeled": 0
                                                    })
                else:
                    if s >= histo_dict[l]["threshold"]:
                        ann_id += 1
                        pseudo_dict["annotations"].append({"image_id": int(i),
                                                        "bbox": [x1, y1, w, h],
                                                        "category_id": inverse_mapping[l],
                                                        "id": ann_id,
                                                        "iscrowd": 0,
                                                        "segmentation": [],
                                                        "area": w*h,
                                                        "score": s,
                                                        "labeled": 0
                                                    })

    pseudo_dict["categories"] = labeled["categories"]
    print(f'Number of pseudo images: {len(pseudo_dict["images"])}')

    for i, c in enumerate(pseudo_dict["categories"]):
        if args.threshold:
            c["threshold"] = args.threshold
        else:
            if args.glob:
                c["threshold"] = histo_dict["all"]["threshold"]
            else:
                c["threshold"] = histo_dict[i+1]["threshold"]
        
    # pseudo_dict["images"].extend(labeled["images"])
    # pseudo_dict["annotations"].extend(labeled["annotations"])

    json.dump(pseudo_dict, open(args.output_file, "w"))

    print(f"file saved: {args.output_file}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
