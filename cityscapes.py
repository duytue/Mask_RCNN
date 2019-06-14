import os
import cv2
import sys
import time
import argparse
import numpy as np
import imgaug

from tqdm import tqdm

import matplotlib
matplotlib.use('agg')

from cityscapes_dataset import CityPersonDataset, evaluate_coco

ROOT_DIR = os.path.abspath('.')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

# Path to log directory
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# class_names = ["person", "rider"]
class_names = ["__background__", "person"]
class_names = np.asarray(['__background__', 'person', 'bicycle',
                             'car', 'motorcycle', 'airplane', 'bus',
                             'train', 'truck', 'boat', 'traffic light',
                             'fire hydrant', 'stop sign', 'parking meter',
                             'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                             'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                             'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                             'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                             'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                             'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                             'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                             'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                             'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                             'refrigerator', 'book', 'clock', 'vase', 'scissors',
                             'teddy bear', 'hair drier', 'toothbrush'])

############################################################
#  Configurations
############################################################

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    NAME = "coco"

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class CocoInferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

class CityPersonConfig(Config):
    """Configuration for training on CityPerson sub-dataset of Cityscapes.
    """
    # Give the configuration a recognizable name
    NAME = "city_person"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # Person

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 100

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    MAX_GT_INSTANCES = 50

    RPN_ANCHOR_SCALES = (32, 64, 128, 256)

class InferenceConfig(CityPersonConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN on CityPerson Coco version.")
    
    parser.add_argument("--mode", default="train",
                        choices=["train", "evaluate", "inference"],
                        help="Select running mode for Mask RCNN.")
    parser.add_argument("--dataset", required=True,
                        help="Path to CityPerson dataset directory.")
    parser.add_argument("--model", required=True,
                        default="mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image_dir', default=None,
                        help="Path to image directory for inference")
    parser.add_argument('--result_dir', default="results/",
                        help="Path to result directory for overlaid images")
    parser.add_argument('--epochs', type=int,
                        default=40,
                        help='Number of epochs to train')

    return parser.parse_args()

def inference(model, config, image_dir, output_dir):

    if not os.path.isdir(image_dir):
        print("Invalid input directory!")

    output_dir = os.path.join(output_dir, config.name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    file_list = os.listdir(image_dir)

    total_time = time.time()
    for item in tqdm(sorted(file_list)):
        if not os.path.splitext(item)[:-1] in ['.jpg', '.png']:
            continue
        
        filepath = os.path.join(image_dir, item)
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        r = model.detect([img])
        end_time = time.time()

        result = visualize.draw_instances(img, r['rois'], r['masks'], 
                                                r['class_ids'], class_names,
                                                draw_box=True, draw_mask=False)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, item), result)

    print("Inference on %d images took %.1fs" % (len(file_list), time.time() - total_time))





def main():
    args = parse_arguments()

    if args.mode == "train":
        config = CityPersonConfig()
    else:
        config = InferenceConfig()

    config.display()

    # Create model
    if args.mode == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=args.logs)

    # Load weights
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        model_path = model.find_last()
    else:
        model_path = args.model

    model.load_weights(model_path, by_name=True, exclude=[
            'mrcnn_class_logits', 'mrcnn_bbox_fc',
            'mrcnn_bbox', 'mrcnn_mask'
        ])
    print("Loaded weights from ", model_path)

    if args.mode == "train":
        # Training set
        dataset_train = CityPersonDataset()
        dataset_train.load_coco(args.dataset, "train")
        dataset_train.prepare()

        # Validation set
        dataset_val = CityPersonDataset()
        dataset_val.load_coco(args.dataset, "val")
        dataset_val.prepare()


        # Image augmentation
        augmentation = imgaug.augmenters.SomeOf(3, [
                        imgaug.augmenters.Fliplr(0.5),
                        imgaug.augmenters.Affine(
                            # scale=(0.6, 1.4),  # scale images to 60-140% of their size
                            scale=(0.4, 2),
                            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        )])

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=args.epochs,
                    layers="heads",
                    augmentation=augmentation)

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=120,
        #             layers='4+',
        #             augmentation=augmentation)

        # # Training - Stage 3
        # # Fine tune all layers
        # print("Fine tune all layers")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=160,
        #             layers='all',
        #             augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CityPersonDataset()
        coco = dataset_val.load_coco(args.dataset, "val", return_coco=True)
        dataset_val.prepare()
        # print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox")
    elif args.command == "inference":
        assert args.image_dir != None
        inference(model, config, args.image_dir, args.result_dir)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate' or 'inference'".format(args.command))


if __name__ == "__main__":
    main()