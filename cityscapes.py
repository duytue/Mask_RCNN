import os
import cv2
import sys
import time
import datetime
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

    STEPS_PER_EPOCH = 2000
    VALIDATION_STEPS = 500

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    MAX_GT_INSTANCES = 50

    RPN_ANCHOR_SCALES = (32, 64, 128, 256)

class CityPersonConfig1(Config):
    NAME = "city_person_res50"
    BACKBONE = "resnet50"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1 # Person

    STEPS_PER_EPOCH = 2000
    VALIDATION_STEPS = 500

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    MAX_GT_INSTANCES = 50

    RPN_ANCHOR_SCALES = (32, 64, 128, 256)

class CityPersonConfig2(Config):
    NAME = "city_person2"
    BACKBONE = "resnet50"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1 # Person

    STEPS_PER_EPOCH = 2000
    VALIDATION_STEPS = 500

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    MAX_GT_INSTANCES = 50

    RPN_ANCHOR_SCALES = (32, 64, 128, 256)

    TRAIN_ROIS_PER_IMAGE = 256
    ROI_POSITIVE_RATIO = 0.33

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.2,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.2,
        "mrcnn_mask_loss": 0.8
    }

class CityPersonConfig3(Config):
    NAME = "city_person3"
    BACKBONE = "resnet50"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1 # Person

    STEPS_PER_EPOCH = 2000
    VALIDATION_STEPS = 500

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    MAX_GT_INSTANCES = 50

    RPN_ANCHOR_SCALES = (32, 64, 128, 256)

    TRAIN_ROIS_PER_IMAGE = 256
    ROI_POSITIVE_RATIO = 0.33

    # LOSS_WEIGHTS = {
    #     "rpn_class_loss": 1.,
    #     "rpn_bbox_loss": 1.2,
    #     "mrcnn_class_loss": 1.,
    #     "mrcnn_bbox_loss": 1.2,
    #     "mrcnn_mask_loss": 0.8
    # }

    IMAGE_RESIZE_MODE = "none"


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
    parser.add_argument("--resume", action='store_true',
                        help="Resume or training from scratch")
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
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('--gpu_id', type=str,
                        default="0",
                        help="GPU ID to train model.")
    parser.add_argument('--version', required=True, type=str,
                        help="Select config version to run.")
    parser.add_argument('--output_type', default=None,
                        help="Whether to output images or not. 'media' for image output.")

    return parser.parse_args()

def inference(model, config, image_dir, output_dir, output_type=None):
    def set_up_vcap(video_path, result_path):
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        length = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
        filename = os.path.join(result_path, 'video_{:%Y%m%dT%H%M%S}.avi'.format(datetime.datetime.now()))
        vwriter = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'),
                                fps, (width, height))
        return vcapture, vwriter, filename, length

    output_dir = os.path.join(output_dir, config.NAME)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Input is video
    if image_dir.split('.')[-1] in ['mp4', 'mov', 'MOV', 'MP4', 'avi']:
        vcapture, vwriter, filename, cap_frames = set_up_vcap(image_dir, output_dir)
        start = 0
        end = cap_frames
        count = 0
        success = True
        prev_r = None

        start_time = time.time()
        while success:
            print('frame: ', count, end='\r')
            success, image = vcapture.read()
            if success and start <= count < end:
                image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                r = model.detect([image_yuv], verbose=0)[0]

                detection = visualize.draw_instances(image_yuv, r['rois'], r['masks'], 
                                                    r['class_ids'], r['scores'], class_names,
                                                    draw_box=True, draw_mask=False)
                detection = detection[..., ::-1]

                vis = cv2.resize(detection, None, fx=0.5, fy=0.5)
                cv2.imshow('detection', vis)
                vwriter.write(detection)

                # Press ESC on keyboard to  exit
                if cv2.waitKey(27) == ord('q'):
                    break
            elif count >= end:
                success = False
            count += 1

        vwriter.release()
        end_time = time.time()
        print('Saved to %s. Total time: %.2f' % (filename, end_time - start_time))

    # Input is directory of images
    else:
        if not os.path.isdir(image_dir):
            print("Invalid input directory!")
        
        file_list = os.listdir(image_dir)

        pure_dectetion_times = []

        total_time = time.time()
        for item in tqdm(sorted(file_list)):
            if not os.path.splitext(item)[-1] in ['.jpg', '.png']:
                continue
            
            filepath = os.path.join(image_dir, item)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            r = model.detect([img])[0]
            end_time = time.time()
            pure_dectetion_times.append(end_time - start_time)

            if output_type == "media":
                result = visualize.draw_instances(img, r['rois'], r['masks'], 
                                                        r['class_ids'], r['scores'], class_names,
                                                        draw_box=True, draw_mask=False)

                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, item), result)

        avg_pure = sum(pure_dectetion_times) / len(pure_dectetion_times)
        print("Inference on %d images took %.1fs (avg pure detection time: %.3fs)" % (len(file_list), time.time() - total_time, avg_pure))


def getConfigVersion(args):
    version = args.version

    if version == "coco":
        return CocoConfig()
    elif version == "0":
        return CityPersonConfig()
    elif version == "1":
        return CityPersonConfig1()
    elif version == "1.1":
        return CityPersonConfig1_1()
    elif version == "2":
        return CityPersonConfig2()
    elif version == "3":
        return CityPersonConfig3()


def main():
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # if args.mode == "train":
    #     config = CityPersonConfig2()
    # else:
    #     if args.model == "coco":
    #         config = CocoInferenceConfig()
    #     else:
    #         config = InferenceConfig()

    config = getConfigVersion(args)

    config.LEARNING_RATE = args.lr


    # Create model
    if args.mode == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)
    else:
        class InferenceConfig(config.__class__):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.7
            DETECTION_MAX_INSTANCES = 200

            NO_MASK = True
        
        config = InferenceConfig()
        model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=args.logs)
    
    config.display()


    # Load weights
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        model_path = model.find_last()
    else:
        model_path = args.model

    if args.mode in ["evaluate", "inference"] or args.resume:
        model.load_weights(model_path, by_name=True)
        print("Loaded weights from ", model_path)
    else:
        model.load_weights(model_path, by_name=True, exclude=[
                'mrcnn_class_logits', 'mrcnn_bbox_fc',
                'mrcnn_bbox', 'mrcnn_mask'
            ])
        print("Loaded weights without heads from ", model_path)

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
        print("Training network heads using Config %s" % config.NAME)
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

    elif args.mode == "evaluate":
        # Validation dataset
        dataset_val = CityPersonDataset()
        coco = dataset_val.load_coco(args.dataset, "val", return_coco=True)
        dataset_val.prepare()
        # print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=0)
    elif args.mode == "inference":
        assert args.image_dir != None
        inference(model, config, args.image_dir, args.result_dir, args.output_type)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate' or 'inference'".format(args.mode))


if __name__ == "__main__":
    main()