import os
import sys
import time
import numpy as np

from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from mrcnn import utils, visualize
############################################################
#  Dataset
############################################################

class MOTDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):

        coco = COCO("{}/annotations/mot17_{}.json".format(dataset_dir, subset))

        image_dir = "{}/".format(dataset_dir)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])
            print("loading dataset with [%s]" % coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_bbox(self, image_id):
        """Load instance bboxes for the given image.
        COCO bbox format: [N, [bbox_left ,bbox_top , bbox_wid, bbox_height]]
        
        Return:
            list of bboxes: [N, [y1, x1, y2, x2]]
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(MOTDataset, self).load_bbox(image_id)

        instance_bboxes = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                x1, y1, w, h = annotation['bbox']
                x2 = x1 + w
                y2 = y1 + h
                instance_bboxes.append([y1, x1, y2, x2])
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            bboxes = np.array(instance_bboxes, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            return bboxes, class_ids
        else:
            # Call super class to return an empty mask
            return super(MOTDataset, self).load_bbox(image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(MOTDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(MOTDataset, self).load_mask(image_id)

    def filter_mask(self, segm):
        """
        For Cityscapes dataset only
        Remove segmentation items with less than two points (len <= 4).
        This causes exception error while parsing object in annToRLE() below
        """
        ret = []
        for poly in segm:
            if len(poly) > 4:
                ret.append(poly)

        return ret

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            segm = self.filter_mask(segm)
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m



############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]

            # Skip all class but person
            if class_id != 1:
                continue
                
            score = scores[i]
            bbox = np.around(rois[i], 1)
            if masks is not None:
                masks = masks.astype(np.uint8)
                mask = masks[:, :, i]

                result = {
                    "image_id": image_id,
                    "category_id": dataset.get_source_class_id(class_id, "coco"),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    "segmentation": maskUtils.encode(np.asfortranarray(mask))
                }
            else:
                result = {
                    "image_id": image_id,
                    "category_id": dataset.get_source_class_id(class_id, "coco"),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score
                }
            results.append(result)
    return results
class_names = ["__background__", "Pedestrian"]

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None, debug=True):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    import cv2
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    # For MOT17Det challenge, we'll use only the MOT17-04 videos with 1050 frames for evaluation
    # https://motchallenge.net/vis/MOT17-04
    # Image ids for this video: 4266-5315
    if limit:
        assert limit == 1050
        image_ids = image_ids[4266:4266+limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    if limit != 0:
        limit = len(image_ids)
    else:
        limit = len(image_ids)
    print("Evaluating on %d images" % limit)

    if debug:
        import datetime
        width = 1920
        height = 1080
        fps = 30
        filename = os.path.join('results', 'video_{:%Y%m%dT%H%M%S}.avi'.format(datetime.datetime.now()))
        vwriter = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'),
                                fps, (width, height))

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in tqdm(enumerate(image_ids)):
        if i == limit:
            break
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool

        if debug:
            result = visualize.draw_instances(image, r['rois'], r['masks'], 
                                                            r['class_ids'], r['scores'], class_names,
                                                            draw_box=True, draw_mask=False)

            # Ground truth
            bbox, class_ids = dataset.load_bbox(image_id)
            result = visualize.draw_instances(result, bbox, None, 
                                                            class_ids, None, class_names,
                                                            draw_box=True, draw_mask=False)

            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            vwriter.write(result)
            vis = cv2.resize(result, None, fx=0.5, fy=0.5)
            cv2.imshow('image', vis)
            if cv2.waitKey(30) == ord('q'):
                break
        

        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"])
        results.extend(image_results)

    if debug:
        vwriter.release()

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)
    cv2.destroyAllWindows()

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)