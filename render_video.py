import os
import cv2
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-i', dest='img_dir', required=True,
                    help="Path to directory that contains images for rendering")
parser.add_argument('-o', dest='out_dir', default='results/output.avi',
                    help="File path to save output video ('results/output.avi').")

args = parser.parse_args()

image_list = [x for x in os.listdir(args.img_dir) if x.split('.')[-1] in ['jpg', 'png']]

# Init video writer
temp = cv2.imread(os.path.join(args.img_dir, image_list[0]))
height, width, _ = temp.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.out_dir, fourcc, 5.0, (width, height))

for filename in tqdm(sorted(image_list)):
    path = os.path.join(args.img_dir, filename)

    img = cv2.imread(path)
    
    out.write(img)

print("Video written to %s" % args.out_dir)
out.release()
