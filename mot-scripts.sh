# logs/coco_person20190618T1624/mask_rcnn_coco_person_0057.h5
# models/mask_rcnn_city_person_0054.h5
python3 mot.py --mode evaluate \
                        --dataset data/mot17 \
                        --gpu_id 1 \
                        --model models/mask_rcnn_city_person_0054.h5 \
                        --version 0


python3 mot.py --mode inference \
                        --dataset data/mot17 \
                        --output_type media \
                        --version cocoperson \
                        --gpu_id 1 \
                        --image_dir /media/kitemetric/data/datasets/MOTChallenge/MOT17Det/test/MOT17-03/img1/ \
                        --result_dir results/MOT17-03/ \
                        --model logs/coco_person20190618T1624/mask_rcnn_coco_person_0055.h5