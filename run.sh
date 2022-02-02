#!/usr/bin/env bash
# reppoints on sysu
python3 ./mmdetection/tools/train.py ./configs/reppoints_moment_r50_fpn_1x.py --eval  --work_dir './work_dirs/dmr_reppoints_sysu'
# reppoints on prw
python3 ./mmdetection/tools/train.py ./configs/reppoints_moment_r50_fpn_1x_prw.py --eval  --work_dir './work_dirs/dmr_reppoints_prw'
# retinanet on sysu
python3 ./mmdetection/tools/train.py ./mmdetection/configs/retinanet_r50_fpn_1x.py --eval  --work_dir './work_dirs/dmr_retina_sysu'
# retinanet on prw
python3 ./mmdetection/tools/train.py ./mmdetection/configs/retinanet_r50_fpn_1x_prw.py --eval  --work_dir './work_dirs/dmr_retina_prw'
