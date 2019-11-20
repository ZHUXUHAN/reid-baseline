python tools/train.py --config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('0,5,6,7')" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/datasets')" \
MODEL.PRETRAIN_PATH "('/home/zxh/.cache/torch/checkpoints/resnext101_ibn_a.pth.tar')" \
OUTPUT_DIR "('new_experiment/ser101_ibn_pcb')" \
MODEL.PCB True \
INPUT.SIZE_TRAIN "([384, 128])" \
INPUT.RE_PROB 0.0 \
MODEL.PCB_RPP False \
MODEL.USE_FOCAL_LOSS False \
SOLVER.IMS_PER_BATCH 32
