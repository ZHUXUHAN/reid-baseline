python tools/train.py --config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('4,5')" \
MODEL.NAME "('densenet161')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/datasets')" \
MODEL.PRETRAIN_PATH "('/home/zxh/.cache/torch/checkpoints/densenet161-8d451a50.pth')" \
OUTPUT_DIR "('new_experiment/dense161_pcb')" \
MODEL.PCB False \
INPUT.SIZE_TRAIN "([384, 128])" \
INPUT.RE_PROB 0.3 \
MODEL.PCB_RPP False \
MODEL.USE_FOCAL_LOSS False \
SOLVER.IMS_PER_BATCH 64 \
INPUT.PADDING 0
