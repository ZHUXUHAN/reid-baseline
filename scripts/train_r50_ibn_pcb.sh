# resnet101_ibn_a
#INPUT.PIXEL_MEAN "([0.097, 0.1831, 0.2127])" \
#INPUT.PIXEL_STD "([0.1356, 0.1482, 0.1938])"
#INPUT.PADDING 10
#SOLVER.CHECKPOINT_PERIOD 20
python tools/train.py --config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('0,1')" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/datasets')" \
MODEL.PRETRAIN_PATH "('/home/zxh/.cache/torch/checkpoints/r50_ibn_a.pth')" \
OUTPUT_DIR "('new_experiment/r50_ibn_pcb_sampler')" \
MODEL.PCB True \
INPUT.SIZE_TRAIN "([384, 128])" \
INPUT.RE_PROB 0.3 \
MODEL.PCB_RPP False \
MODEL.USE_FOCAL_LOSS False \
MODEL.SUM False \
SOLVER.MAX_EPOCHS 180 \
SOLVER.CHECKPOINT_PERIOD 20


