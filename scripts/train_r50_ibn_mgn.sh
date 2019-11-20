#SOLVER.BASE_LR 0.01
python tools/train.py --config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('2, 6')" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/datasets')" \
MODEL.PRETRAIN_PATH "('/home/zxh/.cache/torch/checkpoints/r50_ibn_a.pth')" \
OUTPUT_DIR "('new_experiment/r50_ibn_mgn')" \
MODEL.MGN True \
INPUT.SIZE_TRAIN "([384, 128])" \
MODEL.LAST_STRIDE 2 \
INPUT.RE_PROB 0.0 \
SOLVER.MARGIN 1.3
