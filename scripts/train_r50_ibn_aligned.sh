python tools/train.py --config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('0,1')" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/datasets')" \
MODEL.PRETRAIN_PATH "('/home/zxh/.cache/torch/checkpoints/r50_ibn_a.pth')" \
OUTPUT_DIR "('new_experiment/r50_ibn_align_pad10')" \
MODEL.ALIGNED True \
INPUT.SIZE_TRAIN "([256, 128])" \
INPUT.RE_PROB 0.0 \
INPUT.PADDING 10