python tools/train.py --config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('1,2')" \
MODEL.NAME "('se_resnet101_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/datasets')" \
MODEL.PRETRAIN_PATH "('/home/zxh/.cache/torch/checkpoints/se_resnet101_ibn_a.pth.tar')" \
OUTPUT_DIR "('naicreid/ser101_ibn_aligned')" \
MODEL.ALIGNED True \
MODEL.IF_WITH_CENTER "('yes')" \
MODEL.METRIC_LOSS_TYPE "('triplet_center')" \
INPUT.SIZE_TRAIN "([256, 128])"
