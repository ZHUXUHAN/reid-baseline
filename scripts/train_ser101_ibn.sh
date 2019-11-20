python ../tools/train.py --config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('2,3')" \
MODEL.NAME "('se_resnet101_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/ReID/deep-person-reid/data')" \
MODEL.PRETRAIN_PATH "('/home/zxh/.cache/torch/checkpoints/se_resnet101_ibn_a.pth.tar')" \
OUTPUT_DIR "('naicreid/haveing_erase')" \
MODEL.PCB True
