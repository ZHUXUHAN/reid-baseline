# Model resnet50_ibn_a se_resnet101_ibn_a
#INPUT.PIXEL_MEAN "([0.097, 0.1831, 0.2127])" \
#INPUT.PIXEL_STD "([0.1356, 0.1482, 0.1938])"
# 注意每次生成新的json文件时 要把data 数据流路径改了
python tools/test.py \
--config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/datasets')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/home/zxh/ReID/reid-strong-baseline/new_experiment/r50_ibn_pcb_sampler/resnet50_ibn_a_model_120.pth')" \
TEST.RE_RANKING  "('yes')" \
MODEL.ALIGNED False \
MODEL.PCB True \
INPUT.SIZE_TEST "([384, 128])" \
TEST.ADJUST_RERANK True \
MODEL.CAM False
