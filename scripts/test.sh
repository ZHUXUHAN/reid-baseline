# Model resnet50_ibn_a se_resnet101_ibn_a densenet161
#INPUT.PIXEL_MEAN "([0.097, 0.1831, 0.2127])" \
#INPUT.PIXEL_STD "([0.1356, 0.1482, 0.1938])"
# 注意每次生成新的json文件时 要把data 数据流路径改了
# 测试流程：1、先调rerank ADJUST_RERANK:True MERGE:False 2、再保存npy，把数据集改掉  ADJUST_RERANK:False MERGE:False 3、cat生成json  ADJUST_RERANK:False MERGE:True
# 注意过程中时刻要调整路径
python tools/test.py \
--config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('6,7')" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/zxh/datasets')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/home/zxh/reid-strong-baseline/new_experiment/r50_ibn_pcb_sampler/resnet50_ibn_a_model_120.pth')" \
TEST.RE_RANKING  "('yes')" \
MODEL.ALIGNED False \
MODEL.PCB True \
INPUT.SIZE_TEST "([384, 128])" \
TEST.ADJUST_RERANK False \
MODEL.CAM False \
TEST.MERGE False

