# resnet101_ibn_a
#INPUT.PIXEL_MEAN "([0.097, 0.1831, 0.2127])" \
#INPUT.PIXEL_STD "([0.1356, 0.1482, 0.1938])"
#INPUT.PADDING 10
nohup /root/anaconda3/bin/python -u tools/train.py --config_file='configs/softmax_triplet_with_center.yml' \
MODEL.DEVICE_ID "('0,1')" \
MODEL.NAME "('self')" \
DATASETS.NAMES "('mydata')" \
DATASETS.ROOT_DIR "('/mnt/baseline/reid-baseline/data')" \
MODEL.PRETRAIN_PATH "('/mnt/baseline/reid-baseline/pretrainmodel/resnet50_ibn_a_model_80.pth')" \
OUTPUT_DIR "('/mnt/baseline/reid-baseline/output2')" \
MODEL.PCB True \
INPUT.SIZE_TRAIN "([384, 128])" \
INPUT.RE_PROB 0.3 \
MODEL.PCB_RPP False \
MODEL.USE_FOCAL_LOSS False \
MODEL.SUM False   > test.log 2>&1 &
