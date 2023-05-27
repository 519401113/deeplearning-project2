# deeplearning-project2

## prepare data
```
mkdir data
```
download VOC2012 dataset and unzip it to './data' ,  then you will have './data/VOCdevkit/VOC2012'

## train faster rcnn
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/faster_rcnn/voc_faster_rcnn_r50_fpn_1x.py 4
```

## train fcos
```
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_voc.py 1
```

## visualize images

```
mkdir checkpoints
```
download the trained checkpoints of faster rcnn and fcos into './checkpoints'
```
cd demo
python vis.py
```
