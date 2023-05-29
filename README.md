# deeplearning-project2 for Q1

## prepare data
You can directly run the file **train.py** , the CIFAR100 data will be automaicly downloaded in './Q1/cifar-100-python'

## train resnet18 with different data augmentation
run for baseline
```
CUDA_VISIBLE_DEVICES=0 python train.py --name test --batch_size 512 --lr 0.001 --num_epochs 200
```
run for cutout method
```
CUDA_VISIBLE_DEVICES=0 python train.py --name test --batch_size 512 --lr 0.001 --num_epochs 200 --cutout --cutout_prob 0.5
```
run for mixup method
```
CUDA_VISIBLE_DEVICES=0 python train.py --name test --batch_size 512 --lr 0.001 --num_epochs 200 --mixup --mixup_prob 0.5
```
run for cutmix method
```
CUDA_VISIBLE_DEVICES=0 python train.py --name test --batch_size 512 --lr 0.001 --num_epochs 200 --cutmix --cutmix_prob 0.5
```
## visualize images
run the function **visualize()** in file **utils.py**
1 image of 4x4 images（original images，cutout，mixup，cutmix）for visualization is in './Q1'

链接：https://pan.baidu.com/s/1GqHdfA2el4nyEtHFM8GuXQ 
提取码：kanv

# deeplearning-project2 for Q2

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
download the trained checkpoints of faster rcnn and fcos as './checkpoints'

The trained models and training logs are uploaded in the Baidu Netdisk.

链接：https://pan.baidu.com/s/1-KWjwT1bfJPayeXXISw0Vg 
提取码：15wc
```
cd demo
python vis.py
```
4 images for visualization are in './demo/vis_image'

visualization results for faster rcnn and fcos will be './demo/xxx_result'
