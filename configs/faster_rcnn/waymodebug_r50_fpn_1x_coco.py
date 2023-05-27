_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/waymo_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=3)))

data_root = 'data/waymo_kitti/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'waymo_infos_train_1k.pkl',)
)
# ori 1x lr=0.02
# now for 4 gpu
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)