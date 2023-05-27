from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import numpy as np
import cv2

image_list = sorted(os.listdir('vis_image'))
image_paths = [os.path.join('vis_image', a) for a in image_list]

config_file = '../configs/faster_rcnn/voc_faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '../work_dirs/voc_faster_rcnn_r50_fpn_1x/latest.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
out_dir_fasterrcnn = 'faster_rcnn_result'
os.makedirs(out_dir_fasterrcnn, exist_ok=True)


for i in image_paths:
    result = inference_detector(model, i)
    show_result_pyplot(model, i, result, out_file=i.replace('vis_image', out_dir_fasterrcnn))
    bboxes = np.load('proposal_bboxes.npy')
    h0,w0 = bboxes[:,3].max(), bboxes[:,2].max()
    img1 = cv2.imread(i)
    h,w,_ = img1.shape
    for j in range(bboxes.shape[0]):
        bbox = bboxes[j][:4]
        bbox[0] = int(bbox[0]/w0*w)
        bbox[2] = int(bbox[2]/w0*w)
        bbox[1] = int(bbox[1]/h0*h)
        bbox[3] = int(bbox[3]/h0*h)
        img2 = cv2.rectangle(img1, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(0,255,0),2)
    cv2.imwrite(i.replace('vis_image', out_dir_fasterrcnn).replace('.','_stage1_bboxes.'), img2)



config_file = '../configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_voc.py'
checkpoint_file = '../work_dirs/fcos_r50_caffe_fpn_gn-head_1x_voc/latest.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
out_dir = 'fcos_result'
os.makedirs(out_dir, exist_ok=True)


for i in image_paths:
    result = inference_detector(model, i)
    show_result_pyplot(model, i, result, out_file=i.replace('vis_image', out_dir))



