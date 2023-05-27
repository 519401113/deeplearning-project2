import copy
import os.path as osp
import os
import mmcv
import numpy as np
import torch

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
vertices = np.array([[0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,-0.5,0.5],
                     [-0.5,0.5,-0.5],[-0.5,-0.5,-0.5],[0.5,0.5,0.5]])

def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def rotation_3d_in_axis(points,
                        angles,
                        axis=0,
                        return_mat=False,
                        clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple | float):
            Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 \
        and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(f'axis should in range '
                             f'[-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new

def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 1.0, 0.5),
                           axis=1):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): Origin point relate to
            smallest point. Use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0)
            in lidar. Defaults to (0.5, 1.0, 0.5).
        axis (int, optional): Rotation axis. 1 for camera and 2 for lidar.
            Defaults to 1.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(lwh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = torch.tensor(corners)
        angles = torch.tensor(angles)
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
        corners = corners.numpy()

    corners += centers.reshape([-1, 1, 3])
    return corners


def recal_bbox(info):
    P2 = info['calib']['P0']

    repro_recs = []
    # if no annotations in info (test dataset), then return
    if 'annos' not in info:
        return repro_recs

    # Get all the annotation with the specified visibilties.
    ann_dicts = info['annos']
    mask = [(ocld in [0,1,2,3]) for ocld in ann_dicts['occluded']]
    for k in ann_dicts.keys():
        ann_dicts[k] = ann_dicts[k][mask]

    # convert dict of list to list of dict
    ann_recs = []
    for i in range(len(ann_dicts['occluded'])):
        ann_rec = {}
        for k in ann_dicts.keys():
            ann_rec[k] = ann_dicts[k][i]
        ann_recs.append(ann_rec)
    # import pdb; pdb.set_trace()
    bbox = []
    # from PIL import Image
    # aa = Image.open(os.path.join('data/waymo_kitti', info['image']['image_path']))
    # aa.save('debug.png')
    # import cv2
    # image = cv2.imread(os.path.join('data/waymo_kitti', info['image']['image_path']))
    #
    # for i in info['annos']['bbox']:
    #     first_point = (int(i[0]), int(i[1]))
    #     end_point = (int(i[2]), int(i[3]))
    #     cv2.rectangle(image, first_point, end_point, (0, 255, 0), 2)
    #
    #
    # cv2.imwrite('debug.png', image)
    for ann_idx, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        # ann_rec['sample_annotation_token'] = \
        #     f"{info['image']['image_idx']}.{ann_idx}"
        # ann_rec['sample_data_token'] = info['image']['image_idx']
        # sample_data_token = info['image']['image_idx']

        loc = ann_rec['location'][np.newaxis, :]
        dim = ann_rec['dimensions'][np.newaxis, :]
        rot = ann_rec['rotation_y'][np.newaxis, np.newaxis]
        # transform the center from [0.5, 1.0, 0.5] to [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        loc = loc + dim * (dst - src)
        gt_bbox_3d = np.concatenate([loc, dim, rot], axis=1).astype(np.float32)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = center_to_corner_box3d(
            gt_bbox_3d[:, :3],
            gt_bbox_3d[:, 3:6],
            gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
            axis=1)
        corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
        # in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        # corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = P2[:3,:3]@corners_3d/np.abs(corners_3d[2,:])


        w_min, h_min, w_max, h_max = corner_coords[0].min(),corner_coords[1].min(),\
                                     corner_coords[0].max(),corner_coords[1].max()
        w_min = max(w_min, 0)
        h_min = max(h_min, 0)
        w_max = min(w_max, 1920)
        h_max = min(h_max, 1280)
        import pdb; pdb.set_trace()
        if w_min>=w_max or h_min>=h_max:
            continue
        else:
            bbox.append([w_min, h_min, w_max, h_max])



    return np.array(bbox)



@DATASETS.register_module()
class KittiDataset(CustomDataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        # image_list = mmcv.list_from_file(self.ann_file)
        annos = mmcv.load(self.ann_file)
        root = os.getcwd()
        def s2v(bbox):
            return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        data_infos = []
        annotations = annos['annotations']
        images = annos['images']
        filename0 = annotations[0]['file_name']
        height, width = annos['images'][0]['height'], annos['images'][0]['width']
        data_info = dict(filename=os.path.join(root,self.img_prefix,filename0), width=width, height=height)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_bboxes.append(s2v(annotations[0]['bbox']))
        gt_labels.append(cat2label[annotations[0]['category_name']])
        for anno in annotations[1:]:
            if anno['file_name'] != filename0:
                data_anno = dict(
                    bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(gt_labels, dtype=np.long),
                    bboxes_ignore=np.array(gt_bboxes_ignore,
                                           dtype=np.float32).reshape(-1, 4),
                    labels_ignore=np.array(gt_labels_ignore, dtype=np.long))
                data_info.update(ann=data_anno)
                data_infos.append(data_info)
                filename0 = anno['file_name']
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                # height, width = image['height'], image['width']
                data_info = dict(filename=os.path.join(root, self.img_prefix, filename0), width=width, height=height)
                gt_bboxes.append(s2v(anno['bbox']))
                gt_labels.append(cat2label[anno['category_name']])
            else:
                gt_bboxes.append(s2v(anno['bbox']))
                gt_labels.append(cat2label[anno['category_name']])
        data_anno = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(gt_labels, dtype=np.long),
            bboxes_ignore=np.array(gt_bboxes_ignore,
                                   dtype=np.float32).reshape(-1, 4),
            labels_ignore=np.array(gt_labels_ignore, dtype=np.long))
        data_info.update(ann=data_anno)
        data_infos.append(data_info)
        return data_infos
        # import pdb; pdb.set_trace()

        # data_infos_pkl = []
        # annos_pkl = mmcv.load(self.ann_file.replace('_mono3d.coco.json','.pkl'))
        # for info in annos_pkl:
        #     filename = os.path.join(root,self.img_prefix,info['image']['image_path'])
        #     height, width = info['image']['image_shape'][:2]
        #     data_info = dict(filename=filename, width=width, height=height)
        #     anno = info['annos']
        #     if len(anno['bbox']) == 0:
        #         continue
        #
        #
        #     data_anno = dict(
        #         bboxes=anno['bbox'],
        #         labels=np.array([cat2label[c] for c in anno['name']], dtype=np.long),
        #         bboxes_ignore=np.array([],
        #                                dtype=np.float32).reshape(-1, 4),
        #         labels_ignore=np.array([], dtype=np.long))
        #     data_info.update(ann=data_anno)
        #     data_infos_pkl.append(data_info)
        # import pdb; pdb.set_trace()
        #
        # data_infos_pkl = []
        # annos_pkl = mmcv.load(self.ann_file.replace('_mono3d.coco.json','.pkl'))
        # for info in annos_pkl:
        #     filename = os.path.join(root,self.img_prefix,info['image']['image_path'])
        #     height, width = info['image']['image_shape'][:2]
        #     data_info = dict(filename=filename, width=width, height=height)
        #     anno = info['annos']
        #     if len(anno['bbox']) == 0:
        #         continue
        #     bboxes = recal_bbox(info)
        #     data_anno = dict(
        #         bboxes=bboxes,
        #         labels=np.array([cat2label[c] for c in anno['name']], dtype=np.long),
        #         bboxes_ignore=np.array([],
        #                                dtype=np.float32).reshape(-1, 4),
        #         labels_ignore=np.array([], dtype=np.long))
        #     data_info.update(ann=data_anno)
        #     data_infos_pkl.append(data_info)
        #
        #
        # return data_infos



        # convert annotations to middle format
        # for image_id in image_list:
        #     filename = f'{self.img_prefix}/{image_id}.png'
        #     image = mmcv.imread(filename)
        #     height, width = image.shape[:2]
        #
        #     data_info = dict(filename=f'{image_id}.png', width=width, height=height)
        #
        #     # load annotations
        #     label_prefix = self.img_prefix.replace('image_2', 'label_2')
        #     lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))
        #
        #     content = [line.strip().split(' ') for line in lines]
        #     bbox_names = [x[0] for x in content]
        #     bboxes = [[float(info) for info in x[4:8]] for x in content]
        #
        #     gt_bboxes = []
        #     gt_labels = []
        #     gt_bboxes_ignore = []
        #     gt_labels_ignore = []
        #
        #     # filter 'DontCare'
        #     for bbox_name, bbox in zip(bbox_names, bboxes):
        #         if bbox_name in cat2label:
        #             gt_labels.append(cat2label[bbox_name])
        #             gt_bboxes.append(bbox)
        #         else:
        #             gt_labels_ignore.append(-1)
        #             gt_bboxes_ignore.append(bbox)
        #
            # data_anno = dict(
            #     bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            #     labels=np.array(gt_labels, dtype=np.long),
            #     bboxes_ignore=np.array(gt_bboxes_ignore,
            #                            dtype=np.float32).reshape(-1, 4),
            #     labels_ignore=np.array(gt_labels_ignore, dtype=np.long))
        #
        #     data_info.update(ann=data_anno)
        #     data_infos.append(data_info)
        #
        # return data_infos