import torch

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from .utils import bbox_iou


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

import cv2






class FastSAMPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'

    def postprocess(self, preds, img, orig_imgs):
        """TODO: filter by classes."""
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        try:
            # p1 = [tensor.cpu().numpy() for tensor in p]
            cropped_imgs = []
            print(img.shape)
            # print(len(p[0][1]))s
            for box in p[0]:
                box = box.cpu().numpy()
                x1, y1, x2, y2 = box[:4].astype(int)  # Đảm bảo thứ tự tọa độ chính xác

                # Kiểm tra giá trị tọa độ và kích thước ảnh
                print("Tọa độ bounding box:", x1, y1, x2, y2)
                print("Kích thước ảnh gốc:", img.shape)

                # Cắt ảnh từ box
                if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                    print("Xoá bounding box vì tọa độ nằm ngoài ảnh")
                    continue
                cropped = img[y1:y2, x1:x2]
                print(cropped.shape)

        except Exception as e:
            print(e)

        results = []
        if len(p) == 0 or len(p[0]) == 0:
            print("No object detected.")
            return results

        full_box = torch.zeros_like(p[0][0])
        full_box[2], full_box[3], full_box[4], full_box[6:] = img.shape[3], img.shape[2], 1.0, 1.0
        full_box = full_box.view(1, -1)
        critical_iou_index = bbox_iou(full_box[0][:4], p[0][:, :4], iou_thres=0.9, image_shape=img.shape[2:])
        if critical_iou_index.numel() != 0:
            full_box[0][4] = p[0][critical_iou_index][:,4]
            full_box[0][6:] = p[0][critical_iou_index][:,6:]
            p[0][critical_iou_index] = full_box
        
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            if not len(pred):  # save empty boxes
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
                continue
            if self.args.retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
