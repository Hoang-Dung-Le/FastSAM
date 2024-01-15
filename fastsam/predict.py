import torch

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from .utils import bbox_iou

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

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        bboxes = p[0]

# Lấy hình ảnh gốc
        orig_img = orig_imgs

        fig, ax = plt.subplots(figsize=(10, 10))

# Hiển thị hình ảnh gốc
        ax.imshow(orig_img)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            conf = bbox[4]

            # Tạo một hình chữ nhật với màu đỏ và độ mờ 0.5
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none', alpha=0.5)
            ax.add_patch(rect)

            # Hiển thị độ tin cậy phía trên bounding box
            ax.text(x1, y1 - 5, f"{conf:.2f}", bbox=dict(facecolor='white', alpha=0.5))
        
        print("len p: ", len(p))
        print(type(p))
        # print(p.shape)
        print(len(p[0]))

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
