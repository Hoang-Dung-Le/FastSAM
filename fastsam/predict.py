import torch

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from .utils import bbox_iou

from .resnet import CustomResNet34Classifier

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from PIL import Image
from google.colab.patches import cv2_imshow





class FastSAMPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'
        # print(type(self.model))
        
    def predict(self, image):
        self.model = self._load_model('/content/drive/MyDrive/CV/fastsam/classifier_checkpoint/model_resnet34.pth', 2)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Thêm chiều batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.model)
        self.model.to(device)
        input_batch = input_batch.to(device)
        # Dự đoán
        
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_batch)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

    def _load_model(self, model_path, num_classes):
        # Khởi tạo mô hình ResNet34
        model = models.resnet34()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load trạng thái đã được lưu của mô hình
        model.load_state_dict(torch.load(model_path))
        return model
    def postprocess(self, preds, img, orig_imgs):
        """TODO: filter by classes."""
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        
        # try:
        #     img_test = img[0]  # Assuming img is defined elsewhere in your code
        #     img_test = img_test.cpu().numpy()

            

        #     img_test = np.transpose(img_test, (2, 1, 0))
        #     cv2.imwrite("/content/anhgoc.png", img_test*255)
        #     new_p = []  # Create an empty list to store the filtered bounding boxes
        #     for box in p[0]:
        #         box = box.cpu().numpy()
        #         x1, y1, x2, y2 = box[:4].astype(int)

        #         if x1 < 0 or y1 < 0 or x2 > img_test.shape[1] or y2 > img_test.shape[0]:
        #             print("Xoá bounding box vì tọa độ nằm ngoài ảnh")
        #             continue
        #         cropped = img_test[y1:y2, x1:x2]

        #         cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        #         cv2.imwrite(f"/content/{x1}.png", cropped * 255)

                # cropped = torch.from_numpy(cropped)  # Uncomment if needed for prediction
            #     pred = self.predict(cropped)
            #     print(pred)
            #     if pred != 1:  # Keep the box only if the prediction is not 0
            #         new_p.append(box)
            #         print("ok")

            # p[0] = new_p  # Update the original list with the filtered boxes

        # except Exception as e:
        #     print(e)
        results = []
        if len(p) == 0 or len(p[0]) == 0:
            print("No object detected.")
            return results

        full_box = torch.zeros_like(p[0][0])
        full_box[2], full_box[3], full_box[4], full_box[6:] = img.shape[3], img.shape[2], 1.0, 1.0
        full_box = full_box.view(1, -1)
       
        critical_iou_index = bbox_iou(full_box[0][:4], p[0][:, :4], iou_thres=0.9, image_shape=img.shape[2:])
        # print(critical_iou_index)
    
        if critical_iou_index.numel() != 0:
            print('da vaog')
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
                    print(pred.shape)
                    print(orig_img.shape)
                    try:

                        for item in pred:
                            print(item.shape)
                            x1, y1, x2, y2 = item[:4]
                            print(x1, x2, y1, y2)
                            # cropped_img = orig_img[y1:y2, x1:x2]
                            # print(cropped_img.shape)
                    except Exception as e:
                        print(e)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
