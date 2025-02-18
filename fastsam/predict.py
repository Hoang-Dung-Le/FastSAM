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
from torchvision.models import resnet18
import torch.nn as nn
import cv2
from PIL import Image
from google.colab.patches import cv2_imshow


class SimpleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 56 * 56)
        x = self.fc1(x)
        return x



class FastSAMPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None,output=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'
        self.output =output
        self.model_1 = SimpleNet(num_classes=2)
        self.model_1.load_state_dict(torch.load('/content/drive/MyDrive/CV/fastsam/classifier_checkpoint/modelresnet_ade20k_val.pth'))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # self.model_1 = resnet18(pretrained=True)
        # self.model_1.eval()

        # # print(self.model_1)

        # # Define image preprocessing
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     # transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def predict(self, image):
        # self.model_1 = self._load_model('/content/drive/MyDrive/CV/fastsam/classifier_checkpoint/model_resnet34.pth', 2)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Thêm chiều batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_1.to(device)
        input_batch = input_batch.to(device)
        input_batch = input_batch.float()
        with torch.no_grad():
            self.model_1.eval()
            output = self.model_1(input_batch)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

    def _load_model(self, model_path, num_classes):
        # Khởi tạo mô hình ResNet34
        # model = models.resnet34()
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load trạng thái đã được lưu của mô hình
        # model.load_state_dict(torch.load(model_path))
        # return model
        pass

    def postprocess(self, preds, img, orig_imgs):
        """TODO: filter by classes."""
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        
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
            full_box[0][4] = p[0][critical_iou_index][:,4]
            full_box[0][6:] = p[0][critical_iou_index][:,6:]
            p[0][critical_iou_index] = full_box
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        
        # print(proto.shape)
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            if not len(pred):  # save empty boxes
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
                continue
            if self.args.retina_masks:
                kept_boxes = torch.tensor([])
                kept_boxes = kept_boxes.cuda()
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)     
                try:
                    test = torch.clone(pred)
                    for idx in range(test.shape[0]):
                        pr = test[idx]
                        box_np = pr.detach().cpu().numpy()
                        x1, y1, x2, y2 = box_np[:4].astype(int)
                        cropped_img = orig_img[y1:y2, x1:x2]
                        path = self.output + f"img{x1}.png"
                        # print(path)
                        # cv2.imwrite(path, cropped_img)
                        cropped_img = cropped_img / 255.
                        prediction = self.predict(cropped_img)
                        # print(prediction)
                        if prediction == 1:  
                            
                            kept_boxes = torch.cat([kept_boxes, test[idx].unsqueeze(0)])
                        # print(prediction)
                    pred = kept_boxes
                except Exception as e:
                    print("loi ne ", e)

                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
