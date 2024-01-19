import torch
from torchvision import transforms, models
import torch.nn as nn

from PIL import Image

class CustomResNet34Classifier:
    def __init__(self, model_path):
        # Khởi tạo mô hình và load trạng thái đã được lưu
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self, model_path):
        # Khởi tạo mô hình ResNet34
        model = models.resnet34()
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load trạng thái đã được lưu của mô hình
        model.load_state_dict(torch.load(model_path))
        
        return model

    def predict(self, image):
        # Mở ảnh và áp dụng các biến đổi
        # image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Thêm chiều batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Chuyển mô hình và dữ liệu lên GPU nếu có sẵn
        self.model.to(device)
        input_batch = input_batch.to(device)

        # Dự đoán
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_batch)

        # Lấy nhãn có xác suất cao nhất
        _, predicted_class = torch.max(output, 1)

        return predicted_class.item()