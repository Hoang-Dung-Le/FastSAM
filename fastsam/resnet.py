import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

class Resnet50:
    def __init__(self) -> None:
        pass
    
    def predict(self, image):

        # Load mô hình ResNet đã được huấn luyện trước
        model = resnet50(pretrained=True)
        model.eval()

        # Chuẩn bị ảnh cho việc dự đoán
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # image = Image.open(image_path)
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)

        # Dự đoán
        with torch.no_grad():
            output = model(input_batch)

        # Lấy kết quả phân loại
        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()
        return predicted_label