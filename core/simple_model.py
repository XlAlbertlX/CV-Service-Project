import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleFoodClassifier:
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Используем предобученный ResNet
            self.model = models.resnet50(pretrained=True)
            self.model.eval()
            self.model.to(self.device)

            # Трансформации для изображения
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Базовые классы ImageNet (первые 20 для примера)
            self.class_names = [
                "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
                "electric ray", "stingray", "cock", "hen", "ostrich",
                "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
                "robin", "bulbul", "jay", "magpie", "chickadee"
            ]

            logger.info("Simple model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading simple model: {e}")
            raise

    def predict(self, image):
        try:
            # Преобразуем и предсказываем
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            return probabilities.cpu().numpy()

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def get_class_name(self, class_idx):
        """Get human-readable class name"""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx].replace('_', ' ').title()
        return f"Class {class_idx}"