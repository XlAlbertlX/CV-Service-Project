from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FoodClassifier:
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Используем доступную модель для классификации еды
            model_name = "nateraw/food"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Классы для модели food (основные категории)
            self.class_names = [
                "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
                "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
                "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
                "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
                "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
                "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
                "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
                "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
                "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
                "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
                "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
                "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
                "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
                "mussels", "nachos", "omelette", "onion_rings", "oysters",
                "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
                "pho", "pizza", "pork_chop", "poutine", "prime_rib",
                "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
                "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
                "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
                "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
                "waffles"
            ]

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Попробуем альтернативную модель
            try:
                logger.info("Trying alternative model...")
                model_name = "microsoft/resnet-50"
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModelForImageClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                self.class_names = ["Class " + str(i) for i in range(1000)]  # Generic classes
                logger.info("Alternative model loaded successfully")
            except Exception as e2:
                logger.error(f"Error loading alternative model: {e2}")
                raise

    def predict(self, image):
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probabilities.cpu().numpy()

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def get_class_name(self, class_idx):
        """Get human-readable class name"""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx].replace('_', ' ').title()
        return f"Class {class_idx}"