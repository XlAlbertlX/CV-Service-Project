from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image
import io

try:
    from core.model import FoodClassifier

    classifier = FoodClassifier()
    print("Food model loaded successfully!")
except Exception as e:
    print(f"Error loading food model: {e}")
    try:
        from core.simple_model import SimpleFoodClassifier

        classifier = SimpleFoodClassifier()
        print("Simple model loaded successfully!")
    except Exception as e2:
        print(f"Error loading simple model: {e2}")
        classifier = None

from core.preprocessing import load_image

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and classifier:
            try:
                # Чтение изображения
                image_bytes = file.read()
                image = load_image(image_bytes)

                # Предсказание
                predictions = classifier.predict(image)
                predicted_class = np.argmax(predictions)
                confidence = float(np.max(predictions))
                class_name = classifier.get_class_name(predicted_class)

                return render_template('index.html',
                                       prediction=predicted_class,
                                       class_name=class_name,
                                       confidence=confidence,
                                       success=True)

            except Exception as e:
                return render_template('index.html', error=f'Error processing image: {str(e)}')

        elif not classifier:
            return render_template('index.html', error='No model available. Please check installation.')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)