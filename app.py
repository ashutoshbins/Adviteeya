from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)

        # Call your prediction function
        predictions = predict_image(file_path)
        print(predictions[0])
        # Get the predicted class label
        predicted_class = np.argmax(predictions)

        # Get the class names based on your dataset
        class_names = ['A10', 'A400M', 'AG600', 'AV8B', 'B1', 'B2', 'B52', 'Be200', 'C130', 'C17']  # Replace with actual class names

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class]

        return render_template('result.html', prediction=predicted_class_name, image_path=file_path)

def predict_image(path):
    # Load your model
    model = tf.keras.models.load_model('model.h5')

    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    input_arr = np.array([img_array])

    # Make prediction
    pred = model.predict(input_arr)
    print("pred",pred)
    return pred

if __name__ == '__main__':
    app.run(debug=True)
