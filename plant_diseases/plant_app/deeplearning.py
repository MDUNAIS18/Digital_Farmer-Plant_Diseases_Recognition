from keras.models import load_model
from keras.preprocessing import image
import numpy as np


# Load the pre-trained model (no graph needed for TF 2.x)
model = load_model('plant_app/model.hdf5')

# Define the output dictionary
output_dict = {
    'Apple___Apple_scab': 0,
    'Apple___Black_rot': 1,
    'Apple___Cedar_apple_rust': 2,
    'Apple___healthy': 3,
    'Blueberry___healthy': 4,
    'Cherry_(including_sour)___healthy': 5,
    'Cherry_(including_sour)___Powdery_mildew': 6,
    'Corn_(maize)___Common_rust_': 7,
    'Corn_(maize)___healthy': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9,
    'Grape___Black_rot': 10,
    'Grape___Esca_(Black_Measles)': 11
}

# Inverse mapping to get label from index
output_list = [k for k, v in sorted(output_dict.items(), key=lambda item: item[1])]



# Function to predict from image path (no graph needed)
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize as per model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    predicted_label = output_list[class_index]
    return predicted_label

# Example usage
if __name__ == "__main__":
    image_path = 'test.png'  # Change to your test image path
    result = predict_disease(image_path)
    print("Predicted Disease/Label:", result)
