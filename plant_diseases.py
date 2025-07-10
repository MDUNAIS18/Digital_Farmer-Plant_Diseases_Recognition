
# Unified Roboflow model interface for plant disease prediction
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import os

CLASS_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "cause": "Apple scab",
        "solution": "Remove and destroy infected leaves. Apply recommended fungicides. Ensure good air circulation."
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "cause": "Black rot",
        "solution": "Prune and destroy infected branches. Use fungicides. Remove mummified fruit."
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "cause": "Cedar apple rust",
        "solution": "Remove nearby junipers. Apply fungicides at bud break."
    },
    "Apple___healthy": {
        "plant": "Apple",
        "cause": "Healthy",
        "solution": "No action needed."
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "cause": "Healthy",
        "solution": "No action needed."
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "cause": "Healthy",
        "solution": "No action needed."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "cause": "Powdery mildew",
        "solution": "Apply sulfur-based fungicides. Remove infected leaves. Improve air circulation."
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "cause": "Common rust",
        "solution": "Use resistant hybrids. Apply fungicides if severe."
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "cause": "Healthy",
        "solution": "No action needed."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "cause": "Northern Leaf Blight",
        "solution": "Use resistant varieties. Rotate crops. Apply fungicides if needed."
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "cause": "Black rot",
        "solution": "Remove infected leaves and fruit. Apply fungicides. Prune for air flow."
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "cause": "Esca (Black Measles)",
        "solution": "Remove and destroy infected vines. Avoid pruning wounds in wet weather."
    },
    # Add more classes as needed
}

def roboflow_predict(image_path):
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="jzXPy98vn9XnQRzvKCAu"
    )
    result = client.run_workflow(
        workspace_name="mohammed-unais",
        workflow_id="custom-workflow-3",
        images={"image": image_path},
        use_cache=True
    )
    predictions = result[0]["predictions"]["predictions"]
    if not predictions:
        return {
            "plant": None,
            "cause": None,
            "solution": None,
            "class_name": None,
            "confidence": 0,
            "output_image_path": None
        }
    best_pred = max(predictions, key=lambda p: p.get("confidence", 0))
    class_name = best_pred.get("class", "Unknown").strip()
    confidence = best_pred.get("confidence", 0)
    x = best_pred.get("x", 0)
    y = best_pred.get("y", 0)
    width = best_pred.get("width", 0)
    height = best_pred.get("height", 0)

    # Draw bounding box
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    left = x - width / 2
    top = y - height / 2
    right = x + width / 2
    bottom = y + height / 2
    draw.rectangle([left, top, right, bottom], outline="red", width=3)
    draw.text((left, top - 10), f"{class_name} ({confidence:.2%})", fill="red")

    # Save output image
    output_path = os.path.splitext(image_path)[0] + "_output.jpg"
    img.save(output_path)

    info = CLASS_INFO.get(class_name)
    # Always extract plant and disease name from class_name
    if '___' in class_name:
        plant_name, disease_name = class_name.split('___', 1)
    else:
        plant_name, disease_name = class_name, ''

    return {
        "plant": info["plant"] if info else plant_name,
        "cause": info["cause"] if info else disease_name.replace('_', ' '),
        "solution": info["solution"] if info else "No solution found.",
        "class_name": class_name,
        "confidence": confidence,
        "output_image_path": output_path
    }

# Example usage (for testing only)
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not image_path:
        print("No image selected.")
        exit()
    result = roboflow_predict(image_path)
    print(f"Plant: {result['plant']}")
    print(f"Disease: {result['cause']}")
    print(f"Solution: {result['solution']}")
    print(f"Class: {result['class_name']} (confidence: {result['confidence']:.2%})")
    print(f"Output image saved as: {result['output_image_path']}")
