from inference_sdk import InferenceHTTPClient
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import os

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="jzXPy98vn9XnQRzvKCAu"
)

# Prompt user to select an image file
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Select an image file",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("No image selected.")
    exit()

# Acknowledge connection
print("[INFO] Proceeding with Roboflow API connection (no health check available).")

try:
    result = client.run_workflow(
        workspace_name="mohammed-unais",
        workflow_id="plants-diseases-detection-and-classification-0uvur-instant-3",
        images={"image": image_path},
        use_cache=True
    )
    print("[INFO] Workflow executed successfully.")
except Exception as e:
    print(f"[ERROR] Failed to execute workflow: {e}")
    exit()

# Get predictions
predictions = result[0]["predictions"]["predictions"]

# Load and show input image
img = Image.open(image_path)

if predictions:
    # Pick highest confidence prediction
    best_pred = max(predictions, key=lambda p: p.get("confidence", 0))
    class_name = best_pred.get("class", "Unknown")
    confidence = best_pred.get("confidence", 0)
    x = best_pred.get("x", 0)
    y = best_pred.get("y", 0)
    width = best_pred.get("width", 0)
    height = best_pred.get("height", 0)

    # Draw bounding box
    draw = ImageDraw.Draw(img)
    left = x - width / 2
    top = y - height / 2
    right = x + width / 2
    bottom = y + height / 2
    draw.rectangle([left, top, right, bottom], outline="red", width=3)
    draw.text((left, top - 10), f"{class_name} ({confidence:.2%})", fill="red")

    # Save and show output image
    output_path = os.path.splitext(image_path)[0] + "_output.jpg"
    img.save(output_path)
    img.show(title="Detection Result")

    print(f"✅ Conclusion: {class_name} (confidence: {confidence:.2%})")
    print(f"📸 Output image saved as: {output_path}")
else:
    print("❌ No detections found.")
    img.show(title="No Detection")
