from django.shortcuts import render
import base64
from io import BytesIO
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import os
import tempfile

def index(request):
    if request.method == 'POST' and request.FILES.get('myfile'):
        myfile = request.FILES['myfile']

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            for chunk in myfile.chunks():
                temp_img.write(chunk)
            temp_img_path = temp_img.name

        # Roboflow inference
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="jzXPy98vn9XnQRzvKCAu"
        )

        result = client.run_workflow(
            workspace_name="mohammed-unais",
            workflow_id="custom-workflow-5",
            images={"image": temp_img_path},
            use_cache=True
        )
        predictions = result[0]["predictions"]["predictions"]

        img = Image.open(temp_img_path)
        output_img_b64 = None
        output_img_path = None
        plant_name, disease_name, solution = None, None, None

        

        if predictions:
            best_pred = max(predictions, key=lambda p: p.get("confidence", 0))
            class_name = best_pred.get("class", "Unknown").strip()
            confidence = int(best_pred.get("confidence", 0) * 100)  # Convert to percentage
            x = best_pred.get("x", 0)
            y = best_pred.get("y", 0)
            width = best_pred.get("width", 0)
            height = best_pred.get("height", 0)
            draw = ImageDraw.Draw(img)
            left = x - width / 2
            top = y - height / 2
            right = x + width / 2
            bottom = y + height / 2
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            draw.text((left, top - 10), f"{class_name} ({confidence:.2%})", fill="red")

            # Always extract plant and disease name
            if "___" in class_name:
                plant_name, disease_name = class_name.split("___", 1)
            else:
                plant_name, disease_name = class_name, ""

            print(plant_name)
            print(disease_name)
            # Solution dictionary (expand as needed)
            solution_dict = {
                "Apple_scab_leaf": "Remove infected leaves, apply fungicide.",
                "Black_rot": "Remove mummified fruit, prune infected branches, apply fungicide.",
                "Cedar_apple_rust": "Remove nearby junipers, apply fungicide, prune affected areas.",
                "Powdery_mildew": "Prune affected areas, use sulfur-based fungicide.",
                "Leaf_blight": "Remove infected leaves, improve air circulation, apply fungicide.",
                "Common_rust": "Use resistant hybrids, apply fungicide if severe.",
                "Northern_Leaf_Blight": "Use resistant varieties, rotate crops, apply fungicide.",
                "Esca_(Black_Measles)": "Remove infected vines, avoid pruning wounds, use clean planting material.",
                "Black_rot_grape": "Remove mummified berries, apply fungicide, prune infected shoots.",
                "Healthy": "No action needed.",
            }
            # Normalize for lookup
            def normalize_key(name):
                return name.lower().replace(" ", "_").replace("-", "_")
            disease_key = normalize_key(disease_name)
            solution = solution_dict.get(disease_key, "No solution found.")

            # Save output image
            output_img_path = os.path.splitext(temp_img_path)[0] + "_output.jpg"
            img.save(output_img_path)

        # Convert output image to base64 for web display
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        output_img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Clean up temp file
        os.remove(temp_img_path)

        print(plant_name)
        print(disease_name)
        print(solution)

        return render(request, "plant_app/index.html", {
            'output_img_b64': output_img_b64,
            'output_img_path': output_img_path,
            'plant_name': plant_name,
            'disease_name': disease_name.replace('_', ' ') if disease_name else None,
          
        })

    return render(request, "plant_app/index.html")