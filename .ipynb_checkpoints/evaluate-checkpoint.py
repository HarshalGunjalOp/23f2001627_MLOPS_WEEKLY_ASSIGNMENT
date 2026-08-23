import json
import vertexai
from vertexai.generative_models import GenerativeModel
from sklearn.metrics import accuracy_score, classification_report
import os

PROJECT_ID = os.popen('gcloud config get-value project').read().strip()
vertexai.init(project=PROJECT_ID, location="us-central1")

# Your specific deployed endpoints
ENDPOINT_V1 = "projects/678631406051/locations/us-central1/endpoints/1337820327735984128"
ENDPOINT_V2 = "projects/678631406051/locations/us-central1/endpoints/5758103362000125952"

def evaluate_model(endpoint_name, test_file, format_type):
    model = GenerativeModel(endpoint_name)
    y_true, y_pred = [], []
    valid_formats = 0
    total = 0
    
    with open(test_file, 'r') as f:
        for line in f:
            total += 1
            record = json.loads(line)
            
            # THE FIX: Parse the new Gemini 'contents' format
            input_text = record["contents"][0]["parts"][0]["text"]
            expected_output = record["contents"][1]["parts"][0]["text"]
            
            y_true.append(expected_output)
            
            # Predict using the extracted input
            response = model.generate_content(input_text)
            pred_text = response.text.strip()
            y_pred.append(pred_text)
            
            # Format Compliance Check
            if format_type == 'v1':
                if pred_text.lower() in ["setosa", "versicolor", "virginica"]:
                    valid_formats += 1
            elif format_type == 'v2':
                if pred_text.startswith("This is Iris ") and pred_text.endswith("."):
                    valid_formats += 1
                    
    print(f"\n--- Evaluation: {format_type.upper()} ---")
    print(f"Format Compliance: {(valid_formats/total)*100:.1f}%")
    print(f"Exact Match Accuracy: {accuracy_score(y_true, y_pred)*100:.1f}%")
    print(classification_report(y_true, y_pred, zero_division=0))

evaluate_model(ENDPOINT_V1, "iris_v1_test.jsonl", "v1")
evaluate_model(ENDPOINT_V2, "iris_v2_test.jsonl", "v2")