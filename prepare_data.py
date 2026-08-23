import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
target_names = iris.target_names

def generate_jsonl(X_data, y_data, filename, format_type):
    with open(filename, 'w') as f:
        for features, label_idx in zip(X_data, y_data):
            label = target_names[label_idx]
            
            if format_type == 'v1':
                user_text = f"sepal_length: {features[0]}, sepal_width: {features[1]}, petal_length: {features[2]}, petal_width: {features[3]}"
                model_text = label
            elif format_type == 'v2':
                user_text = f"A flower specimen has a sepal length of {features[0]} cm, sepal width of {features[1]} cm, petal length of {features[2]} cm, and petal width of {features[3]} cm. Identify the iris species."
                model_text = f"This is Iris {label}."
            
            # THE CORRECT GEMINI FORMAT
            record = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_text}]
                    },
                    {
                        "role": "model",
                        "parts": [{"text": model_text}]
                    }
                ]
            }
            f.write(json.dumps(record) + '\n')

# Generate Training & Test sets
generate_jsonl(X_train, y_train, "iris_v1_train.jsonl", "v1")
generate_jsonl(X_test, y_test, "iris_v1_test.jsonl", "v1")
generate_jsonl(X_train, y_train, "iris_v2_train.jsonl", "v2")
generate_jsonl(X_test, y_test, "iris_v2_test.jsonl", "v2")

print("Generated ACTUAL Gemini-compatible JSONL files successfully.")