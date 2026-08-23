import vertexai
from vertexai.tuning import sft
import os

PROJECT_ID = os.popen('gcloud config get-value project').read().strip()
BUCKET_NAME = f"{PROJECT_ID}-llm-tuning"

vertexai.init(project=PROJECT_ID, location="us-central1") #

def launch_tuning(train_file, job_name):
    print(f"Launching {job_name}...")
    job = sft.train(
        source_model="gemini-2.5-flash", # <-- Bumped to the supported 2.5 generation
        train_dataset=f"gs://{BUCKET_NAME}/{train_file}",
        epochs=3, 
        tuned_model_display_name=job_name, 
    )
    return job

# Launch both jobs
job_v1 = launch_tuning("iris_v1_train.jsonl", "iris-v1-raw-tuning")
job_v2 = launch_tuning("iris_v2_train.jsonl", "iris-v2-desc-tuning")

print("Both jobs submitted! Check the Vertex AI Console to monitor progress.")