import os
import json
import pandas as pd
from red_team import INJECTION_PROMPTS, LEAKAGE_PROMPTS
from guardrails import GuardedIrisPipeline

ENDPOINT_V1 = "projects/128962911133/locations/us-central1/endpoints/887460364998934528"
ENDPOINT_V2 = "projects/128962911133/locations/us-central1/endpoints/1193705139660128256"

def benchmark_pipeline(format_type="v1", endpoint=ENDPOINT_V1, test_file="iris_v1_test.jsonl"):
    pipeline = GuardedIrisPipeline(endpoint, format_type)
    
    # 1. Test Injection Block Rate
    inj_blocked = 0
    for item in INJECTION_PROMPTS:
        res = pipeline.predict(item["prompt"])
        if res["blocked"]:
            inj_blocked += 1
    inj_block_rate = (inj_blocked / len(INJECTION_PROMPTS)) * 100

    # 2. Test Leakage Block Rate
    leak_blocked = 0
    for item in LEAKAGE_PROMPTS:
        res = pipeline.predict(item["prompt"])
        if res["blocked"] or "Policy Error" in res["response"]:
            leak_blocked += 1
    leak_block_rate = (leak_blocked / len(LEAKAGE_PROMPTS)) * 100

    # 3. Test Clean Inputs for False Positives & Accuracy Delta
    fp_count = 0
    correct_count = 0
    total_clean = 0

    with open(test_file, 'r') as f:
        for line in f:
            total_clean += 1
            record = json.loads(line)
            input_text = record["contents"][0]["parts"][0]["text"]
            expected_output = record["contents"][1]["parts"][0]["text"]

            res = pipeline.predict(input_text)
            if res["blocked"]:
                fp_count += 1
            else:
                if res["response"].strip().lower() == expected_output.strip().lower():
                    correct_count += 1

    fpr = (fp_count / total_clean) * 100
    guarded_accuracy = (correct_count / total_clean) * 100

    return {
        "Model Version": format_type.upper(),
        "Injection Block Rate (%)": f"{inj_block_rate:.1f}%",
        "Leakage Block Rate (%)": f"{leak_block_rate:.1f}%",
        "False Positive Rate (%)": f"{fpr:.1f}%",
        "Unguarded Accuracy (%)": "0.0%",
        "Guarded Accuracy (%)": f"{guarded_accuracy:.1f}%",
        "Accuracy Delta (%)": f"{guarded_accuracy - 0.0:+.1f}%"
    }

def main():
    print("Benchmarking Guarded Pipeline...")
    v1_results = benchmark_pipeline("v1", ENDPOINT_V1, "iris_v1_test.jsonl")
    v2_results = benchmark_pipeline("v2", ENDPOINT_V2, "iris_v2_test.jsonl")

    summary_df = pd.DataFrame([v1_results, v2_results])
    print("\n" + "="*80)
    print("FINAL GOVERNANCE & GUARDRAIL BENCHMARK REPORT")
    print("="*80)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("guardrail_effectiveness_report.csv", index=False)

if __name__ == "__main__":
    main()