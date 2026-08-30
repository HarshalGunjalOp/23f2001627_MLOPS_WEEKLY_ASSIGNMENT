import re
import datetime
import json
import vertexai
from vertexai.generative_models import GenerativeModel

# Audit log storage
AUDIT_LOG_FILE = "guardrail_audit.log"

def log_audit(event_type, details):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "details": details
    }
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

class InputGuardrail:
    def __init__(self):
        # Rule-based blocklist keywords and regex patterns
        self.injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
            r"system\s+(prompt|instruction|context)",
            r"repeat\s+everything",
            r"print\s+context",
            r"act\s+as",
            r"you\s+are\s+now",
            r"what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+",
            r"translate",
            r"capital\s+of",
            r"fictional\s+universe",
            r"training\s+(examples|dataset|data)"
        ]

    def validate(self, raw_input: str, format_type: str = "v1") -> dict:
        # Check 1: Rule-Based Pattern & Keyword Matching
        for pattern in self.injection_patterns:
            if re.search(pattern, raw_input, re.IGNORECASE):
                log_audit("INPUT_BLOCKED_INJECTION", {"rule": pattern, "input": raw_input})
                return {"blocked": True, "reason": f"Input violates injection/leakage policy: matched '{pattern}'"}

        # Check 2: Structural / Schema Validation
        if format_type == "v1":
            # Strict regex for: sepal_length: float, sepal_width: float, petal_length: float, petal_width: float
            v1_schema = r"^sepal_length:\s*[\d\.]+,\s*sepal_width:\s*[\d\.]+,\s*petal_length:\s*[\d\.]+,\s*petal_width:\s*[\d\.]+$"
            if not re.match(v1_schema, raw_input.strip()):
                log_audit("INPUT_BLOCKED_SCHEMA_MISMATCH", {"expected": "v1_schema", "input": raw_input})
                return {"blocked": True, "reason": "Input failed V1 structural schema validation."}
                
        elif format_type == "v2":
            # Check presence of expected natural language tokens and values
            v2_schema = r"A flower specimen has a sepal length of [\d\.]+ cm, sepal width of [\d\.]+ cm, petal length of [\d\.]+ cm, and petal width of [\d\.]+ cm\.\s*Identify the iris species\."
            if not re.search(v2_schema, raw_input.strip(), re.IGNORECASE):
                log_audit("INPUT_BLOCKED_SCHEMA_MISMATCH", {"expected": "v2_schema", "input": raw_input})
                return {"blocked": True, "reason": "Input failed V2 structural sentence validation."}

        return {"blocked": False, "reason": "Passed input validation"}

class OutputGuardrail:
    def __init__(self):
        self.sensitive_keywords = ["system instruction", "context window", "training examples", "hyperparameter", "fine-tuned"]

    def filter(self, raw_output: str, format_type: str = "v1") -> str:
        # Check 1: Leakage and sensitive text filtering
        for kw in self.sensitive_keywords:
            if kw in raw_output.lower():
                log_audit("OUTPUT_LEAKAGE_REDACTED", {"keyword": kw, "output": raw_output})
                return "Policy Error: Output contained sensitive internal context and was redacted."

        # Check 2: Extract species and force strict format compliance
        lower_out = raw_output.lower()
        species = None
        if "setosa" in lower_out:
            species = "setosa"
        elif "versicolor" in lower_out:
            species = "versicolor"
        elif "virginica" in lower_out:
            species = "virginica"

        if species:
            if format_type == "v1":
                return species
            elif format_type == "v2":
                return f"This is Iris {species}."
        
        # Fallback for unrecognized outputs
        log_audit("OUTPUT_FALLBACK_APPLIED", {"raw_output": raw_output})
        return "Unknown species / Non-compliant output format."

class GuardedIrisPipeline:
    def __init__(self, endpoint_name: str, format_type: str = "v1"):
        self.endpoint_name = endpoint_name
        self.format_type = format_type
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        self.model = GenerativeModel(endpoint_name)

    def predict(self, raw_input: str) -> dict:
        # 1. Run Input Guardrail
        input_check = self.input_guardrail.validate(raw_input, self.format_type)
        if input_check["blocked"]:
            return {
                "blocked": True,
                "reason": input_check["reason"],
                "response": "Request blocked by safety policy."
            }

        # 2. Call LLM Endpoint
        try:
            raw_response = self.model.generate_content(raw_input).text.strip()
        except Exception as e:
            return {"blocked": False, "error": str(e), "response": "Inference failure"}

        # 3. Run Output Guardrail
        safe_response = self.output_guardrail.filter(raw_response, self.format_type)
        return {
            "blocked": False,
            "raw_response": raw_response,
            "response": safe_response
        }