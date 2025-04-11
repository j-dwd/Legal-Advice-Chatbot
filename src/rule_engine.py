import json
import os

def apply_neuro_symbolic_rules(text):
    rules_file = os.path.join("models", "neuro_symbolic_rules", "rules.json")
    with open(rules_file, "r", encoding="utf-8") as f:
        rules = json.load(f)
    text_lower = text.lower()
    risk_score = 0.5  # Default
    for rule in rules["rules"]:
        if "contains" in rule["if"]["contract_text"]:
            condition = rule["if"]["contract_text"].split("contains ")[1]
            if condition in text_lower:
                risk_score = rule["then"]["risk"]
                break
    return {"risk_score": float(risk_score)}