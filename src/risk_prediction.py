from openai import OpenAI
from rule_engine import apply_neuro_symbolic_rules
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

def predict_risk(text):
    rules_output = apply_neuro_symbolic_rules(text)
    risk_score = rules_output["risk_score"]
    prompt = f"Analyze the legal risks in this clause: {text}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    gpt_analysis = response.choices[0].message.content.strip()
    return {"risk_score": risk_score, "analysis": gpt_analysis}