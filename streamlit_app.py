import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from lime.lime_text import LimeTextExplainer
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from multimodal_processor import process_multimodal_input, enhance_ocr_text
from rule_engine import apply_advanced_rules
from utils import extract_legal_entities
from logger import setup_logger

# Load environment and logger
load_dotenv()
logger = setup_logger()
OPENAI_API_KEY = os.getenv("AIzaSyB3buhIbbp9OYRs-oPq2ytsZeHoZdAuhnQ")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load models
model = RobertaForSequenceClassification.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
ner_pipeline = pipeline("ner", model="nlpaueb/legal-bert-ner")
explainer = LimeTextExplainer(class_names=["Not Audit Clause", "Audit Clause"])

# Streamlit setup
st.set_page_config(page_title="Legal Advice Chatbot", layout="wide")
st.title("Legal Advice Chatbot")
st.markdown("Analyze contracts with BERT and Gemini.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
st.sidebar.header("Chat History")
for i, msg in enumerate(st.session_state.messages):
    st.sidebar.write(f"Q{i+1}: {msg['question'][:50]}... -> {msg['answer'][:50]}...")

# Input
input_type = st.radio("Input Type:", ["Text", "Image", "PDF"])
data = None
if input_type in ["Image", "PDF"]:
    uploaded_file = st.file_uploader(f"Upload {input_type.lower()} file", type=["jpg", "png", "pdf"])
    if uploaded_file:
        with st.spinner("Processing..."):
            raw_text = process_multimodal_input(uploaded_file)
            data = enhance_ocr_text(raw_text)
            logger.info(f"Processed {input_type.lower()} file: {uploaded_file.name}")
        st.success("Processed!")
else:
    data = st.text_area("Enter clause:", height=200)

# Chat input
user_input = st.chat_input("Enter a contract clause or question...")
if user_input or data:
    clause = user_input if user_input else data
    if st.button("Analyze") or user_input:
        with st.spinner("Analyzing..."):
            try:
                # Classification
                def classify_clause(text):
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                    outputs = model(**inputs)
                    return torch.argmax(outputs.logits, dim=-1).item()

                def predict_proba(texts):
                    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                    outputs = model(**inputs)
                    return torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

                # Risk analysis
                def run_risk_analysis(clause):
                    prompt = "Identify risks in this legal clause."
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": clause}]
                    )
                    gpt_risk = response.choices[0].message.content.strip()
                    rule_risk = apply_advanced_rules(clause)
                    return f"{gpt_risk}\n\nRule-Based: {rule_risk}"

                # Process
                label = "Audit Clause" if classify_clause(clause) == 1 else "Not Audit Clause"
                explanation = explainer.explain_instance(clause, predict_proba, num_features=10)
                entities = extract_legal_entities(clause)
                risk_analysis = run_risk_analysis(clause)
                risk_score = np.random.uniform(0, 1)  # Replace with actual scoring
                response = f"Classification: {label}\nRisk Analysis: {risk_analysis}"

                # Store and display
                st.session_state.messages.append({"question": clause, "answer": response})
                for i, msg in enumerate(st.session_state.messages):
                    with st.chat_message("user"):
                        st.write(msg["question"])
                    with st.chat_message("assistant"):
                        st.write(msg["answer"])

                # Tabs for detailed view
                tab1, tab2, tab3, tab4 = st.tabs(["Classification", "Risk", "Entities", "Trends"])
                with tab1:
                    st.write("Classification:", label)
                    st.components.v1.html(explanation.as_html(), height=300)
                with tab2:
                    st.metric("Risk Score", f"{risk_score:.2f}")
                    st.bar_chart({"Risk": [risk_score * 100]})
                with tab3:
                    highlighted = clause
                    for word, entity in entities:
                        highlighted = highlighted.replace(word, f"<span style='color:red'>{word}</span>")
                    st.markdown(highlighted, unsafe_allow_html=True)
                with tab4:
                    if len(st.session_state.messages) > 1:
                        trends = [np.random.uniform(0, 1) for _ in st.session_state.messages]
                        st.line_chart({"Risk Trend": trends})
                    else:
                        st.write("Analyze more clauses for trends.")

                logger.info(f"Clause analyzed: {clause}")
            except Exception as e:
                logger.error(f"Error: {e}")
                st.error(f"Error: {e}")