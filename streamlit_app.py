import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from lime.lime_text import LimeTextExplainer
from google.genai import GenerativeModel
from dotenv import load_dotenv
import os
import sys
import numpy as np
from loguru import logger

# Adjust path to include src directory
sys.path.append('./src')
from src.multimodal_processor import process_multimodal_input, enhance_ocr_text
from src.rule_engine import apply_advanced_rules
try:
    from src.utils import extract_legal_entities
    from src.logger import setup_logger
except ModuleNotFoundError as e:
    logger.warning(f"Missing module {e}. Using fallback.")
    def extract_legal_entities(text): return {}
    def setup_logger(): return logger

# Load environment variables and set up logger
load_dotenv()
logger = setup_logger()
GEMINI_API_KEY = os.getenv("AIzaSyB3buhIbbp9OYRs-oPq2ytsZeHoZdAuhnQ")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in .env file")
    st.error("Gemini API key missing. Please set GEMINI_API_KEY in .env.")
    st.stop()

# Initialize Gemini model
try:
    gemini_model = GenerativeModel(api_key=GEMINI_API_KEY, model_name='gemini-2.0-pro')
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Gemini initialization failed: {e}")
    st.error(f"Failed to initialize Gemini: {e}")
    st.stop()

# Load and configure BERT model
try:
    model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ner_pipeline = pipeline("ner", model="nlpaueb/legal-bert-ner")
    explainer = LimeTextExplainer(class_names=["Not Audit Clause", "Audit Clause"])
    logger.info("BERT and NER models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    st.error(f"Failed to load models: {e}")
    st.stop()

# Streamlit setup
st.set_page_config(page_title="Legal Advice Chatbot", layout="wide")
st.title("Legal Advice Chatbot (BERT + Gemini)")
st.markdown("Analyze contracts with BERT classification and Gemini-powered responses and risk analysis.")

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
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            raw_text = process_multimodal_input(file_path)
            data = enhance_ocr_text(raw_text)
            os.remove(file_path)
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
                # Classification with BERT
                def classify_clause(text):
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                    outputs = model(**inputs)
                    return torch.argmax(outputs.logits, dim=-1).item()

                def predict_proba(text):
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                    outputs = model(**inputs)
                    return torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]

                # Chatbot response with Gemini
                def get_gemini_response(prompt):
                    response = gemini_model.generate_content(f"Legal expert: {prompt}")
                    return response.text.strip() if response.text else "No response from Gemini."

                # Risk analysis with Gemini and rule engine
                def run_risk_analysis(clause):
                    prompt = "Identify risks in this legal clause and provide a concise analysis."
                    gemini_response = gemini_model.generate_content(f"{prompt}\nClause: {clause}")
                    gemini_risk = gemini_response.text.strip() if gemini_response.text else "No risk analysis available."
                    rule_risk = apply_advanced_rules(clause)
                    return f"Gemini Analysis: {gemini_risk}\nRule-Based: {rule_risk}"

                # Process
                label = "Audit Clause" if classify_clause(clause) == 1 else "Not Audit Clause"
                explanation = explainer.explain_instance(clause, predict_proba, num_features=6)
                entities = extract_legal_entities(clause)
                risk_analysis = run_risk_analysis(clause)
                risk_score = apply_advanced_rules(clause)["risk_score"]
                chatbot_response = get_gemini_response(clause)
                response = (f"Classification: {label}\n"
                           f"Chatbot Response: {chatbot_response}\n"
                           f"Risk Analysis: {risk_analysis}\n"
                           f"Overall Risk Score: {risk_score:.2f}")

                # Store and display
                st.session_state.messages.append({"question": clause, "answer": response})
                for i, msg in enumerate(st.session_state.messages):
                    with st.chat_message("user"):
                        st.write(msg["question"])
                    with st.chat_message("assistant"):
                        st.write(msg["answer"])

                # Tabs for detailed view
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Classification", "Chatbot", "Risk", "Entities", "Trends"])
                with tab1:
                    st.write("Classification:", label)
                    st.components.v1.html(explanation.as_html(), height=300)
                with tab2:
                    st.write("Chatbot Response:", chatbot_response)
                with tab3:
                    st.metric("Risk Score", f"{risk_score:.2f}")
                    st.bar_chart({"Risk": [risk_score * 100]})
                with tab4:
                    highlighted = clause
                    for word, entity in entities.items():
                        highlighted = highlighted.replace(word, f"<span style='color:red'>{word}</span>")
                    st.markdown(highlighted, unsafe_allow_html=True)
                with tab5:
                    if len(st.session_state.messages) > 1:
                        trends = [apply_advanced_rules(msg["question"])["risk_score"] for msg in st.session_state.messages]
                        st.line_chart({"Risk Trend": trends})
                    else:
                        st.write("Analyze more clauses for trends.")

                logger.info(f"Clause analyzed: {clause[:50]}...")
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                st.error(f"Error during analysis: {str(e)}")