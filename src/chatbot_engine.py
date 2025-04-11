from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

def legal_chatbot_response(query, context):
    context = context or "No context provided"
    prompt = f"Context: {context}\nQuestion: {query}\nProvide a concise legal response:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()