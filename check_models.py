import google.generativeai as genai

from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it before running the application.")
else:
    pass

genai.configure(api_key=API_KEY)

print("Available Models for your key:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
