
import os
from dotenv import load_dotenv
from groq import Groq

# Load env vars
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"Loaded API Key: {api_key[:10]}... (length: {len(api_key) if api_key else 0})")

if not api_key:
    print("❌ Error: GROQ_API_KEY is not set.")
    exit(1)

client = Groq(api_key=api_key)

print("Testing Groq API connection...")
try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, are you working?",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    print("✅ Success! Response:")
    print(chat_completion.choices[0].message.content)
except Exception as e:
    print("❌ API Call Failed!")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
