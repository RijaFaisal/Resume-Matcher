import os
import pandas as pd
import json
from src.rag.prompts import get_chat_prompt, SYSTEM_ROLE, PROMPT_INSTRUCTION
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def evaluate():
    print("🚀 Starting Prompt Evaluation...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ SKIPPING: GROQ_API_KEY not found.")
        # We don't fail the build in strictly CI if secrets aren't available, 
        # but for this task we want to simulate a real check. 
        # Return 0 to pass 'build' but log warning.
        exit(0) 

    client = Groq(api_key=api_key)
    
    # Test Data (Small subset)
    test_cases = [
        {
            "query": "Edit my resume for a Data Scientist role.",
            "type": "extraction",
            "expected_key": "generate_resume"
        },
        {
            "query": "What is a good summary for a backend engineer?",
            "type": "qa",
            "forbidden_key": "generate_resume"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for case in test_cases:
        print(f"\nEvaluating: '{case['query']}'")
        
        prompt = get_chat_prompt(
            SYSTEM_ROLE, 
            PROMPT_INSTRUCTION, 
            context="Context: Data Science involves statistics and coding.", 
            user_context="My name is Alice.", 
            query=case['query']
        )
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            print(f"Output: {content[:100]}...")
            
            if case["type"] == "extraction":
                if case["expected_key"] in content or "action" in content:
                     print("✅ Passed JSON intent check")
                     passed += 1
                else:
                     print("❌ Failed JSON intent check")
            else:
                if case.get("forbidden_key") and case["forbidden_key"] not in content:
                    print("✅ Passed QA text check")
                    passed += 1
                else:
                    print(f"❌ Failed: Found forbidden key {case.get('forbidden_key')}")
                    
        except Exception as e:
            print(f"❌ API Error: {e}")
            
    print(f"\n📊 Result: {passed}/{total} Passed")
    if passed < total:
        print("❌ Some tests failed.")
        exit(1)
    else:
        print("✅ All tests passed.")
        exit(0)

if __name__ == "__main__":
    evaluate()
