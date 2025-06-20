# main_script_minimal_gemini_test.py
import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GEMINI_LLM_MODEL_FOR_TEST = "models/gemini-2.5-flash" # 你的目标模型

# !!! 使用从文档图片中获取的准确输出 token 限制 !!!
MODEL_MAX_OUTPUT_TOKENS_LIMIT = 65536 # <--- 更新为 65536

llm_client_minimal = None 

def generate_gemini_minimal(prompt, max_tokens_to_request=MODEL_MAX_OUTPUT_TOKENS_LIMIT, temp=0.7):
    # ... (函数其余部分与上一版完全相同) ...
    global llm_client_minimal
    print(f"\n--- Minimal Test: Sending Prompt to GEMINI ---")
    try:
        if not llm_client_minimal:
            api_key = os.getenv(GEMINI_API_KEY_ENV_VAR)
            if not api_key: 
                raise ValueError(f"环境变量 {GEMINI_API_KEY_ENV_VAR} 未设置!")
            genai.configure(api_key=api_key)
            
            llm_client_minimal = genai.GenerativeModel(
                GEMINI_LLM_MODEL_FOR_TEST,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens_to_request, 
                    temperature=temp
                )
            )
            print(f"Minimal Test: LLM Client initialized with model {GEMINI_LLM_MODEL_FOR_TEST} and max_tokens={max_tokens_to_request}.")
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        print(f"  Minimal Test: Safety settings: {safety_settings}")

        print(f"--- Minimal Test: Sending Prompt ---\n{prompt}\n------------------------------------")
        response = llm_client_minimal.generate_content(prompt, safety_settings=safety_settings)
        
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"Prompt Blocked: {response.prompt_feedback.block_reason}"
        
        if not response.parts:
            reason_name = "UNKNOWN_REASON"
            finish_reason_val = "N/A"
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0] 
                finish_reason_val = candidate.finish_reason
                if hasattr(candidate.finish_reason, 'name'):
                    reason_name = candidate.finish_reason.name
            return f"No parts in response. Finish reason (val): {finish_reason_val}, Finish reason (name): {reason_name}"
        
        return response.text.strip() if response.text else "Empty text in response"

    except google_exceptions.RetryError as e: 
        return f"RetryError (Timeout/Connection): {e}"
    except Exception as e:
        return f"Error in minimal Gemini call: {type(e).__name__}: {e}"

if __name__ == "__main__":
    print(f"Minimal Test: HTTP_PROXY={os.getenv('HTTP_PROXY')}")
    print(f"Minimal Test: HTTPS_PROXY={os.getenv('HTTPS_PROXY')}")
    
    test_prompt_minimal = "写一个关于月亮的简短诗句。" 
    
    result = generate_gemini_minimal(test_prompt_minimal) 
    
    print("\n--- Minimal Test Result ---")
    print(result)