# test_gemini.py
import google.generativeai as genai
import os
import time # For adding delays between retries or tests

# 尝试从环境变量获取代理设置
HTTP_PROXY = os.environ.get("HTTP_PROXY")
HTTPS_PROXY = os.environ.get("HTTPS_PROXY")

print(f"当前 HTTP_PROXY: {HTTP_PROXY}")
print(f"当前 HTTPS_PROXY: {HTTPS_PROXY}")

# 获取 Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("错误：GEMINI_API_KEY 环境变量未设置！")
    exit()

try:
    print(f"正在配置 Gemini API Key...")
    genai.configure(api_key=GEMINI_API_KEY)

    print("\n列出可用的 Gemini 模型 (支持 'generateContent')：")
    print("--------------------------------------------------")
    models_to_test = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  模型: {m.name}")
            models_to_test.append(m.name) # Add to our list for testing
    
    if not models_to_test:
        print("错误：未能找到任何支持 'generateContent' 的可用模型。请检查您的 API Key 权限或区域。")
        exit()
    print("--------------------------------------------------")

    successful_models = []
    failed_models = {}

    print("\n开始逐个测试可用模型：")
    print("==========================")

    for model_name in models_to_test:
        print(f"\n[测试模型]: {model_name}")
        print("-------------------------")
        
        # 短暂延迟，避免过于频繁的请求，特别是如果上一个请求因配额失败
        time.sleep(2) # 2秒延迟

        try:
            print(f"  正在创建 '{model_name}' 模型实例...")
            model_instance = genai.GenerativeModel(model_name)

            print(f"  正在向 '{model_name}' 发送请求...")
            # 使用一个非常简短的、不太可能消耗大量token的prompt
            prompt = "你好！" 
            
            # 配置安全设置为最低，以避免因内容安全策略导致误判为模型不可用
            # (注意：实际应用中应根据需要调整安全设置)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            response = model_instance.generate_content(
                prompt,
                safety_settings=safety_settings
            )

            print(f"  [成功] 模型: {model_name}")
            print(f"    响应: {response.text[:100]}..." if response.text else "无文本响应") # 打印部分响应
            successful_models.append(model_name)
            print("-------------------------")
            # 如果找到一个能工作的，可以考虑在这里停止，或者继续测试所有模型
            # break # 如果只想找到第一个能工作的就取消注释这行

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f"  [失败] 模型: {model_name}")
            print(f"    错误类型: {error_type}")
            # 简化错误信息打印，避免过多重复的quota信息
            if "ResourceExhausted" in error_type or "429" in error_message:
                print(f"    错误: 配额超限 (429 ResourceExhausted)")
            elif "NotFound" in error_type or "404" in error_message:
                 print(f"    错误: 模型未找到 (404 Not Found)")
            else:
                print(f"    错误信息: {error_message[:200]}...") # 打印部分错误信息
            failed_models[model_name] = f"{error_type}: {error_message[:100]}..."
            print("-------------------------")
            
            # 如果是配额错误，可以增加更长的等待时间
            if "ResourceExhausted" in error_type or "429" in error_message:
                print("    检测到配额错误，将等待10秒后尝试下一个模型...")
                time.sleep(10)


    print("\n\n测试总结：")
    print("============")
    if successful_models:
        print("成功调用的模型：")
        for sm in successful_models:
            print(f"  - {sm}")
    else:
        print("没有模型调用成功。")

    if failed_models:
        print("\n调用失败的模型及原因 (摘要)：")
        for fm_name, fm_error in failed_models.items():
            print(f"  - {fm_name}: {fm_error}")

except Exception as e:
    # 捕获配置或其他顶级错误
    print(f"\n脚本执行过程中发生顶级错误:")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")

    # ... (在 test_gemini.py 中，打印完可用模型列表之后) ...

model_to_use_for_generate = "models/gemini-2.5-flash-preview-05-20" # 或者你主脚本中配置的那个
# 确保这个模型在 models_to_test 列表中，或者直接使用
if model_to_use_for_generate not in models_to_test and models_to_test:
    print(f"警告: {model_to_use_for_generate} 不在可用列表中，将使用列表中的第一个进行内容生成测试。")
    model_to_use_for_generate = models_to_test[0]
elif not models_to_test:
    print("没有模型可用于内容生成测试。")
    exit()


print(f"\n--- 测试使用模型 '{model_to_use_for_generate}' 进行 generateContent ---")
try:
    model_instance = genai.GenerativeModel(model_to_use_for_generate)
    simple_prompt = "写一个关于猫的简短一行笑话。"
    print(f"  发送简单 Prompt: '{simple_prompt}'")
    
    safety_settings_for_generate = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        # ... 其他安全设置 ...
    ]

    response = model_instance.generate_content(
        simple_prompt,
        safety_settings=safety_settings_for_generate
    )

    if response.prompt_feedback and response.prompt_feedback.block_reason:
        print(f"  [失败] Prompt 被阻止: {response.prompt_feedback.block_reason}")
    elif not response.parts or not response.text:
        print(f"  [失败] 模型返回空内容。")
        if hasattr(response, 'candidates'):
            for cand in response.candidates:
                print(f"    候选完成原因: {cand.finish_reason.name if hasattr(cand.finish_reason, 'name') else cand.finish_reason}")
    else:
        print(f"  [成功] 模型: {model_to_use_for_generate}")
        print(f"    响应: {response.text}")

except Exception as e_gen:
    print(f"  [失败] 调用 generateContent 时发生错误:")
    print(f"    错误类型: {type(e_gen).__name__}")
    print(f"    错误信息: {e_gen}")
print("---------------------------------------------------------")