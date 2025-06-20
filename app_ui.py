# app_ui.py
import streamlit as st
import logging
import os 

try:
    import novel_core 
except ImportError:
    st.error("错误：无法导入 novel_core.py。请确保该文件存在且包含必要的函数。")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI小说写作助手",
    page_icon="✍️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Logger for UI ---
ui_logger = logging.getLogger("NovelAppUI") # Ensure this logger is configured if you use its methods directly
# BasicConfig is often called once. If novel_core also calls it, ensure it's compatible or use specific handlers.
# For simplicity, assuming basicConfig in novel_core is sufficient for now, or UI logs via add_log.

# --- Initialize Session State ---
def initialize_session_state():
    defaults = {
        "system_initialized": False,
        "log_messages": ["应用已启动，等待初始化..."],
        "current_chapter": 1,
        "current_segment_number": 0,
        "current_generated_text": "", 
        "user_directive_for_current_segment": "",
        "selected_embedding_provider_key": list(novel_core.embedding_providers_map_core.keys())[1], 
        "selected_llm_provider_key": list(novel_core.llm_providers_map_core.keys())[2], 
        "api_keys": {}, 
        "milvus_initialized": False, 
        "max_tokens_per_llm_call": int(os.getenv("MAX_TOKENS_PER_LLM_CALL", "7800")),
        "segments_per_chapter_advance": int(os.getenv("SEGMENTS_PER_CHAPTER_ADVANCE", "3")),
        "last_adopted_segment_text": "这是故事的开端，尚无先前的故事片段。",
        "selected_embedding_provider_identifier": None,
        "selected_st_model_name": None,
        "embedding_dimension": None,
        "current_llm_provider": None,
        "user_directive_for_current_segment_buffer": "", # Buffer for rewrite/expand
        "show_expand_input": False, # For expand/supplement UI toggle
        "system_initialized_successfully": False, # More specific flag for core init success
        "num_recent_segments_to_fetch_ui": 2, # For recent story context
        "llm_temperature": 0.7, # Default temperature
        "novel_md_output_dir_ui": novel_core.NOVEL_MD_OUTPUT_DIR_CORE # Get default from core
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Helper function to add logs to UI and logger ---
def add_log(message, level="info", UImodule="UI"):
    timestamp = f"[{UImodule}][{st.session_state.get('current_llm_provider', 'SYS')}] "
    log_entry = f"{timestamp}{message}"
    st.session_state.log_messages.insert(0, log_entry)
    
    core_logger = logging.getLogger("NovelCore") # Use the same logger name as in novel_core for consistency
    if level == "info": core_logger.info(message) # Log to console via core's logger setup
    elif level == "warning": core_logger.warning(message)
    elif level == "error": core_logger.error(message)


# --- UI Layout ---
st.title("✍️ AI 小说写作助手")

with st.sidebar:
    st.header("系统配置")

    st.session_state.selected_embedding_provider_key = st.selectbox(
        "选择嵌入模型",
        options=list(novel_core.embedding_providers_map_core.keys()),
        format_func=lambda x: f"{x}. {novel_core.embedding_providers_map_core[x][0].upper()} {novel_core.embedding_providers_map_core[x][2]}",
        index=list(novel_core.embedding_providers_map_core.keys()).index(st.session_state.selected_embedding_provider_key),
        key="emb_provider_selector_ui", # Unique key for UI widget
        disabled=st.session_state.system_initialized 
    )

    st.session_state.selected_llm_provider_key = st.selectbox(
        "选择LLM模型",
        options=list(novel_core.llm_providers_map_core.keys()),
        format_func=lambda x: f"{x}. {novel_core.llm_providers_map_core[x].upper()}",
        index=list(novel_core.llm_providers_map_core.keys()).index(st.session_state.selected_llm_provider_key),
        key="llm_provider_selector_ui",
        disabled=st.session_state.system_initialized
    )
    
    with st.expander("高级配置/API Keys (可选)"):
        # Simplified API key input, assuming core logic will use these or fallback to .env
        st.session_state.api_keys[novel_core.OPENAI_API_KEY_ENV_NAME] = st.text_input(
            "OpenAI API Key", value=st.session_state.api_keys.get(novel_core.OPENAI_API_KEY_ENV_NAME, os.getenv(novel_core.OPENAI_API_KEY_ENV_NAME,"")), 
            type="password", key="openai_key_input_ui", disabled=st.session_state.system_initialized
        )
        st.session_state.api_keys[novel_core.GEMINI_API_KEY_ENV_NAME] = st.text_input(
            "Gemini API Key", value=st.session_state.api_keys.get(novel_core.GEMINI_API_KEY_ENV_NAME, os.getenv(novel_core.GEMINI_API_KEY_ENV_NAME,"")), 
            type="password", key="gemini_key_input_ui", disabled=st.session_state.system_initialized
        )
        # Add DeepSeek and Custom Proxy Key inputs similarly if needed

        st.session_state.max_tokens_per_llm_call = st.number_input(
            "LLM单次调用最大Token数", min_value=500, max_value=16000, 
            value=st.session_state.max_tokens_per_llm_call, step=100, key="max_tokens_input_ui"
        )
        st.session_state.llm_temperature = st.slider(
            "LLM Temperature", min_value=0.0, max_value=2.0, 
            value=st.session_state.llm_temperature, step=0.05, key="temp_slider_ui"
        )
        st.session_state.segments_per_chapter_advance = st.number_input(
            "每章节多少片段后提示进阶", min_value=1, max_value=20,
            value=st.session_state.segments_per_chapter_advance, step=1, key="segments_per_chap_input_ui"
        )
        st.session_state.num_recent_segments_to_fetch_ui = st.number_input(
            "检索最近故事片段数量 (上下文)", min_value=1, max_value=5,
            value=st.session_state.num_recent_segments_to_fetch_ui, step=1, key="recent_segs_fetch_ui"
        )
        st.session_state.novel_md_output_dir_ui = st.text_input(
            "Markdown输出目录", 
            value=st.session_state.novel_md_output_dir_ui, 
            key="md_output_dir_ui"
        )


    if not st.session_state.system_initialized:
        if st.button("🚀 初始化系统", key="init_button_main_ui", type="primary", use_container_width=True):
            with st.spinner("系统初始化中，请稍候... (可能需要下载模型)"):
                try:
                    success = novel_core.core_initialize_system(
                        st.session_state.selected_embedding_provider_key,
                        st.session_state.selected_llm_provider_key,
                        st.session_state.api_keys 
                    )
                    if success:
                        st.session_state.system_initialized = True
                        st.session_state.system_initialized_successfully = True # Set this crucial flag
                        add_log("系统初始化成功！", "info", "UI")
                        st.success("系统初始化成功！")
                    # No explicit else, error is raised by core and caught below
                except Exception as e:
                    add_log(f"系统初始化失败: {e}", "error", "UI")
                    st.error(f"初始化失败: {str(e)[:300]}...")
            # No explicit rerun, Streamlit handles it.
    else:
        st.success(f"系统已初始化！LLM: {novel_core.llm_providers_map_core[st.session_state.selected_llm_provider_key].upper()}, "
                   f"Embedding: {novel_core.embedding_providers_map_core[st.session_state.selected_embedding_provider_key][0].upper()}")
        if st.button("🔄 重新初始化/切换模型", key="reinit_button_ui", use_container_width=True):
            st.session_state.system_initialized = False
            st.session_state.system_initialized_successfully = False
            st.session_state.milvus_initialized = False 
            # Clear other relevant states that core_initialize_system would reset
            st.session_state.current_chapter = 1 
            st.session_state.current_segment_number = 0
            st.session_state.current_generated_text = ""
            st.session_state.user_directive_for_current_segment_buffer = ""
            add_log("用户请求重新初始化系统。", "info", "UI")
            # No explicit rerun

if st.session_state.system_initialized and st.session_state.system_initialized_successfully:
    st.header("📝 小说创作区")
    
    # Resume logic based on core's findings
    if 'last_known_chapter' in st.session_state and st.session_state.get('resume_choice_made', False) is False:
        resume_q_text = (f"检测到上次写作到 章节 {st.session_state.last_known_chapter}, "
                         f"片段 {st.session_state.last_known_segment}。是否继续？")
        resume_options = ["从上次继续", "从新章节开始", "从头开始(Ch1, Seg0)"]
        choice = st.radio(resume_q_text, resume_options, key="resume_radio", horizontal=True)
        
        if st.button("确认继续方式", key="confirm_resume"):
            if choice == "从上次继续":
                st.session_state.current_chapter = st.session_state.last_known_chapter
                st.session_state.current_segment_number = st.session_state.last_known_segment
                add_log(f"从章节 {st.session_state.current_chapter}, 片段 {st.session_state.current_segment_number} 后继续。", "info", "UI")
            elif choice == "从新章节开始":
                st.session_state.current_chapter = st.session_state.last_known_chapter + 1
                st.session_state.current_segment_number = 0
                add_log(f"从新章节 {st.session_state.current_chapter} 开始。", "info", "UI")
            else: # From scratch
                st.session_state.current_chapter = 1
                st.session_state.current_segment_number = 0
                add_log("从头开始写作 (章节1, 片段0)。", "info", "UI")
            st.session_state.resume_choice_made = True # Prevent re-asking
            # No explicit rerun

    st.info(f"当前写作进度：章节 {st.session_state.current_chapter}, "
            f"计划生成片段号 {st.session_state.current_segment_number + 1}")

    st.session_state.user_directive_for_current_segment = st.text_area(
        "请输入当前片段的写作指令/概要 (详细描述核心事件、爽点、钩子、情感线、篇幅引导等):",
        height=250,
        key="directive_input_main_ui",
        value=st.session_state.get("user_directive_for_current_segment_buffer", "") 
    )

    if st.button("✨ 生成故事片段", key="generate_segment_button_ui", type="primary"):
        if not st.session_state.user_directive_for_current_segment.strip():
            st.warning("写作指令不能为空！")
        else:
            st.session_state.current_generated_text = "" 
            with st.spinner("AI 正在奋笔疾书...请耐心等待..."):
                try:
                    # *** THIS IS THE CORRECTED FUNCTION CALL ***
                    generated_text = novel_core.core_generate_segment_text_for_ui(
                        st.session_state.user_directive_for_current_segment
                    )
                    st.session_state.current_generated_text = generated_text if generated_text else "AI未能生成有效内容。"
                    st.session_state.user_directive_for_current_segment_buffer = st.session_state.user_directive_for_current_segment
                    add_log("AI已生成新片段。", "info", "UI")
                except Exception as e:
                    add_log(f"生成片段时发生错误: {e}", "error", "UI")
                    st.session_state.current_generated_text = f"生成失败: {str(e)[:300]}..."
            # No explicit rerun

    if st.session_state.current_generated_text:
        st.subheader("🤖 AI 生成的片段")
        # Use markdown for better display, but ensure content is safe or use st.text
        st.markdown(st.session_state.current_generated_text) 
        st.caption(f"当前片段字数: {len(st.session_state.current_generated_text)}")

        st.subheader("操作选项")
        cols_actions = st.columns(5)
        with cols_actions[0]:
            if st.button("👍 采纳", key="adopt_seg_button_ui", use_container_width=True, type="primary"):
                with st.spinner("正在采纳片段..."):
                    try:
                        directive_snippet = st.session_state.user_directive_for_current_segment_buffer.splitlines()[0][:100] \
                            if st.session_state.user_directive_for_current_segment_buffer.splitlines() else "无"
                        
                        success = novel_core.core_adopt_segment_from_ui(
                            st.session_state.current_generated_text,
                            st.session_state.current_chapter,
                            st.session_state.current_segment_number + 1,
                            directive_snippet
                        )
                        if success:
                            st.session_state.current_segment_number += 1
                            st.session_state.current_generated_text = "" 
                            st.session_state.user_directive_for_current_segment_buffer = "" 
                            add_log(f"片段 {st.session_state.current_chapter}-{st.session_state.current_segment_number} 已采纳。", "info", "UI")
                            st.success(f"片段 {st.session_state.current_chapter}-{st.session_state.current_segment_number} 已采纳！")

                            if st.session_state.current_segment_number >= st.session_state.segments_per_chapter_advance:
                                # For simplicity, auto-advance or use a session state flag for next chapter prompt
                                st.session_state.prompt_next_chapter = True
                        else:
                             st.error("采纳片段时发生错误，请检查日志。")
                    except Exception as e:
                        add_log(f"采纳片段时发生错误: {e}", "error", "UI")
                        st.error(f"采纳失败: {str(e)[:200]}...")
                # No explicit rerun

        with cols_actions[1]:
            if st.button("🔄 重写", key="rewrite_seg_button_ui", use_container_width=True):
                st.warning("将使用相同的指令要求AI重写当前片段。")
                st.session_state.current_generated_text = "" 
                with st.spinner("AI 正在尝试重写..."):
                    try:
                        generated_text = novel_core.core_generate_segment_text_for_ui(
                             st.session_state.user_directive_for_current_segment_buffer
                        )
                        st.session_state.current_generated_text = generated_text if generated_text else "AI重写失败。"
                        add_log("AI已重写片段。", "info", "UI")
                    except Exception as e:
                        add_log(f"重写片段时发生错误: {e}", "error", "UI")
                        st.session_state.current_generated_text = f"重写失败: {str(e)[:200]}..."
                # No explicit rerun

        with cols_actions[2]:
            if st.button("✍️ 扩写/补充", key="expand_seg_button_ui", use_container_width=True):
                st.session_state.show_expand_input = True
                # No explicit rerun
        
        with cols_actions[3]:
            if st.button("👎 丢弃", key="discard_seg_button_ui", use_container_width=True):
                st.session_state.current_generated_text = ""
                # user_directive_for_current_segment_buffer is kept for potential reuse/modification
                add_log("当前AI生成片段已丢弃。请修改指令或生成新片段。", "info", "UI")
                st.info("当前AI生成片段已丢弃。")
                # No explicit rerun
        
        with cols_actions[4]:
            if st.button("⏩ 下一片段指令", key="next_seg_directive_button_ui", use_container_width=True):
                if st.session_state.current_generated_text : 
                     add_log("用户跳过当前生成内容，准备为下一片段输入指令（如果当前已采纳）或为当前片段重新输入指令（如果未采纳）。", "info", "UI")
                st.session_state.current_generated_text = ""
                # Buffer is kept so user can slightly modify previous directive if they discarded
                # No explicit rerun

        # Conditional UI for next chapter based on flag
        if st.session_state.get('prompt_next_chapter', False):
            if st.button(f"✅ 进入新章节 ({st.session_state.current_chapter + 1})", key="confirm_next_chapter_ui"):
                st.session_state.current_chapter += 1
                st.session_state.current_segment_number = 0
                st.session_state.last_adopted_segment_text = "这是故事的开端，尚无先前的故事片段。" # Reset for new chapter context
                add_log(f"进入新章节: {st.session_state.current_chapter}", "info", "UI")
                st.session_state.prompt_next_chapter = False # Reset flag
            # No explicit rerun

        if st.session_state.get("show_expand_input", False):
            st.subheader("补充与扩展指令")
            expand_directive = st.text_area(
                "请输入对AI扩写/补充的具体要求:", height=100, key="expand_directive_input_ui"
            )
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("提交补充指令", key="submit_expand_button_ui", type="primary"):
                    if not expand_directive.strip():
                        st.warning("补充指令不能为空！")
                    else:
                        original_text = st.session_state.current_generated_text
                        st.session_state.current_generated_text = "" 
                        with st.spinner("AI 正在补充内容..."):
                            try:
                                continuation_prompt = (
                                    f"你是一位优秀的小说作家，正在基于以下已生成内容和新的指令进行补充和扩展。\n\n"
                                    f"---先前已生成内容---\n{original_text}\n\n"
                                    f"---补充与扩展指令---\n{expand_directive}\n\n"
                                    f"请严格按照“补充与扩展指令”，在不改变“先前已生成内容”核心情节的前提下，进行自然的补充、细节扩展或内容延续。直接开始撰写需要补充或扩展的部分，使其能与原文流畅衔接。"
                                )
                                additional_text = novel_core.core_generate_with_llm( # Use the general LLM call
                                    st.session_state.current_llm_provider,
                                    continuation_prompt,
                                    temperature=st.session_state.llm_temperature,
                                    max_tokens_override=st.session_state.max_tokens_per_llm_call,
                                    system_message_override="你是一位优秀的小说作家，正在基于用户提供的已生成内容和新的补充指令进行内容的补充和扩展。请确保补充内容与原文自然融合。"
                                )
                                if additional_text and additional_text.strip():
                                    add_log(f"AI已生成补充内容 (字数: {len(additional_text)}).", "info", "UI")
                                    # Simple append for now
                                    st.session_state.current_generated_text = original_text + "\n\n" + "---补充内容---\n" + additional_text.strip()
                                else:
                                    add_log("AI未能生成有效补充内容。", "warning", "UI")
                                    st.session_state.current_generated_text = original_text 
                            except Exception as e:
                                add_log(f"扩写/补充时发生错误: {e}", "error", "UI")
                                st.session_state.current_generated_text = original_text + f"\n\n---扩写失败: {str(e)[:200]}---"
                        st.session_state.show_expand_input = False 
            with col_exp2:
                if st.button("取消补充", key="cancel_expand_button_ui"):
                    st.session_state.show_expand_input = False
            # No explicit rerun

elif not st.session_state.system_initialized:
    st.warning("👈 请在侧边栏选择模型并点击“初始化系统”以开始使用。")
else: # system_initialized is True, but system_initialized_successfully is False
    st.error("🤷 系统初始化失败，请检查侧边栏的日志并尝试重新初始化。")


st.sidebar.subheader("运行日志")
log_container = st.sidebar.container()
# A simple way to make the log area scrollable if it gets too long
log_html = "<div style='max-height: 400px; overflow-y: auto; border: 1px solid #e6e6e6; padding: 5px;'>"
for msg in st.session_state.log_messages: # Show all logs, newest first due to insert(0,...)
    log_html += f"<pre style='white-space: pre-wrap; word-wrap: break-word; margin-bottom: 2px; font-size: 0.8em;'>{st.html(msg)}</pre>" # Use st.html for safe HTML
log_html += "</div>"
log_container.markdown(log_html, unsafe_allow_html=True)

if st.sidebar.button("清除日志", key="clear_logs_ui"):
    st.session_state.log_messages = ["日志已清除。"]
    # No explicit rerun