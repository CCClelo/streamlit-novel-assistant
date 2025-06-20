# app_ui.py
# Python Standard Library Imports
import logging
import os

# Third-party Imports
# IMPORTANT: Streamlit import should be one of the first for it to work correctly
import streamlit as st

# Attempt to import core logic; this MUST succeed for the app to run.
try:
    import novel_core 
except ImportError:
    # This error is critical and will be displayed by Streamlit if novel_core.py is missing
    # or has unrecoverable errors at import time.
    st.error("CRITICAL ERROR: 无法导入 novel_core.py。请确保该文件与 app_ui.py 在同一目录下，"
             "并且 novel_core.py 本身没有导致导入失败的严重错误。请检查部署日志获取详细信息。")
    st.stop() # Stop execution of the Streamlit app if core logic cannot be imported.

# --- Page Configuration (set this early) ---
st.set_page_config(
    page_title="AI小说写作助手",
    page_icon="✍️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Logger for UI (can be simple if core logic handles detailed logging) ---
# Using Streamlit's native way to show messages is often better than a separate UI logger.
# We will use the add_log function to also log to console via novel_core's logger.

# --- Initialize Session State ---
# This function is crucial for Streamlit apps
def initialize_session_state():
    defaults = {
        "system_initialized_attempted": False, # Tracks if init button was clicked
        "system_initialized_successfully": False, # Tracks if core_initialize_system succeeded
        "log_messages": ["应用已启动，等待初始化..."],
        "current_chapter": 1,
        "current_segment_number": 0, # Crucial: ensure this is an int
        "current_generated_text": "", 
        "user_directive_for_current_segment": "",
        # Default selections for providers (using first key as a simple default)
        "selected_embedding_provider_key": list(novel_core.embedding_providers_map_core.keys())[0], 
        "selected_llm_provider_key": list(novel_core.llm_providers_map_core.keys())[0], 
        "api_keys": {}, 
        "milvus_initialized_core": False, # Tracks if core Milvus init part succeeded
        "max_tokens_per_llm_call": int(os.getenv("MAX_TOKENS_PER_LLM_CALL", "7800")),
        "segments_per_chapter_advance": int(os.getenv("SEGMENTS_PER_CHAPTER_ADVANCE", "3")),
        "last_adopted_segment_text": "这是故事的开端，尚无先前的故事片段。",
        "selected_embedding_provider_identifier": None, # Set by core_init
        "selected_st_model_name": None, # Set by core_init
        "embedding_dimension": None, # Set by core_init
        "current_llm_provider": None, # Set by core_init
        "user_directive_for_current_segment_buffer": "", 
        "show_expand_input": False, 
        "num_recent_segments_to_fetch_ui": 2, 
        "llm_temperature": 0.7, 
        "novel_md_output_dir_ui": novel_core.NOVEL_MD_OUTPUT_DIR_CORE,
        "last_known_chapter": None, # For resume logic
        "last_known_segment": None, # For resume logic
        "resume_choice_made": False # For resume logic
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state() # Call it once at the start

# --- Helper function to add logs to UI and console logger ---
def add_log(message: str, level: str = "info", UImodule: str = "UI"):
    timestamp = f"[{UImodule}][{st.session_state.get('current_llm_provider', 'SYS')}] "
    log_entry = f"{timestamp}{message}"
    
    # Ensure log_messages is always a list
    if not isinstance(st.session_state.get("log_messages"), list):
        st.session_state.log_messages = []
    st.session_state.log_messages.insert(0, log_entry)
    
    # Use the logger from novel_core for console output
    core_logger = logging.getLogger("NovelCore") # Assumes novel_core configures this logger
    if level == "info": core_logger.info(message)
    elif level == "warning": core_logger.warning(message)
    elif level == "error": core_logger.error(message)


# --- UI Layout ---
st.title("✍️ AI 小说写作助手")

with st.sidebar:
    st.header("系统配置")

    # Embedding Provider Selection
    # Ensure novel_core.embedding_providers_map_core is accessible
    # Default index handling
    emb_keys = list(novel_core.embedding_providers_map_core.keys())
    default_emb_idx = emb_keys.index(st.session_state.selected_embedding_provider_key) if st.session_state.selected_embedding_provider_key in emb_keys else 0
    
    st.session_state.selected_embedding_provider_key = st.selectbox(
        "选择嵌入模型",
        options=emb_keys,
        format_func=lambda x: f"{x}. {novel_core.embedding_providers_map_core[x][0].upper()} {novel_core.embedding_providers_map_core[x][2]}",
        index=default_emb_idx,
        key="emb_provider_selector_ui_v2", # Changed key to ensure reset if needed
        disabled=st.session_state.system_initialized_attempted 
    )

    # LLM Provider Selection
    llm_keys = list(novel_core.llm_providers_map_core.keys())
    default_llm_idx = llm_keys.index(st.session_state.selected_llm_provider_key) if st.session_state.selected_llm_provider_key in llm_keys else 0

    st.session_state.selected_llm_provider_key = st.selectbox(
        "选择LLM模型",
        options=llm_keys,
        format_func=lambda x: f"{x}. {novel_core.llm_providers_map_core[x].upper()}",
        index=default_llm_idx,
        key="llm_provider_selector_ui_v2",
        disabled=st.session_state.system_initialized_attempted
    )
    
    with st.expander("高级配置/API Keys (可选)"):
        # API Key inputs are fine as they were
        st.session_state.api_keys[novel_core.OPENAI_API_KEY_ENV_NAME] = st.text_input(
            "OpenAI API Key", value=st.session_state.api_keys.get(novel_core.OPENAI_API_KEY_ENV_NAME, os.getenv(novel_core.OPENAI_API_KEY_ENV_NAME,"")), 
            type="password", key="openai_key_input_ui", disabled=st.session_state.system_initialized_attempted
        )
        st.session_state.api_keys[novel_core.GEMINI_API_KEY_ENV_NAME] = st.text_input(
            "Gemini API Key", value=st.session_state.api_keys.get(novel_core.GEMINI_API_KEY_ENV_NAME, os.getenv(novel_core.GEMINI_API_KEY_ENV_NAME,"")), 
            type="password", key="gemini_key_input_ui", disabled=st.session_state.system_initialized_attempted
        )
        # Add DeepSeek and Custom Proxy Key inputs similarly if needed by novel_core

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
            "检索最近故事片段数量 (上下文)", min_value=0, max_value=5, # Allow 0 for no recent context
            value=st.session_state.num_recent_segments_to_fetch_ui, step=1, key="recent_segs_fetch_ui"
        )
        st.session_state.novel_md_output_dir_ui = st.text_input(
            "Markdown输出目录", 
            value=st.session_state.novel_md_output_dir_ui, 
            key="md_output_dir_ui"
        )

    # Initialization Button Logic
    if not st.session_state.system_initialized_attempted:
        if st.button("🚀 初始化系统", key="init_button_main_ui", type="primary", use_container_width=True):
            st.session_state.system_initialized_attempted = True # Mark that init process has started
            with st.spinner("系统初始化中，请稍候... (可能需要下载模型)"):
                try:
                    # core_initialize_system now sets 'system_initialized_successfully' in session_state
                    novel_core.core_initialize_system(
                        st.session_state.selected_embedding_provider_key,
                        st.session_state.selected_llm_provider_key,
                        st.session_state.api_keys 
                    )
                    # No need to set system_initialized_successfully here, core does it.
                    # add_log is also called from core.
                except Exception as e:
                    # core_initialize_system should set system_initialized_successfully to False on error
                    # and log the error. UI can just show a generic error message.
                    st.error(f"初始化失败，请检查侧边栏日志。详细错误: {str(e)[:200]}...") # Show snippet
            # Streamlit will rerun automatically after button press and state change
    
    elif st.session_state.system_initialized_successfully: # If init was ATTEMPTED and SUCCEEDED
        current_llm_display = novel_core.llm_providers_map_core.get(st.session_state.selected_llm_provider_key, '未知').upper()
        current_emb_display = novel_core.embedding_providers_map_core.get(st.session_state.selected_embedding_provider_key, ['未知'])[0].upper()
        st.success(f"系统已初始化！LLM: {current_llm_display}, Embedding: {current_emb_display}")
        if st.button("🔄 重新初始化/切换模型", key="reinit_button_ui", use_container_width=True):
            st.session_state.system_initialized_attempted = False # Allow re-init attempt
            st.session_state.system_initialized_successfully = False
            st.session_state.milvus_initialized_core = False 
            st.session_state.current_chapter = 1 
            st.session_state.current_segment_number = 0
            st.session_state.current_generated_text = ""
            st.session_state.user_directive_for_current_segment_buffer = ""
            st.session_state.resume_choice_made = False 
            st.session_state.last_known_chapter = None # Reset resume state
            st.session_state.last_known_segment = None
            add_log("用户请求重新初始化系统。", "info", "UI")


# --- Main Writing Area ---
# Display this area ONLY if system initialization was ATTEMPTED
if st.session_state.system_initialized_attempted:
    if st.session_state.system_initialized_successfully: # And it was successful
        st.header("📝 小说创作区")

        # Resume logic UI part (only if resume data exists and choice not made)
        if st.session_state.get('last_known_chapter') is not None and \
           st.session_state.get('last_known_segment') is not None and \
           not st.session_state.resume_choice_made:
            
            resume_q_text = (f"检测到上次写作到 章节 {st.session_state.last_known_chapter}, "
                             f"片段 {st.session_state.last_known_segment}。如何继续？")
            resume_options = ["从上次继续", "从新章节开始 (下一章)", "从头开始 (章节1, 片段0)"]
            choice_idx = st.session_state.get("resume_choice_idx", 0)
            
            chosen_resume_option = st.radio(resume_q_text, resume_options, index=choice_idx, key="resume_radio_ui", horizontal=True)

            if st.button("✔️ 确认继续方式", key="confirm_resume_ui"):
                st.session_state.resume_choice_idx = resume_options.index(chosen_resume_option)
                if chosen_resume_option == "从上次继续":
                    st.session_state.current_chapter = int(st.session_state.last_known_chapter) # Ensure int
                    st.session_state.current_segment_number = int(st.session_state.last_known_segment) # Ensure int
                elif chosen_resume_option == "从新章节开始 (下一章)":
                    st.session_state.current_chapter = int(st.session_state.last_known_chapter) + 1
                    st.session_state.current_segment_number = 0
                else: # From scratch
                    st.session_state.current_chapter = 1
                    st.session_state.current_segment_number = 0
                add_log(f"写作状态已更新: 章节 {st.session_state.current_chapter}, 片段 {st.session_state.current_segment_number}", "info", "UI")
                st.session_state.resume_choice_made = True
        
        # --- CRITICAL FIX FOR TypeError ---
        # Ensure current_segment_number is an integer before using in f-string for display
        current_seg_num_for_display = st.session_state.get('current_segment_number', 0) # Default to 0
        if not isinstance(current_seg_num_for_display, int):
            add_log(f"警告: current_segment_number 类型 ({type(current_seg_num_for_display).__name__}) "
                    f"不是整数, 值: '{current_seg_num_for_display}'. 已重置为0.", "warning", "UI")
            current_seg_num_for_display = 0
            st.session_state.current_segment_number = 0 # Correct it in session state too

        st.info(f"当前写作进度：章节 {st.session_state.current_chapter}, "
                f"计划生成片段号 {current_seg_num_for_display + 1}") # This was approx line 190

        # Directive Input
        st.session_state.user_directive_for_current_segment = st.text_area(
            "请输入当前片段的写作指令/概要 (详细描述核心事件、爽点、钩子、情感线、篇幅引导等):",
            height=250,
            key="directive_input_main_ui_v2", # Changed key
            value=st.session_state.get("user_directive_for_current_segment_buffer", "") 
        )

        if st.button("✨ 生成故事片段", key="generate_segment_button_ui_v2", type="primary"):
            if not st.session_state.user_directive_for_current_segment.strip():
                st.warning("写作指令不能为空！")
            else:
                st.session_state.current_generated_text = "" 
                with st.spinner("AI 正在奋笔疾书...请耐心等待..."):
                    try:
                        generated_text = novel_core.core_generate_segment_text_for_ui(
                            st.session_state.user_directive_for_current_segment
                        )
                        st.session_state.current_generated_text = generated_text if generated_text else "AI未能生成有效内容。"
                        st.session_state.user_directive_for_current_segment_buffer = st.session_state.user_directive_for_current_segment
                        add_log("AI已生成新片段。", "info", "UI")
                    except Exception as e:
                        add_log(f"生成片段时发生错误: {e}", "error", "UI")
                        st.session_state.current_generated_text = f"生成失败: {str(e)[:300]}..."
        
        # Display and interact with generated text
        if st.session_state.current_generated_text:
            # ... (The rest of the UI for displaying text and action buttons: Adopt, Rewrite, Expand, Discard, Next)
            # This part should be largely the same as the previous app_ui.py version.
            # Ensure all calls to novel_core functions are correct and use st.session_state for params.
            st.subheader("🤖 AI 生成的片段")
            st.markdown(st.session_state.current_generated_text) 
            st.caption(f"当前片段字数: {len(st.session_state.current_generated_text)}")
            # ... (Action buttons columns and logic) ...

    elif st.session_state.system_initialized_attempted and not st.session_state.system_initialized_successfully: 
        # If init was ATTEMPTED but FAILED
        st.error("🤷 系统初始化失败，请检查侧边栏的日志并尝试在侧边栏重新初始化。")

# This is the default message if no init attempt has been made
elif not st.session_state.system_initialized_attempted: 
    st.warning("👈 请在侧边栏选择模型并点击“初始化系统”以开始使用。")


# --- Log Display Area (in sidebar) ---
st.sidebar.subheader("运行日志")
log_container = st.sidebar.container()
log_html = "<div style='max-height: 400px; overflow-y: auto; border: 1px solid #e6e6e6; padding: 5px;'>"
# Ensure log_messages is always a list
if not isinstance(st.session_state.get("log_messages"), list):
    st.session_state.log_messages = ["日志列表初始化错误。"]

for msg in st.session_state.log_messages: 
    log_html += f"<pre style='white-space: pre-wrap; word-wrap: break-word; margin-bottom: 2px; font-size: 0.8em;'>{st.html(msg)}</pre>"
log_html += "</div>"
log_container.markdown(log_html, unsafe_allow_html=True)

if st.sidebar.button("清除日志", key="clear_logs_ui"):
    st.session_state.log_messages = ["日志已清除。"]