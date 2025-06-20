# app_ui.py
# Python Standard Library Imports
import logging
import os

# Third-party Imports
import streamlit as st

try:
    import novel_core 
except ImportError:
    st.error("CRITICAL ERROR: 无法导入 novel_core.py。请确保该文件与 app_ui.py 在同一目录下，"
             "并且 novel_core.py 本身没有导致导入失败的严重错误。请检查部署日志获取详细信息。")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="AI小说写作助手", page_icon="✍️", layout="wide", initial_sidebar_state="expanded")

# --- Initialize Session State ---
def initialize_session_state():
    defaults = {
        "system_initialized_attempted": False,
        "system_initialized_successfully": False,
        "log_messages": ["应用已启动，等待初始化..."],
        "current_chapter": 1,
        "current_segment_number": 0, # Default to int
        "current_generated_text": "", 
        "user_directive_for_current_segment": "",
        # Default selections using keys from novel_core's maps
        "selected_embedding_provider_key": list(novel_core.embedding_providers_map_core.keys())[1], # Default to BGE
        "selected_llm_provider_key": list(novel_core.llm_providers_map_core.keys())[2], # Default to Gemini
        "api_keys": {}, 
        "milvus_initialized_core": False, 
        "max_tokens_per_llm_call": int(os.getenv("MAX_TOKENS_PER_LLM_CALL", "7800")),
        "segments_per_chapter_advance": int(os.getenv("SEGMENTS_PER_CHAPTER_ADVANCE", "3")),
        "last_adopted_segment_text": "这是故事的开端，尚无先前的故事片段。",
        "user_directive_for_current_segment_buffer": "", 
        "show_expand_input": False, 
        "num_recent_segments_to_fetch_ui": 2, 
        "llm_temperature": 0.7, 
        "novel_md_output_dir_ui": novel_core.NOVEL_MD_OUTPUT_DIR_CORE, # Get default from core
        "last_known_chapter": None, 
        "last_known_segment": None, 
        "resume_choice_made": False,
        "resume_choice_idx": 0 # To store radio button index
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# --- Helper function to add logs ---
def add_log(message: str, level: str = "info", UImodule: str = "UI"):
    # This function should now be part of app_ui.py as it directly interacts with st.session_state.log_messages
    # novel_core can have its own logger for console/file logging.
    timestamp = f"[{UImodule}][{st.session_state.get('current_llm_provider', 'SYS')}] "
    log_entry = f"{timestamp}{message}"
    if not isinstance(st.session_state.get("log_messages"), list): st.session_state.log_messages = []
    st.session_state.log_messages.insert(0, log_entry)
    # Console logging (optional, if you want UI actions logged to console too)
    # console_logger = logging.getLogger("NovelAppUI") # Separate logger for UI if needed
    # if level == "info": console_logger.info(message) ...


# --- UI Layout ---
st.title("✍️ AI 小说写作助手")

with st.sidebar:
    # ... (Sidebar content: selectors, API keys, init button - REMAINS LARGELY THE SAME) ...
    # Ensure the init button sets system_initialized_attempted = True
    # and calls novel_core.core_initialize_system.
    # The core_initialize_system will set system_initialized_successfully.
    st.header("系统配置")
    # ... (Selectors and advanced config as in previous complete app_ui.py) ...
    if not st.session_state.system_initialized_attempted:
        if st.button("🚀 初始化系统", key="init_button_main_ui_v3", type="primary", use_container_width=True):
            st.session_state.system_initialized_attempted = True
            with st.spinner("系统初始化中..."):
                try:
                    novel_core.core_initialize_system(
                        st.session_state.selected_embedding_provider_key,
                        st.session_state.selected_llm_provider_key,
                        st.session_state.api_keys
                    )
                    # core_initialize_system sets 'system_initialized_successfully'
                except Exception as e_init_ui:
                    # This catch is for errors raised by core_initialize_system itself
                    add_log(f"UI层面捕获到初始化失败: {e_init_ui}", "error", "UI")
                    # st.session_state.system_initialized_successfully is already False or set by core
    elif st.session_state.system_initialized_successfully:
        # ... (Display "System Initialized!" message) ...
        pass # Status will be shown in main area
    # ... (Re-init button)


# --- Main Writing Area ---
if st.session_state.system_initialized_attempted:
    if st.session_state.system_initialized_successfully:
        st.header("📝 小说创作区")

        # Resume logic UI part
        if st.session_state.get('last_known_chapter') is not None and \
           st.session_state.get('last_known_segment') is not None and \
           not st.session_state.resume_choice_made:
            # ... (Radio button and confirm button for resume choice - from previous version) ...
            # Ensure this logic correctly updates st.session_state.current_segment_number as int
            pass # Placeholder for brevity, assume it's implemented

        # --- CRITICAL FIX FOR TypeError ---
        # Ensure current_segment_number is an integer for display and calculations
        current_seg_num_for_display = st.session_state.get('current_segment_number', 0)
        if not isinstance(current_seg_num_for_display, int):
            add_log(f"警告: current_segment_number 类型 ({type(current_seg_num_for_display).__name__}) "
                    f"不是整数, 值: '{current_seg_num_for_display}'. 已重置为0.", "warning", "UI")
            current_seg_num_for_display = 0
            st.session_state.current_segment_number = 0 # Correct it in session state
        
        # Ensure current_chapter is also an int
        current_chap_for_display = st.session_state.get('current_chapter', 1)
        if not isinstance(current_chap_for_display, int):
            add_log(f"警告: current_chapter 类型 ({type(current_chap_for_display).__name__}) "
                    f"不是整数, 值: '{current_chap_for_display}'. 已重置为1.", "warning", "UI")
            current_chap_for_display = 1
            st.session_state.current_chapter = 1


        st.info(f"当前写作进度：章节 {current_chap_for_display}, "
                f"计划生成片段号 {current_seg_num_for_display + 1}")

        # ... (Rest of the main writing area: directive input, generate button, display, actions) ...
        # Ensure all calls to novel_core functions are correct.
        # Example:
        # st.session_state.user_directive_for_current_segment = st.text_area(...)
        # if st.button("✨ 生成故事片段", ...):
        #     generated_text = novel_core.core_generate_segment_text_for_ui(...)
        # if st.session_state.current_generated_text:
        #     if st.button("👍 采纳", ...):
        #         novel_core.core_adopt_segment_from_ui(...)

    else: # system_initialized_attempted is True, but system_initialized_successfully is False
        st.error("🤷 系统初始化失败，请检查侧边栏的日志并尝试在侧边栏重新初始化。")
elif not st.session_state.system_initialized_attempted:
    st.warning("👈 请在侧边栏选择模型并点击“初始化系统”以开始使用。")

# --- Log Display Area (in sidebar) ---
# ... (Your log display code - from previous version) ...
st.sidebar.subheader("运行日志")
# ... (rest of log display)