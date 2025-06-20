# app_ui.py
# Python Standard Library Imports
import logging
import os

# Third-party Imports
import streamlit as st

try:
    import novel_core # Ensure this is at the top after Streamlit
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
        "current_segment_number": 0, 
        "current_generated_text": "", 
        "user_directive_for_current_segment": "",
        "selected_embedding_provider_key": list(novel_core.embedding_providers_map_core.keys())[1], # Default BGE
        "selected_llm_provider_key": list(novel_core.llm_providers_map_core.keys())[2], # Default Gemini
        "api_keys": {}, 
        "milvus_initialized_core": False, 
        "max_tokens_per_llm_call": int(os.getenv("MAX_TOKENS_PER_LLM_CALL", "7800")),
        "segments_per_chapter_advance": int(os.getenv("SEGMENTS_PER_CHAPTER_ADVANCE", "3")),
        "last_adopted_segment_text": "这是故事的开端，尚无先前的故事片段。",
        "user_directive_for_current_segment_buffer": "", 
        "show_expand_input": False, 
        "num_recent_segments_to_fetch_ui": 2, 
        "llm_temperature": 0.7, 
        "novel_md_output_dir_ui": novel_core.NOVEL_MD_OUTPUT_DIR_CORE,
        "last_known_chapter": None, 
        "last_known_segment": None, 
        "resume_choice_made": False,
        "resume_choice_idx": 0 
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

def add_log(message: str, level: str = "info", UImodule: str = "UI"):
    timestamp = f"[{UImodule}][{st.session_state.get('current_llm_provider', 'SYS')}] "
    log_entry = f"{timestamp}{message}"
    if not isinstance(st.session_state.get("log_messages"), list): st.session_state.log_messages = []
    st.session_state.log_messages.insert(0, log_entry)
    core_logger = logging.getLogger("NovelCore") 
    if level.lower() == "info": core_logger.info(message)
    elif level.lower() == "warning": core_logger.warning(message)
    elif level.lower() == "error": core_logger.error(message)
    elif level.lower() == "fatal": core_logger.critical(message) # Use critical for FATAL
    elif level.lower() == "debug": core_logger.debug(message)
    elif level.lower() == "success": core_logger.info(message) # INFO for success in console

# --- UI Layout ---
st.title("✍️ AI 小说写作助手")

with st.sidebar:
    st.header("系统配置")
    # ... (Embedding and LLM selectors, Advanced Config / API Keys from previous app_ui.py) ...
    # Initialization Button
    if not st.session_state.system_initialized_attempted:
        if st.button("🚀 初始化系统", key="init_button_main_ui_v4", type="primary", use_container_width=True):
            st.session_state.system_initialized_attempted = True
            with st.spinner("系统初始化中... (可能需要较长时间下载模型或连接服务)"):
                try:
                    novel_core.core_initialize_system(
                        st.session_state.selected_embedding_provider_key,
                        st.session_state.selected_llm_provider_key,
                        st.session_state.api_keys
                    )
                    # No need to set system_initialized_successfully here, core_initialize_system does that
                    # add_log("UI: 系统初始化流程已调用。", "info", "UI") # Logged by core more accurately
                except Exception as e_init_ui:
                    add_log(f"UI层面捕获到初始化失败: {e_init_ui}", "error", "UI")
                    # system_initialized_successfully will be False due to exception in core
            # Streamlit reruns automatically after button and state change
            
    elif st.session_state.system_initialized_successfully:
        # ... (Display "System Initialized!" message and Re-init button) ...
        pass
    # If attempted but not successful, main area will show the error.

# --- Main Writing Area ---
if st.session_state.system_initialized_attempted:
    if st.session_state.system_initialized_successfully:
        st.header("📝 小说创作区")
        # ... (Resume logic UI from previous app_ui.py) ...
        
        # --- CRITICAL FIX FOR TypeError ---
        current_seg_num_for_display = st.session_state.get('current_segment_number', 0)
        if not isinstance(current_seg_num_for_display, int):
            add_log(f"警告: current_segment_number ({type(current_seg_num_for_display).__name__}: '{current_seg_num_for_display}') 非整数. 重置为0.", "warning", "UI")
            current_seg_num_for_display = 0
            st.session_state.current_segment_number = 0
        current_chap_for_display = st.session_state.get('current_chapter', 1)
        if not isinstance(current_chap_for_display, int):
            add_log(f"警告: current_chapter ({type(current_chap_for_display).__name__}: '{current_chap_for_display}') 非整数. 重置为1.", "warning", "UI")
            current_chap_for_display = 1
            st.session_state.current_chapter = 1
        st.info(f"当前写作进度：章节 {current_chap_for_display}, 计划生成片段号 {current_seg_num_for_display + 1}")
        # ... (Rest of the main writing area: directive input, generate button, display, actions - from previous app_ui.py)
        # Ensure calls to novel_core functions (core_generate_segment_text_for_ui, core_adopt_segment_from_ui) are correct.
    else:
        st.error("🤷 系统初始化失败，请检查侧边栏的日志并尝试在侧边栏重新初始化。")
elif not st.session_state.system_initialized_attempted:
    st.warning("👈 请在侧边栏选择模型并点击“初始化系统”以开始使用。")

# --- Log Display Area (in sidebar) ---
# ... (Log display code from previous app_ui.py) ...