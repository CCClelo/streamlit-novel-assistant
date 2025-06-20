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
# ... (same as before) ...
st.set_page_config(page_title="AI小说写作助手", page_icon="✍️", layout="wide", initial_sidebar_state="expanded")

# --- Logger for UI ---
# ... (same as before, ensure add_log is defined) ...
def add_log(message, level="info", UImodule="UI"): # Ensure this is defined
    timestamp = f"[{UImodule}][{st.session_state.get('current_llm_provider', 'SYS')}] "
    log_entry = f"{timestamp}{message}"
    st.session_state.log_messages.insert(0, log_entry)
    core_logger = logging.getLogger("NovelCore")
    if level == "info": core_logger.info(message)
    elif level == "warning": core_logger.warning(message)
    elif level == "error": core_logger.error(message)


# --- Initialize Session State ---
# ... (initialize_session_state function remains largely the same, ensure current_segment_number defaults to 0) ...
def initialize_session_state():
    defaults = {
        "system_initialized": False, # Tracks if init button was clicked
        "system_initialized_successfully": False, # Tracks if core_initialize_system succeeded
        "log_messages": ["应用已启动，等待初始化..."],
        "current_chapter": 1,
        "current_segment_number": 0, # Crucial: ensure this is an int
        # ... (all other session state variables from previous version) ...
        "resume_choice_made": False, # For resume logic
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()


# --- UI Layout ---
st.title("✍️ AI 小说写作助手")

# --- Sidebar ---
# ... (Sidebar code remains the same, ensure init button sets system_initialized only,
#     and core_initialize_system sets system_initialized_successfully) ...
with st.sidebar:
    st.header("系统配置")
    # ... (Embedding and LLM selectors) ...
    # ... (Advanced Config / API Keys) ...

    if not st.session_state.system_initialized: # Show init button if not even attempted
        if st.button("🚀 初始化系统", key="init_button_main_ui", type="primary", use_container_width=True):
            st.session_state.system_initialized = True # Mark that init process has started
            with st.spinner("系统初始化中，请稍候..."):
                try:
                    # Call core_initialize_system which now sets system_initialized_successfully
                    novel_core.core_initialize_system(
                        st.session_state.selected_embedding_provider_key,
                        st.session_state.selected_llm_provider_key,
                        st.session_state.api_keys 
                    )
                    # No need to set system_initialized_successfully here, core does it.
                    add_log("系统初始化成功！ (UI触发)", "info", "UI") # Logged by core too
                    # st.success("系统初始化成功！") # Let main area show status
                except Exception as e:
                    # core_initialize_system already logs and sets system_initialized_successfully=False
                    add_log(f"UI捕获到初始化失败: {e}", "error", "UI")
                    # st.error(f"初始化失败: {str(e)[:300]}...") # Let main area show status
    elif st.session_state.system_initialized_successfully: # If init was attempted and succeeded
        st.success(f"系统已初始化！LLM: {novel_core.llm_providers_map_core.get(st.session_state.selected_llm_provider_key, '未知').upper()}, "
                   f"Embedding: {novel_core.embedding_providers_map_core.get(st.session_state.selected_embedding_provider_key, ['未知'])[0].upper()}")
        if st.button("🔄 重新初始化/切换模型", key="reinit_button_ui", use_container_width=True):
            st.session_state.system_initialized = False # Allow re-init attempt
            st.session_state.system_initialized_successfully = False
            st.session_state.milvus_initialized = False 
            st.session_state.current_chapter = 1 
            st.session_state.current_segment_number = 0
            st.session_state.current_generated_text = ""
            st.session_state.user_directive_for_current_segment_buffer = ""
            st.session_state.resume_choice_made = False # Allow resume choice again
            add_log("用户请求重新初始化系统。", "info", "UI")
    # If system_initialized is True but system_initialized_successfully is False, main area will show error.


# --- Main Writing Area ---
if st.session_state.system_initialized: # If initialization process has been started/attempted
    if st.session_state.system_initialized_successfully: # And it was successful
        st.header("📝 小说创作区")

        # Resume logic UI part
        if st.session_state.get('last_known_chapter') is not None and \
           st.session_state.get('last_known_segment') is not None and \
           not st.session_state.resume_choice_made:
            
            resume_q_text = (f"检测到上次写作到 章节 {st.session_state.last_known_chapter}, "
                             f"片段 {st.session_state.last_known_segment}。如何继续？")
            resume_options = ["从上次继续", "从新章节开始 (下一章)", "从头开始 (章节1, 片段0)"]
            choice_idx = st.session_state.get("resume_choice_idx", 0) # Persist radio choice
            
            # This needs to be outside columns for radio to work well with button
            chosen_resume_option = st.radio(resume_q_text, resume_options, index=choice_idx, key="resume_radio_ui", horizontal=True)

            if st.button("✔️ 确认继续方式", key="confirm_resume_ui"):
                st.session_state.resume_choice_idx = resume_options.index(chosen_resume_option)
                if chosen_resume_option == "从上次继续":
                    st.session_state.current_chapter = st.session_state.last_known_chapter
                    st.session_state.current_segment_number = st.session_state.last_known_segment # Should be int
                elif chosen_resume_option == "从新章节开始 (下一章)":
                    st.session_state.current_chapter = st.session_state.last_known_chapter + 1
                    st.session_state.current_segment_number = 0
                else: # From scratch
                    st.session_state.current_chapter = 1
                    st.session_state.current_segment_number = 0
                add_log(f"写作状态已更新: 章节 {st.session_state.current_chapter}, 片段 {st.session_state.current_segment_number}", "info", "UI")
                st.session_state.resume_choice_made = True
        
        # Ensure current_segment_number is an integer before using in f-string for display
        # This is a critical fix for the TypeError
        current_seg_num_for_display = st.session_state.get('current_segment_number', 0)
        if not isinstance(current_seg_num_for_display, int):
            add_log(f"警告: current_segment_number 类型 ({type(current_seg_num_for_display).__name__}) 不是整数, 值: '{current_seg_num_for_display}'. 重置为0.", "warning", "UI")
            current_seg_num_for_display = 0
            st.session_state.current_segment_number = 0 # Correct it in session state

        st.info(f"当前写作进度：章节 {st.session_state.current_chapter}, "
                f"计划生成片段号 {current_seg_num_for_display + 1}") # Line 190 was here

        # ... (rest of the main writing area UI: directive input, generate button, display, actions - from previous app_ui.py) ...
        # Ensure calls to novel_core functions are correct.
        # For example, the generate button's action:
        # generated_text = novel_core.core_generate_segment_text_for_ui(...)
        # And adopt button's action:
        # success = novel_core.core_adopt_segment_from_ui(...)
        # The structure of this part should remain the same as the previous app_ui.py version.
        # Pasting the relevant block from the previous app_ui.py with the TypeError fix point:
        st.session_state.user_directive_for_current_segment = st.text_area(
            "请输入当前片段的写作指令/概要 (详细描述核心事件、爽点、钩子、情感线、篇幅引导等):",
            height=250, key="directive_input_main_ui",
            value=st.session_state.get("user_directive_for_current_segment_buffer", "") 
        )
        if st.button("✨ 生成故事片段", key="generate_segment_button_ui", type="primary"):
            # ... (generate logic as before, calling novel_core.core_generate_segment_text_for_ui) ...
            pass 
        if st.session_state.current_generated_text:
            # ... (display and action buttons as before) ...
            pass

    else: # system_initialized is True, but system_initialized_successfully is False
        st.error("🤷 系统初始化失败，请检查侧边栏的日志并尝试重新初始化。")
elif not st.session_state.system_initialized : # system_initialized is False (init button not clicked yet)
    st.warning("👈 请在侧边栏选择模型并点击“初始化系统”以开始使用。")


# --- Log Display Area (remains the same) ---
# ... (your log display code in the sidebar) ...
st.sidebar.subheader("运行日志")
log_container = st.sidebar.container()
log_html = "<div style='max-height: 400px; overflow-y: auto; border: 1px solid #e6e6e6; padding: 5px;'>"
for msg in st.session_state.log_messages: 
    log_html += f"<pre style='white-space: pre-wrap; word-wrap: break-word; margin-bottom: 2px; font-size: 0.8em;'>{st.html(msg)}</pre>"
log_html += "</div>"
log_container.markdown(log_html, unsafe_allow_html=True)
if st.sidebar.button("清除日志", key="clear_logs_ui"):
    st.session_state.log_messages = ["日志已清除。"]