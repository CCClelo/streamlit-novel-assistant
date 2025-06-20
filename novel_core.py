# novel_core.py
from typing import Dict, Optional, List 

import streamlit as st 
import logging
import os
import uuid
from datetime import datetime
import time 

try:
    import openai
    from sentence_transformers import SentenceTransformer
    from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection
    import httpx
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    from dotenv import load_dotenv
except ImportError as e:
    # Basic logging for pre-streamlit errors
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("NovelCorePreImport").error(f"Core: Missing critical libraries: {e}. Please install them.")
    raise

# --- Global Configurations (load once) ---
load_dotenv()
logger = logging.getLogger("NovelCore")
if not logger.hasHandlers(): # Ensure logger is configured once
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants Definition ---
# API Key Environment Variable Names
OPENAI_API_KEY_ENV_NAME = "OPENAI_API_KEY"
DEEPSEEK_API_KEY_ENV_NAME = "DEEPSEEK_API_KEY"
GEMINI_API_KEY_ENV_NAME = "GEMINI_API_KEY"
CUSTOM_PROXY_API_KEY_ENV_NAME = "CUSTOM_PROXY_API_KEY"

# Proxy URLs
OPENAI_OFFICIAL_HTTP_PROXY_CORE = os.getenv("OPENAI_OFFICIAL_HTTP_PROXY")
OPENAI_OFFICIAL_HTTPS_PROXY_CORE = os.getenv("OPENAI_OFFICIAL_HTTPS_PROXY")
DEEPSEEK_LLM_HTTP_PROXY_CORE = os.getenv("DEEPSEEK_LLM_HTTP_PROXY")
DEEPSEEK_LLM_HTTPS_PROXY_CORE = os.getenv("DEEPSEEK_LLM_HTTPS_PROXY")
CUSTOM_LLM_HTTP_PROXY_CORE = os.getenv("CUSTOM_LLM_HTTP_PROXY")
CUSTOM_LLM_HTTPS_PROXY_CORE = os.getenv("CUSTOM_LLM_HTTPS_PROXY")
GEMINI_HTTP_PROXY_CORE = os.getenv("GEMINI_HTTP_PROXY", os.getenv("GLOBAL_HTTP_PROXY"))
GEMINI_HTTPS_PROXY_CORE = os.getenv("GEMINI_HTTPS_PROXY", os.getenv("GLOBAL_HTTPS_PROXY"))

# Custom Proxy Configuration
CUSTOM_PROXY_BASE_URL_CORE = os.getenv("CUSTOM_PROXY_BASE_URL", "https://api.openai-next.com/v1")
HARDCODED_CUSTOM_PROXY_KEY_CORE = "sk-0rRacqNU0aY2DrOV1601DfE76a4d4a87A14f97383e25095b"

# Milvus Configuration
MILVUS_ALIAS_CORE = "default"
MILVUS_HOST_CORE = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT_CORE = os.getenv("MILVUS_PORT", "19530")
ZILLIZ_CLOUD_URI_ENV_NAME = "ZILLIZ_CLOUD_URI"
ZILLIZ_CLOUD_TOKEN_ENV_NAME = "ZILLIZ_CLOUD_TOKEN"

# Embedding Models
OPENAI_EMBEDDING_MODEL_CORE = "text-embedding-3-small"
ST_MODEL_TEXT2VEC_CORE = "shibing624/text2vec-base-chinese"
ST_MODEL_BGE_LARGE_ZH_CORE = "BAAI/bge-large-zh-v1.5"

# LLM Models
OPENAI_LLM_MODEL_CORE = "gpt-3.5-turbo"
DEEPSEEK_LLM_MODEL_CORE = "deepseek-chat"
GEMINI_LLM_MODEL_CORE = "gemini-1.5-flash-latest"
CUSTOM_PROXY_LLM_MODEL_CORE = "gpt-3.5-turbo"

# Other Paths and Prefixes
DEEPSEEK_BASE_URL_CORE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
SETTINGS_FILES_DIR_CORE = "./novel_setting_files"
NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_chapters_core" # Core's default
COLLECTION_NAME_LORE_PREFIX_CORE = "novel_lore_mv"
COLLECTION_NAME_STORY_PREFIX_CORE = "novel_story_mv"

# Provider Maps - CRITICAL: Define these before any function that might use them, like core_initialize_system
embedding_providers_map_core = {
    "1": ("sentence_transformer_text2vec", ST_MODEL_TEXT2VEC_CORE, "(本地ST Text2Vec中文)", 768),
    "2": ("sentence_transformer_bge_large_zh", ST_MODEL_BGE_LARGE_ZH_CORE, "(本地ST BGE中文Large)", 1024),
    "3": ("openai_official", OPENAI_EMBEDDING_MODEL_CORE, "(官方OpenAI API Key)", 1536)
}
llm_providers_map_core = { # <--- This was the problematic constant
    "1": "openai_official", 
    "2": "deepseek", 
    "3": "gemini", 
    "4": "custom_proxy_llm"
}

# --- Helper Global Variable for Proxies ---
_core_original_os_environ_proxies: Dict[str, Optional[str]] = {}

# --- Helper Functions ---
# (core_get_custom_proxy_key, core_set_temp_os_proxies, core_restore_original_os_proxies,
#  core_get_httpx_client_with_proxy - These are assumed to be correctly implemented from before
#  and use the constants defined above.)

# --- Embedding Functions ---
# (core_get_openai_embeddings, core_get_st_embeddings - Full implementations
#  using constants like OPENAI_EMBEDDING_MODEL_CORE and st.session_state correctly.)

# --- Milvus Related Functions ---
# (core_chunk_text_by_paragraph, core_load_and_vectorize_settings, 
#  core_init_milvus_collections_internal, core_seed_initial_lore, 
#  core_add_story_segment_to_milvus, core_retrieve_relevant_lore, 
#  core_retrieve_recent_story_segments - Ensure these are defined in correct order
#  if they call each other, and that they use constants and session_state correctly.
#  core_init_milvus_collections_internal is a key one that sets up Milvus objects
#  and must be defined before core_initialize_system if it's called directly, or ensure
#  its internal helpers are defined before use.)

# --- LLM Generation Function ---
# (core_generate_with_llm - Full implementation, using LLM model constants, API keys from session_state,
#  proxy constants, and helper functions like core_get_custom_proxy_key.)

# --- Main Initialization Function (called by UI) ---
# THIS FUNCTION MUST BE DEFINED AFTER ALL THE CONSTANTS AND HELPER FUNCTIONS IT USES.
# PARTICULARLY embedding_providers_map_core and llm_providers_map_core.
def core_initialize_system(embedding_choice_key: str, llm_choice_key: str, api_keys_from_ui: dict):
    # Log to st.session_state.log_messages for UI visibility
    log_to_ui = lambda msg, lvl="INFO": st.session_state.log_messages.insert(0, f"[{lvl}][Core] {msg}") if hasattr(st, 'session_state') and 'log_messages' in st.session_state else None

    try:
        log_to_ui("开始核心系统初始化...")
        logger.info(f"Core: Initializing system with embedding_choice_key='{embedding_choice_key}', llm_choice_key='{llm_choice_key}'")
        st.session_state.system_initialized_successfully = False 

        st.session_state.selected_embedding_provider_key = embedding_choice_key
        st.session_state.selected_llm_provider_key = llm_choice_key
        st.session_state.api_keys = api_keys_from_ui

        # --- Embedding Dimension Setup ---
        if not embedding_choice_key or embedding_choice_key not in embedding_providers_map_core: # Uses map
            err_msg = f"无效的 embedding_choice_key: '{embedding_choice_key}'。"
            raise ValueError(err_msg)
        
        emb_data = embedding_providers_map_core[embedding_choice_key] # Uses map
        if len(emb_data) < 4 or not isinstance(emb_data[3], int) or emb_data[3] <= 0:
            err_msg = f"embedding_providers_map_core 中key '{embedding_choice_key}' 的维度信息无效。"
            raise ValueError(err_msg)
        emb_identifier, emb_model_name, _, emb_dim = emb_data
        
        st.session_state.selected_embedding_provider_identifier = emb_identifier
        st.session_state.embedding_dimension = emb_dim 
        st.session_state.selected_st_model_name = emb_model_name if "sentence_transformer" in emb_identifier else None
        log_to_ui(f"嵌入配置: ID='{emb_identifier}', Model='{emb_model_name}', Dim='{emb_dim}'")
        
        if "sentence_transformer" in emb_identifier:
            # ... (ST model loading logic into st.session_state.embedding_model_instance) ...
            logger.info(f"Core: ST Model '{emb_model_name}' setup (actual loading needed).")
        elif emb_identifier == "openai_official":
            st.session_state.embedding_model_instance = None 
            st.session_state.loaded_st_model_name = None

        # THIS IS WHERE llm_providers_map_core IS USED (Example line 154 from your error)
        llm_provider_name = llm_providers_map_core[llm_choice_key] 
        st.session_state.current_llm_provider = llm_provider_name
        log_to_ui(f"选择LLM模型: {llm_provider_name.upper()}。")
        
        # TODO: Implement your full API key validation logic here
        logger.info("Core: API Key checks (full implementation needed).")
        log_to_ui("API Key检查 (需完整实现)。")

        # core_init_milvus_collections_internal() # TODO: Ensure this is fully defined and works
        # core_seed_initial_lore()                # TODO: Ensure this is fully defined and works
        logger.info("Core: Milvus init and seeding (actual implementation needed).") # Placeholder
        st.session_state.milvus_initialized_core = True # Mock
        st.session_state.lore_collection_milvus_obj = "MockCollection" # Mock
        st.session_state.story_collection_milvus_obj = "MockCollection" # Mock


        st.session_state.current_chapter = 1 
        st.session_state.current_segment_number = 0 
        # ... (Story resume logic) ...
        log_to_ui("故事写作状态已初始化/重置。")
        
        st.session_state.system_initialized_successfully = True
        logger.info("Core: System initialization successful.")
        log_to_ui("核心系统初始化流程完成！", "SUCCESS")
        return True

    except Exception as e:
        st.session_state.system_initialized_successfully = False
        st.session_state.current_chapter = st.session_state.get('current_chapter', 1) 
        if not isinstance(st.session_state.current_chapter, int): st.session_state.current_chapter = 1
        st.session_state.current_segment_number = st.session_state.get('current_segment_number', 0)
        if not isinstance(st.session_state.current_segment_number, int): st.session_state.current_segment_number = 0
        logger.error(f"Core: System initialization failed: {e}", exc_info=True)
        log_to_ui(f"核心系统初始化失败: {e}", "FATAL")
        raise

# --- UI Specific Core Functions ---
# (core_generate_segment_text_for_ui, core_adopt_segment_from_ui)
# These must be defined and use the constants and session_state variables correctly.
# They will call other core functions like core_generate_with_llm, etc.
# TODO: Paste your full implementations for these UI-facing core functions.

# --- Ensure ALL other functions (embedding, Milvus ops, LLM call, etc.) are ---
# --- fully defined above the functions that call them.                     ---
# --- For brevity, I'm assuming you have the full bodies of those from previous versions. ---