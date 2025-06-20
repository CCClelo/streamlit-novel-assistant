# novel_core.py
from typing import Dict, Optional, List

# Standard library imports
import logging
import os
import uuid
from datetime import datetime
import time

_STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    class MockSessionState:
        def __init__(self): self._state = {}
        def get(self, key, default=None): return self._state.get(key, default)
        def __getitem__(self, key): return self._state[key]
        def __setitem__(self, key, value): self._state[key] = value
        def __contains__(self, key): return key in self._state
        def insert(self, index, value): 
            if 'log_messages' not in self._state: self._state['log_messages'] = []
            self._state['log_messages'].insert(index, value)
    if not globals().get('st'): 
        st_instance = MockSessionState()
        st = type('StreamlitMock', (), {'session_state': st_instance})()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("NovelCorePreImport").warning("Streamlit not found, using a mock session_state.")


try:
    import openai
    from sentence_transformers import SentenceTransformer
    from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection
    import httpx
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    from dotenv import load_dotenv
except ImportError as e:
    error_msg = f"Core: Missing critical libraries: {e}. Please install them."
    if _STREAMLIT_AVAILABLE: st.error(error_msg)
    else: logging.getLogger("NovelCorePreImport").error(error_msg)
    raise

# --- Global Configurations ---
load_dotenv()
logger = logging.getLogger("NovelCore")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants Definition ---
# (All constants as defined previously, ensure they are complete and correct)
OPENAI_API_KEY_ENV_NAME = "OPENAI_API_KEY"; DEEPSEEK_API_KEY_ENV_NAME = "DEEPSEEK_API_KEY"; GEMINI_API_KEY_ENV_NAME = "GEMINI_API_KEY"; CUSTOM_PROXY_API_KEY_ENV_NAME = "CUSTOM_PROXY_API_KEY"
OPENAI_OFFICIAL_HTTP_PROXY_CORE = os.getenv("OPENAI_OFFICIAL_HTTP_PROXY"); OPENAI_OFFICIAL_HTTPS_PROXY_CORE = os.getenv("OPENAI_OFFICIAL_HTTPS_PROXY")
# ... (all other proxy constants) ...
CUSTOM_PROXY_BASE_URL_CORE = os.getenv("CUSTOM_PROXY_BASE_URL", "https://api.openai-next.com/v1"); HARDCODED_CUSTOM_PROXY_KEY_CORE = "sk-0rRacqNU0aY2DrOV1601DfE76a4d4a87A14f97383e25095b"
MILVUS_ALIAS_CORE = "default"; MILVUS_HOST_CORE = os.getenv("MILVUS_HOST", "localhost"); MILVUS_PORT_CORE = os.getenv("MILVUS_PORT", "19530")
ZILLIZ_CLOUD_URI_ENV_NAME = "ZILLIZ_CLOUD_URI"; ZILLIZ_CLOUD_TOKEN_ENV_NAME = "ZILLIZ_CLOUD_TOKEN"
OPENAI_EMBEDDING_MODEL_CORE = "text-embedding-3-small"; ST_MODEL_TEXT2VEC_CORE = "shibing624/text2vec-base-chinese"; ST_MODEL_BGE_LARGE_ZH_CORE = "BAAI/bge-large-zh-v1.5"
OPENAI_LLM_MODEL_CORE = "gpt-3.5-turbo"; DEEPSEEK_LLM_MODEL_CORE = "deepseek-chat"; GEMINI_LLM_MODEL_CORE = "gemini-1.5-flash-latest" ; CUSTOM_PROXY_LLM_MODEL_CORE = "gpt-3.5-turbo"
DEEPSEEK_BASE_URL_CORE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
SETTINGS_FILES_DIR_CORE = "./novel_setting_files"; NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_chapters_core"
COLLECTION_NAME_LORE_PREFIX_CORE = "novel_lore_mv" ; COLLECTION_NAME_STORY_PREFIX_CORE = "novel_story_mv"

embedding_providers_map_core = { # Ensure dimensions are correct
    "1": ("sentence_transformer_text2vec", ST_MODEL_TEXT2VEC_CORE, "(本地ST Text2Vec中文)", 768),
    "2": ("sentence_transformer_bge_large_zh", ST_MODEL_BGE_LARGE_ZH_CORE, "(本地ST BGE中文Large)", 1024),
    "3": ("openai_official", OPENAI_EMBEDDING_MODEL_CORE, "(官方OpenAI API Key)", 1536)
}
llm_providers_map_core = {
    "1": "openai_official", "2": "deepseek", "3": "gemini", "4": "custom_proxy_llm"
}

_core_original_os_environ_proxies: Dict[str, Optional[str]] = {}

# --- Helper Functions ---
# TODO: PASTE YOUR FULL IMPLEMENTATIONS for core_get_custom_proxy_key, core_set_temp_os_proxies, 
# TODO: core_restore_original_os_proxies, core_get_httpx_client_with_proxy

# --- Embedding Functions ---
# TODO: PASTE YOUR FULL IMPLEMENTATIONS for core_get_openai_embeddings, core_get_st_embeddings

# --- Milvus Related Functions (Ensure definition order if they call each other) ---
def core_chunk_text_by_paragraph(text: str) -> List[str]:
    # TODO: Implement actual logic
    logger.debug(f"Chunking text of length {len(text)}")
    return [p.strip() for p in text.replace('\r\n', '\n').replace('\r', '\n').split('\n\n') if p.strip()]

def core_load_and_vectorize_settings():
    # TODO: Implement actual logic (file reading, chunking, embedding, Milvus insertion)
    # This function MUST use st.session_state.lore_collection_milvus_obj (which should be a Collection object)
    # and check lore_collection.num_entities
    logger.info("Core: core_load_and_vectorize_settings called (actual implementation needed).")
    if not st.session_state.get('milvus_initialized_core') or \
       not isinstance(st.session_state.get('lore_collection_milvus_obj'), Collection):
        logger.error("Milvus lore collection not properly initialized for core_load_and_vectorize_settings.")
        return
    # ... your logic ...

def core_seed_initial_lore():
    core_load_and_vectorize_settings() # Call before
    # TODO: Implement actual fallback lore logic if collection is empty
    # This function MUST use st.session_state.lore_collection_milvus_obj
    logger.info("Core: core_seed_initial_lore called (actual implementation needed).")
    lore_collection = st.session_state.get('lore_collection_milvus_obj')
    if isinstance(lore_collection, Collection) and lore_collection.num_entities == 0:
        logger.info(f"Lore collection '{lore_collection.name}' empty, seeding fallback.")
        # ...
    elif not isinstance(lore_collection, Collection):
         logger.error("Cannot seed lore, lore_collection_milvus_obj is not a Collection.")


def core_init_milvus_collections_internal():
    # --- CRITICAL: ENSURE THIS FUNCTION IS FULLY AND CORRECTLY IMPLEMENTED ---
    # --- AS PER OUR PREVIOUS DISCUSSIONS (ZILLIZ-AWARE, RETURNS ACTUAL COLLECTION OBJECTS) ---
    logger.info("Core: core_init_milvus_collections_internal called.")
    if not st.session_state.get('embedding_dimension') or not isinstance(st.session_state.embedding_dimension, int) or st.session_state.embedding_dimension <= 0:
        err_msg = f"Core: Invalid or missing embedding_dimension in session_state: {st.session_state.get('embedding_dimension')}"
        logger.error(err_msg)
        if _STREAMLIT_AVAILABLE: st.session_state.log_messages.insert(0, f"[FATAL][Core] {err_msg}")
        raise ValueError(err_msg)
    
    # ... (Full Zilliz-aware connection logic from previous response) ...
    # ... (Full _create_or_get_collection helper that returns Collection objects) ...
    # ... (Full logic to set st.session_state.lore_collection_milvus_obj and story_collection_milvus_obj to these Collection objects) ...
    # For this to pass the error point, these must be actual Collection objects:
    # TODO: PASTE YOUR FULL MILVUS INITIALIZATION LOGIC HERE. THIS IS A MOCK:
    st.session_state.lore_collection_milvus_obj = Collection("mock_lore_collection_name_placeholder_needs_schema") # Placeholder
    st.session_state.story_collection_milvus_obj = Collection("mock_story_collection_name_placeholder_needs_schema") # Placeholder
    st.session_state.lore_collection_name = "mock_lore_name" # Set by actual init
    st.session_state.story_collection_name = "mock_story_name" # Set by actual init
    st.session_state.milvus_initialized_core = True
    logger.info("Core: Milvus collections set (MOCK IMPLEMENTATION - REPLACE WITH ACTUAL).")
    if _STREAMLIT_AVAILABLE: st.session_state.log_messages.insert(0, "[INFO][Core] Milvus (mock) 初始化完成。")


# ... (core_add_story_segment_to_milvus, core_retrieve_relevant_lore, core_retrieve_recent_story_segments -
#      FULL IMPLEMENTATIONS needed, using session_state for Milvus objects and checking their type)

# --- LLM Generation Function ---
def core_generate_with_llm(provider_name: str, prompt_text_from_rag: str, temperature: float =0.7, max_tokens_override: Optional[int]=None, system_message_override: Optional[str]=None):
    # TODO: PASTE YOUR FULL, WORKING LLM GENERATION LOGIC HERE
    # This is a large function and needs to be complete.
    # Ensure it uses constants like OPENAI_LLM_MODEL_CORE, GEMINI_LLM_MODEL_CORE, etc.
    # and handles proxies and API keys from session_state / os.getenv.
    logger.info(f"Core: LLM call to {provider_name} (actual implementation needed).")
    return f"[SIMULATED LLM OUTPUT for {provider_name}] Based on: {prompt_text_from_rag[:70]}..."


# --- Main Initialization Function (called by UI) ---
def core_initialize_system(embedding_choice_key: str, llm_choice_key: str, api_keys_from_ui: dict):
    log_to_ui = lambda msg, lvl="INFO": st.session_state.log_messages.insert(0, f"[{lvl.upper()}][Core] {msg}") if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state') and 'log_messages' in st.session_state else None
    
    try:
        log_to_ui("开始核心系统初始化...")
        logger.info(f"Core: Initializing system: Embedding Key='{embedding_choice_key}', LLM Key='{llm_choice_key}'")
        st.session_state.system_initialized_successfully = False # Default for this attempt

        st.session_state.selected_embedding_provider_key = embedding_choice_key
        st.session_state.selected_llm_provider_key = llm_choice_key
        st.session_state.api_keys = api_keys_from_ui

        # --- Embedding Dimension Setup (CRITICAL FIX POINT) ---
        if not embedding_choice_key or embedding_choice_key not in embedding_providers_map_core:
            err_msg = f"无效的 embedding_choice_key: '{embedding_choice_key}'. 可用: {list(embedding_providers_map_core.keys())}"
            logger.error(err_msg)
            log_to_ui(err_msg, "FATAL")
            raise ValueError(err_msg)
        
        emb_data = embedding_providers_map_core[embedding_choice_key]
        if len(emb_data) < 4 or not isinstance(emb_data[3], int) or emb_data[3] <= 0:
            err_msg = f"embedding_providers_map_core 中 key '{embedding_choice_key}' 的维度信息无效: '{emb_data}'. 需要格式 (id, model, text, dimension_int)."
            logger.error(err_msg)
            log_to_ui(err_msg, "FATAL")
            raise ValueError(err_msg)
            
        emb_identifier, emb_model_name, _, emb_dim = emb_data
        
        st.session_state.selected_embedding_provider_identifier = emb_identifier
        st.session_state.embedding_dimension = emb_dim # <<< Dimension is now set here
        st.session_state.selected_st_model_name = emb_model_name if "sentence_transformer" in emb_identifier else None
        logger.info(f"Core: Embedding Config: ID='{emb_identifier}', Model='{emb_model_name}', Dimension='{emb_dim}'")
        log_to_ui(f"嵌入配置: ID='{emb_identifier}', Model='{emb_model_name}', Dim='{emb_dim}'")
        
        if "sentence_transformer" in emb_identifier:
            if 'embedding_model_instance' not in st.session_state or st.session_state.get('loaded_st_model_name') != emb_model_name:
                logger.info(f"Core: Loading ST model: {emb_model_name}")
                log_to_ui(f"正在加载嵌入模型: {emb_model_name}...")
                # TODO: Replace mock with actual SentenceTransformer loading
                # st.session_state.embedding_model_instance = SentenceTransformer(emb_model_name) 
                st.session_state.embedding_model_instance = "MockSTModelInstanceLoaded" # Mock
                st.session_state.loaded_st_model_name = emb_model_name
        elif emb_identifier == "openai_official":
            st.session_state.embedding_model_instance = None 
            st.session_state.loaded_st_model_name = None

        llm_provider_name = llm_providers_map_core[llm_choice_key] # Uses map defined at top
        st.session_state.current_llm_provider = llm_provider_name
        log_to_ui(f"选择LLM模型: {llm_provider_name.upper()}。")
        
        # TODO: Implement your full API key validation logic here based on selections
        logger.info("Core: API Key checks (full implementation needed).")
        log_to_ui("API Key检查 (需完整实现)。")

        core_init_milvus_collections_internal() # This is called AFTER embedding_dimension is set
        core_seed_initial_lore()                
        
        st.session_state.current_chapter = 1 
        st.session_state.current_segment_number = 0 
        st.session_state.last_known_chapter = None 
        st.session_state.last_known_segment = None
        # TODO: Implement actual story resume logic using Milvus. Ensure numbers are int.
        log_to_ui("故事写作状态已初始化/重置。")
        
        st.session_state.system_initialized_successfully = True
        logger.info("Core: System initialization successful.")
        log_to_ui("核心系统初始化流程完成！", "SUCCESS")
        return True

    except Exception as e:
        st.session_state.system_initialized_successfully = False
        st.session_state.current_chapter = int(st.session_state.get('current_chapter', 1)) 
        st.session_state.current_segment_number = int(st.session_state.get('current_segment_number', 0))
        logger.error(f"Core: System initialization failed: {e}", exc_info=True)
        log_to_ui(f"核心系统初始化失败: {e}", "FATAL")
        raise


# --- UI Specific Core Functions (called by app_ui.py) ---
def core_generate_segment_text_for_ui(user_directive: str) -> Optional[str]:
    # TODO: Implement your full logic
    if not st.session_state.get('system_initialized_successfully', False): return "错误: 系统未初始化。"
    logger.info(f"Core: UI generating segment. Directive: {user_directive[:30]}...")
    return f"模拟UI片段生成，指令：{user_directive[:50]}" # Placeholder

def core_adopt_segment_from_ui(text_content: str, chapter: int, segment_num: int, user_directive_snippet: str):
    # TODO: Implement your full logic
    if not st.session_state.get('system_initialized_successfully', False): return False
    logger.info(f"Core: UI adopting segment Ch{chapter}-Seg{segment_num}.")
    st.session_state.last_adopted_segment_text = f"[Ch{chapter}-Seg{segment_num}]\n{text_content}"
    return True