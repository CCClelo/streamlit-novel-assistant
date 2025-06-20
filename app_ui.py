# novel_core.py
from typing import Dict, Optional, List 

import streamlit as st 
import logging
import os
import uuid
from datetime import datetime
import time 

_STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError: # Mock st.session_state if Streamlit is not available (for direct core testing)
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
        st_instance = MockSessionState() # Create instance of the mock
        st = type('StreamlitMock', (), {'session_state': st_instance})() # Assign to st
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("NovelCorePreImport").warning("Streamlit not found, using a mock session_state for core logic.")

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
if not logger.hasHandlers(): # Ensure logger is configured
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants Definition ---
# (All constants as defined previously, ensure they are complete and correct)
# Example:
OPENAI_API_KEY_ENV_NAME = "OPENAI_API_KEY"; DEEPSEEK_API_KEY_ENV_NAME = "DEEPSEEK_API_KEY"; GEMINI_API_KEY_ENV_NAME = "GEMINI_API_KEY"; CUSTOM_PROXY_API_KEY_ENV_NAME = "CUSTOM_PROXY_API_KEY"
OPENAI_EMBEDDING_MODEL_CORE = "text-embedding-3-small"; ST_MODEL_TEXT2VEC_CORE = "shibing624/text2vec-base-chinese"; ST_MODEL_BGE_LARGE_ZH_CORE = "BAAI/bge-large-zh-v1.5"
embedding_providers_map_core = {
    "1": ("sentence_transformer_text2vec", ST_MODEL_TEXT2VEC_CORE, "(本地ST Text2Vec中文)", 768),
    "2": ("sentence_transformer_bge_large_zh", ST_MODEL_BGE_LARGE_ZH_CORE, "(本地ST BGE中文Large)", 1024),
    "3": ("openai_official", OPENAI_EMBEDDING_MODEL_CORE, "(官方OpenAI API Key)", 1536)
}
llm_providers_map_core = {"1": "openai_official", "2": "deepseek", "3": "gemini", "4": "custom_proxy_llm"}
MILVUS_ALIAS_CORE = "default"; MILVUS_HOST_CORE = os.getenv("MILVUS_HOST", "localhost"); MILVUS_PORT_CORE = os.getenv("MILVUS_PORT", "19530")
ZILLIZ_CLOUD_URI_ENV_NAME = "ZILLIZ_CLOUD_URI"; ZILLIZ_CLOUD_TOKEN_ENV_NAME = "ZILLIZ_CLOUD_TOKEN"
COLLECTION_NAME_LORE_PREFIX_CORE = "novel_lore_mv" ; COLLECTION_NAME_STORY_PREFIX_CORE = "novel_story_mv"
NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_chapters_core"
SETTINGS_FILES_DIR_CORE = "./novel_setting_files"
# ... (All other necessary constants)

_core_original_os_environ_proxies: Dict[str, Optional[str]] = {}

# --- Helper Functions (core_get_custom_proxy_key, proxy funcs, etc.) ---
# TODO: PASTE YOUR FULL IMPLEMENTATIONS for these helpers if they are not already here.

# --- Embedding Functions ---
# TODO: PASTE YOUR FULL IMPLEMENTATIONS for core_get_openai_embeddings, core_get_st_embeddings.

# --- Milvus Related Functions (Ensure definition order if they call each other) ---
def core_chunk_text_by_paragraph(text: str) -> List[str]:
    # TODO: Implement actual logic
    return [p.strip() for p in text.replace('\r\n', '\n').replace('\r', '\n').split('\n\n') if p.strip()]

def core_load_and_vectorize_settings():
    # TODO: Implement actual logic, ensuring it uses Collection objects correctly
    if not st.session_state.get('milvus_initialized_core', False) or \
       not isinstance(st.session_state.get('lore_collection_milvus_obj'), Collection):
        logger.error("Core: Milvus lore collection not properly initialized for loading settings.")
        return
    lore_collection = st.session_state.lore_collection_milvus_obj
    logger.info(f"Core: Attempting to load settings into '{lore_collection.name if hasattr(lore_collection, 'name') else 'Invalid Collection'}' (actual logic needed).")
    # ... your logic using lore_collection.num_entities etc. ...


def core_seed_initial_lore():
    core_load_and_vectorize_settings() # Defined above
    # TODO: Implement actual fallback lore logic
    lore_collection = st.session_state.get('lore_collection_milvus_obj')
    if isinstance(lore_collection, Collection) and lore_collection.num_entities == 0:
        logger.info(f"Core: Lore collection '{lore_collection.name}' empty, seeding fallback (actual logic needed).")
    elif not isinstance(lore_collection, Collection):
         logger.error("Cannot seed lore, lore_collection_milvus_obj is not a Collection.")


def core_init_milvus_collections_internal():
    log_to_ui_core = lambda msg, lvl="INFO": st.session_state.log_messages.insert(0, f"[{lvl.upper()}][CoreInitMilvus] {msg}") if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state') and 'log_messages' in st.session_state else None

    logger.info("Core: core_init_milvus_collections_internal called.")
    log_to_ui_core("开始Milvus集合初始化...")

    if not st.session_state.get('embedding_dimension') or \
       not isinstance(st.session_state.embedding_dimension, int) or \
       st.session_state.embedding_dimension <= 0:
        err_msg = f"无效或缺失的 embedding_dimension: {st.session_state.get('embedding_dimension')}"
        logger.error(err_msg)
        log_to_ui_core(err_msg, "FATAL")
        raise ValueError(err_msg)
    
    logger.debug(f"Core: Using embedding_dimension: {st.session_state.embedding_dimension}")

    # --- 1. ESTABLISH CONNECTION FIRST ---
    zilliz_uri = os.getenv(ZILLIZ_CLOUD_URI_ENV_NAME)
    zilliz_token = os.getenv(ZILLIZ_CLOUD_TOKEN_ENV_NAME)
    token_status_for_log = "SET" if zilliz_token and zilliz_token != "YOUR_ACTUAL_ZILLIZ_CLOUD_TOKEN_HERE" else "NOT SET or using placeholder"
    logger.info(f"Core ENV CHECK (Milvus Init): ZILLIZ_CLOUD_URI='{zilliz_uri}', ZILLIZ_CLOUD_TOKEN is {token_status_for_log}")
    log_to_ui_core(f"Env Check: ZILLIZ_CLOUD_URI='{zilliz_uri}', Token Status='{token_status_for_log}'")

    try:
        connection_alias_to_use = MILVUS_ALIAS_CORE # Use the defined alias
        if connections.has_connection(connection_alias_to_use):
            logger.info(f"Core: Removing existing Milvus connection '{connection_alias_to_use}' before reconnecting.")
            connections.remove_connection(connection_alias_to_use)

        if zilliz_uri and zilliz_token and \
           zilliz_uri != "your_zilliz_cluster_uri_from_screenshot" and \
           zilliz_token != "YOUR_ACTUAL_ZILLIZ_CLOUD_TOKEN_HERE": # Check against known placeholder values
            
            logger.info(f"Core: Attempting to connect to Zilliz Cloud Milvus: {zilliz_uri}")
            log_to_ui_core(f"尝试连接到Zilliz Cloud: {zilliz_uri}")
            connections.connect(alias=connection_alias_to_use, uri=zilliz_uri, token=zilliz_token)
            st.session_state.milvus_target = "Zilliz Cloud"
        else:
            logger.info(f"Core: Zilliz Cloud config incomplete or placeholder. Attempting local Milvus: {MILVUS_HOST_CORE}:{MILVUS_PORT_CORE}")
            log_to_ui_core(f"尝试连接到本地Milvus: {MILVUS_HOST_CORE}:{MILVUS_PORT_CORE}")
            connections.connect(alias=connection_alias_to_use, host=MILVUS_HOST_CORE, port=MILVUS_PORT_CORE)
            st.session_state.milvus_target = "Local"
        
        logger.info(f"Core: Milvus connected (Target: {st.session_state.get('milvus_target', '未知')}).")
        log_to_ui_core(f"Milvus连接成功 (目标: {st.session_state.get('milvus_target', '未知')}).")

    except Exception as e:
        logger.error(f"Core: Milvus连接失败 during connect(): {e}", exc_info=True)
        log_to_ui_core(f"Milvus连接失败: {e}", "ERROR")
        raise # Critical error, stop initialization

    # --- 2. DEFINE Collection Names and Schemas (AFTER successful connection) ---
    provider_short, model_short_suffix = "", ""
    selected_provider_id = st.session_state.selected_embedding_provider_identifier
    selected_st_model = st.session_state.get('selected_st_model_name')
    # ... (logic to set provider_short, model_short_suffix based on selected_provider_id) ...
    if selected_provider_id == "openai_official":
        provider_short, model_short_suffix = "oai", OPENAI_EMBEDDING_MODEL_CORE.split('-')[-1][:6]
    elif selected_provider_id.startswith("sentence_transformer_"):
        provider_short = selected_provider_id.replace("sentence_transformer_", "st")[:6]
        if selected_st_model: model_short_suffix = selected_st_model.split('/')[-1].replace('-', '_').replace('.', '_')[:10]
        else: model_short_suffix = "unknownst"


    def sanitize_milvus_name_local_helper(name): # Local helper to avoid global namespace issues
        s_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
        if not (s_name and (s_name[0].isalpha() or s_name[0] == '_')): s_name = "_" + s_name
        return s_name[:255]

    lore_col_name = sanitize_milvus_name_local_helper(f"{COLLECTION_NAME_LORE_PREFIX_CORE}_{provider_short}_{model_short_suffix}")
    story_col_name = sanitize_milvus_name_local_helper(f"{COLLECTION_NAME_STORY_PREFIX_CORE}_{provider_short}_{model_short_suffix}")
    
    st.session_state.lore_collection_name = lore_col_name
    st.session_state.story_collection_name = story_col_name
    log_to_ui_core(f"Lore Collection目标名称: {lore_col_name}")
    log_to_ui_core(f"Story Collection目标名称: {story_col_name}")

    # --- Define Schemas ---
    pk_field = FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=st.session_state.embedding_dimension)
    text_content_field = FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65530)
    timestamp_field = FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64)
    lore_specific_fields = [FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255), FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100)]
    story_specific_fields = [FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=50), FieldSchema(name="segment_number", dtype=DataType.VARCHAR, max_length=50)]
    base_schema_fields = [pk_field, embedding_field, text_content_field, timestamp_field]
    lore_schema_list = base_schema_fields + lore_specific_fields
    story_schema_list = base_schema_fields + story_specific_fields

    # --- 3. Define _create_or_get_collection helper function (AFTER successful connection) ---
    def _create_or_get_collection(name: str, schema_list: list, desc: str) -> Collection:
        # This helper itself uses the established connection implicitly via MILVUS_ALIAS_CORE
        if utility.has_collection(name, using=MILVUS_ALIAS_CORE): # This needs an active connection
            collection = Collection(name, using=MILVUS_ALIAS_CORE) # This also needs an active connection
            logger.info(f"Core: Milvus 集合 '{name}' 已存在，获取对象。")
            log_to_ui_core(f"Milvus集合 '{name}' 已存在。")
        else:
            logger.info(f"Core: Milvus 集合 '{name}' 不存在，创建...")
            log_to_ui_core(f"创建Milvus集合: {name}...")
            schema_obj = CollectionSchema(fields=schema_list, description=desc, enable_dynamic_field=True)
            collection = Collection(name, schema=schema_obj, using=MILVUS_ALIAS_CORE) # And this
            
            index_name = f"idx_emb_{name[:50].replace('_', '')}" # Sanitize index name a bit more
            index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 512}}
            logger.info(f"Core: 为 '{name}' 的 'embedding' 字段创建索引 '{index_name}'...")
            log_to_ui_core(f"为集合 {name} 创建索引...")
            try:
                collection.create_index(field_name="embedding", index_params=index_params, index_name=index_name)
                logger.info(f"Core: 为 '{name}' 的 'embedding' 创建了 HNSW 索引 '{index_name}'。")
                log_to_ui_core(f"集合 {name} 索引创建成功。")
            except Exception as e_create_idx: # Catch specific Milvus errors if possible
                logger.error(f"Core: 为 '{name}' 创建索引失败: {e_create_idx}")
                log_to_ui_core(f"集合 {name} 索引创建失败: {e_create_idx}", "ERROR")
        
        logger.info(f"Core: 尝试加载集合 '{name}'...")
        log_to_ui_core(f"加载集合 {name}...")
        collection.load() 
        logger.info(f"Core: Milvus collection '{name}' is loaded.")
        log_to_ui_core(f"集合 {name} 加载成功。")
        return collection 

    # --- 4. Get or create the actual collections ---
    try:
        st.session_state.lore_collection_milvus_obj = _create_or_get_collection(lore_col_name, lore_schema_list, "Novel Lore")
        st.session_state.story_collection_milvus_obj = _create_or_get_collection(story_col_name, story_schema_list, "Novel Story")
    except Exception as e_coll_create:
        logger.error(f"Core: Failed to create/get collections: {e_coll_create}", exc_info=True)
        log_to_ui_core(f"创建/获取Milvus集合失败: {e_coll_create}", "ERROR")
        raise 

    st.session_state.milvus_initialized_core = True # Mark Milvus part of init as successful
    logger.info(f"Core: Milvus collections '{st.session_state.lore_collection_name}' and '{st.session_state.story_collection_name}' ready.")
    log_to_ui_core("Milvus集合准备就绪。")


# --- Main Initialization Function (called by UI) ---
def core_initialize_system(embedding_choice_key: str, llm_choice_key: str, api_keys_from_ui: dict):
    # ... (Full implementation from previous version, ensuring it calls the corrected
    #      core_init_milvus_collections_internal AFTER embedding_dimension is set.
    #      And ensure it sets st.session_state.system_initialized_successfully = True
    #      ONLY if ALL steps, including Milvus init and seeding, are successful.)
    #      And in its except block, it sets system_initialized_successfully = False.
    #      THIS IS A LARGE FUNCTION, ENSURE IT'S THE COMPLETE, CORRECTED VERSION.
    # For brevity, I'll show the call sequence again:
    log_to_ui_core_init = lambda msg, lvl="INFO": st.session_state.log_messages.insert(0, f"[{lvl.upper()}][CoreSysInit] {msg}") if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state') and 'log_messages' in st.session_state else None
    try:
        log_to_ui_core_init("开始核心系统初始化...")
        st.session_state.system_initialized_successfully = False # Default

        # ... (Embedding provider setup, ST model loading, LLM provider setup, API Key checks - from previous) ...
        # This part sets st.session_state.embedding_dimension correctly BEFORE calling Milvus init.
        # Example from previous:
        emb_identifier, emb_model_name, _, emb_dim = embedding_providers_map_core[embedding_choice_key]
        st.session_state.selected_embedding_provider_identifier = emb_identifier
        st.session_state.embedding_dimension = emb_dim
        st.session_state.selected_st_model_name = emb_model_name if "sentence_transformer" in emb_identifier else None
        # ... (Actual ST model loading if sentence_transformer)

        core_init_milvus_collections_internal() # <<< This is now called after dimension is set
        core_seed_initial_lore()
        
        # ... (Story resume logic) ...

        st.session_state.system_initialized_successfully = True
        log_to_ui_core_init("核心系统初始化流程完成！", "SUCCESS")
        return True
    except Exception as e:
        st.session_state.system_initialized_successfully = False
        # ... (error logging and safe defaults) ...
        log_to_ui_core_init(f"核心系统初始化失败: {e}", "FATAL")
        raise


# --- UI Specific Core Functions ---
# (core_generate_segment_text_for_ui, core_adopt_segment_from_ui)
# TODO: PASTE YOUR FULL IMPLEMENTATIONS FOR THESE, ensuring they use the correctly
# initialized Milvus objects (checking isinstance Collection) and other session_state vars.
# Example:
def core_generate_segment_text_for_ui(user_directive: str) -> Optional[str]:
    if not st.session_state.get('system_initialized_successfully', False): return "错误: 系统未初始化。"
    # ... (full logic: retrieve lore, recent, build prompt, call LLM)
    return "模拟UI片段生成..."

def core_adopt_segment_from_ui(text_content: str, chapter: int, segment_num: int, user_directive_snippet: str):
    if not st.session_state.get('system_initialized_successfully', False): return False
    # ... (full logic: save MD, get vector, add to Milvus)
    return True

# Ensure all other necessary helper functions (proxies, specific embedding/LLM calls, Milvus CRUD ops)
# are fully implemented and defined in the correct order.