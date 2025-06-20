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
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("NovelCorePreImport").error(f"Core: Missing critical libraries: {e}. Please install them.")
    raise

load_dotenv()
logger = logging.getLogger("NovelCore")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants Definition ---
# (All constants from the previous complete version should be here)
OPENAI_API_KEY_ENV_NAME = "OPENAI_API_KEY"; DEEPSEEK_API_KEY_ENV_NAME = "DEEPSEEK_API_KEY"; GEMINI_API_KEY_ENV_NAME = "GEMINI_API_KEY"; CUSTOM_PROXY_API_KEY_ENV_NAME = "CUSTOM_PROXY_API_KEY"
OPENAI_OFFICIAL_HTTP_PROXY_CORE = os.getenv("OPENAI_OFFICIAL_HTTP_PROXY"); OPENAI_OFFICIAL_HTTPS_PROXY_CORE = os.getenv("OPENAI_OFFICIAL_HTTPS_PROXY")
DEEPSEEK_LLM_HTTP_PROXY_CORE = os.getenv("DEEPSEEK_LLM_HTTP_PROXY"); DEEPSEEK_LLM_HTTPS_PROXY_CORE = os.getenv("DEEPSEEK_LLM_HTTPS_PROXY")
CUSTOM_LLM_HTTP_PROXY_CORE = os.getenv("CUSTOM_LLM_HTTP_PROXY"); CUSTOM_LLM_HTTPS_PROXY_CORE = os.getenv("CUSTOM_LLM_HTTPS_PROXY")
GEMINI_HTTP_PROXY_CORE = os.getenv("GEMINI_HTTP_PROXY", os.getenv("GLOBAL_HTTP_PROXY")); GEMINI_HTTPS_PROXY_CORE = os.getenv("GEMINI_HTTPS_PROXY", os.getenv("GLOBAL_HTTPS_PROXY"))
CUSTOM_PROXY_BASE_URL_CORE = os.getenv("CUSTOM_PROXY_BASE_URL", "https://api.openai-next.com/v1"); HARDCODED_CUSTOM_PROXY_KEY_CORE = "sk-0rRacqNU0aY2DrOV1601DfE76a4d4a87A14f97383e25095b"
MILVUS_ALIAS_CORE = "default"; MILVUS_HOST_CORE = os.getenv("MILVUS_HOST", "localhost"); MILVUS_PORT_CORE = os.getenv("MILVUS_PORT", "19530")
ZILLIZ_CLOUD_URI_ENV_NAME = "ZILLIZ_CLOUD_URI"; ZILLIZ_CLOUD_TOKEN_ENV_NAME = "ZILLIZ_CLOUD_TOKEN"
OPENAI_EMBEDDING_MODEL_CORE = "text-embedding-3-small"; ST_MODEL_TEXT2VEC_CORE = "shibing624/text2vec-base-chinese"; ST_MODEL_BGE_LARGE_ZH_CORE = "BAAI/bge-large-zh-v1.5"
OPENAI_LLM_MODEL_CORE = "gpt-3.5-turbo"; DEEPSEEK_LLM_MODEL_CORE = "deepseek-chat"; GEMINI_LLM_MODEL_CORE = "gemini-1.5-flash-latest" ; CUSTOM_PROXY_LLM_MODEL_CORE = "gpt-3.5-turbo"
DEEPSEEK_BASE_URL_CORE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
SETTINGS_FILES_DIR_CORE = "./novel_setting_files"; NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_chapters_core"
COLLECTION_NAME_LORE_PREFIX_CORE = "novel_lore_mv" ; COLLECTION_NAME_STORY_PREFIX_CORE = "novel_story_mv"
embedding_providers_map_core = {
    "1": ("sentence_transformer_text2vec", ST_MODEL_TEXT2VEC_CORE, "(本地ST Text2Vec中文)", 768),
    "2": ("sentence_transformer_bge_large_zh", ST_MODEL_BGE_LARGE_ZH_CORE, "(本地ST BGE中文Large)", 1024),
    "3": ("openai_official", OPENAI_EMBEDDING_MODEL_CORE, "(官方OpenAI API Key)", 1536)
}
llm_providers_map_core = {"1": "openai_official", "2": "deepseek", "3": "gemini", "4": "custom_proxy_llm"}

_core_original_os_environ_proxies: Dict[str, Optional[str]] = {}

# --- Helper Functions ---
# ... (core_get_custom_proxy_key, core_set_temp_os_proxies, core_restore_original_os_proxies,
#      core_get_httpx_client_with_proxy - FULL IMPLEMENTATIONS from previous correct version)

# --- Embedding Functions ---
# ... (core_get_openai_embeddings, core_get_st_embeddings - FULL IMPLEMENTATIONS from previous correct version)

# --- Milvus Related Functions ---
def core_chunk_text_by_paragraph(text: str) -> List[str]:
    # ... (Full implementation) ...
    paragraphs = text.replace('\r\n', '\n').replace('\r', '\n').split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]

def core_load_and_vectorize_settings():
    # ... (Full implementation, ensuring it uses the Collection object correctly) ...
    if not st.session_state.get('milvus_initialized_core', False) or \
       not st.session_state.get('lore_collection_milvus_obj') or \
       not isinstance(st.session_state.lore_collection_milvus_obj, Collection):
        logger.error("Core: Milvus lore collection not properly initialized for seeding.")
        # st.session_state.log_messages.insert(0, ...) # Add log if st is available
        return
    lore_collection = st.session_state.lore_collection_milvus_obj # This is now a Collection object
    # ... (rest of your logic using lore_collection.num_entities etc.)
    logger.info("Core: Loading and vectorizing settings (actual implementation needed).")


def core_seed_initial_lore():
    core_load_and_vectorize_settings()
    lore_collection = st.session_state.get('lore_collection_milvus_obj')
    if isinstance(lore_collection, Collection) and lore_collection.num_entities == 0:
        # ... (Full fallback lore logic) ...
        logger.info(f"Core: Lore collection '{lore_collection.name}' is empty, adding fallback.")
    elif not isinstance(lore_collection, Collection):
        logger.error("Core: lore_collection is not a valid Milvus Collection object in seed_initial_lore.")


def core_init_milvus_collections_internal():
    if not st.session_state.get('embedding_dimension'):
        raise ValueError("Core: Embedding dimension 未在 session_state 中设置。")
    if not st.session_state.get('selected_embedding_provider_identifier'):
        raise ValueError("Core: Embedding provider 未在 session_state 中设置。")

    zilliz_uri_from_env = os.getenv(ZILLIZ_CLOUD_URI_ENV_NAME)
    zilliz_token_from_env = os.getenv(ZILLIZ_CLOUD_TOKEN_ENV_NAME)
    # ... (Zilliz Cloud connection logic from previous correct version) ...

    try:
        # Connection logic (Zilliz Cloud or Local)
        # ... (This part remains the same, ensuring connection is established) ...
        # Example:
        if zilliz_uri_from_env and zilliz_token_from_env and \
           zilliz_uri_from_env != "your_zilliz_cluster_uri_from_screenshot" and \
           zilliz_token_from_env != "YOUR_ACTUAL_ZILLIZ_CLOUD_TOKEN_HERE":
            if connections.has_connection(MILVUS_ALIAS_CORE): connections.remove_connection(MILVUS_ALIAS_CORE)
            connections.connect(alias=MILVUS_ALIAS_CORE, uri=zilliz_uri_from_env, token=zilliz_token_from_env)
            st.session_state.milvus_target = "Zilliz Cloud"
        elif not connections.has_connection(MILVUS_ALIAS_CORE):
            connections.connect(alias=MILVUS_ALIAS_CORE, host=MILVUS_HOST_CORE, port=MILVUS_PORT_CORE)
            st.session_state.milvus_target = "Local"
        logger.info(f"Core: Milvus connected (Target: {st.session_state.get('milvus_target', '未知')}).")
    except Exception as e:
        logger.error(f"Core: Milvus连接失败: {e}", exc_info=True)
        raise

    provider_short, model_short_suffix = "", ""
    selected_provider_id = st.session_state.selected_embedding_provider_identifier
    selected_st_model = st.session_state.get('selected_st_model_name')
    if selected_provider_id == "openai_official":
        provider_short, model_short_suffix = "oai", OPENAI_EMBEDDING_MODEL_CORE.split('-')[-1][:6]
    elif selected_provider_id.startswith("sentence_transformer_"):
        provider_short = selected_provider_id.replace("sentence_transformer_", "st")[:6]
        if selected_st_model: model_short_suffix = selected_st_model.split('/')[-1].replace('-', '_').replace('.', '_')[:10]
        else: model_short_suffix = "unknownst"
    else: raise ValueError(f"Core: 未知的嵌入提供商标识符: {selected_provider_id}")

    def sanitize_milvus_name_core_local(name):
        s_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
        if not (s_name and (s_name[0].isalpha() or s_name[0] == '_')): s_name = "_" + s_name
        return s_name[:255]

    # --- DEFINITION OF lore_col_name and story_col_name ---
    lore_col_name = sanitize_milvus_name_core_local(f"{COLLECTION_NAME_LORE_PREFIX_CORE}_{provider_short}_{model_short_suffix}")
    story_col_name = sanitize_milvus_name_core_local(f"{COLLECTION_NAME_STORY_PREFIX_CORE}_{provider_short}_{model_short_suffix}")
    
    st.session_state.lore_collection_name = lore_col_name
    st.session_state.story_collection_name = story_col_name # Stored for potential use by other functions
    logger.info(f"Core: Lore Collection Name will be: {lore_col_name}")
    logger.info(f"Core: Story Collection Name will be: {story_col_name}")


    pk_field = FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=st.session_state.embedding_dimension)
    text_content_field = FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65530)
    timestamp_field = FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64)
    lore_specific_fields = [FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255), FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100)]
    story_specific_fields = [FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=50), FieldSchema(name="segment_number", dtype=DataType.VARCHAR, max_length=50)]
    base_schema_fields = [pk_field, embedding_field, text_content_field, timestamp_field]
    lore_schema_list = base_schema_fields + lore_specific_fields
    story_schema_list = base_schema_fields + story_specific_fields

    def _create_or_get_collection(name: str, schema_list: list, desc: str) -> Collection:
        # This is the CORRECTED helper function that returns a Collection object
        if utility.has_collection(name, using=MILVUS_ALIAS_CORE):
            collection = Collection(name, using=MILVUS_ALIAS_CORE)
            logger.info(f"Core: Milvus 集合 '{name}' 已存在，获取对象。")
        else:
            logger.info(f"Core: Milvus 集合 '{name}' 不存在，创建...")
            schema = CollectionSchema(fields=schema_list, description=desc, enable_dynamic_field=True)
            collection = Collection(name, schema=schema, using=MILVUS_ALIAS_CORE)
            index_name = f"idx_emb_{name[:50]}"
            index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 512}}
            try:
                collection.create_index(field_name="embedding", index_params=index_params, index_name=index_name)
                logger.info(f"Core: 为 '{name}' 的 'embedding' 创建了 HNSW 索引 '{index_name}'。")
            except Exception as e_idx:
                logger.error(f"Core: 创建索引 for '{name}' 失败: {e_idx}") # Log and continue, load might still work if index exists from previous attempt
        
        logger.info(f"Core: 尝试加载集合 '{name}'...")
        collection.load() 
        logger.info(f"Core: Milvus collection '{name}' is loaded.")
        return collection # Return the actual Collection object

    # These lines now correctly receive Collection objects
    st.session_state.lore_collection_milvus_obj = _create_or_get_collection(lore_col_name, lore_schema_list, "Novel Lore")
    # The error was likely that story_col_name was used LATER in this function or another function
    # WITHOUT being properly passed or retrieved from session_state, if this function was refactored poorly.
    # The line below uses the local variable 'story_col_name' which IS defined above.
    st.session_state.story_collection_milvus_obj = _create_or_get_collection(story_col_name, story_schema_list, "Novel Story")
    
    st.session_state.milvus_initialized_core = True
    logger.info(f"Core: Milvus collections '{st.session_state.lore_collection_name}' and '{st.session_state.story_collection_name}' ready.")


# ... (core_add_story_segment_to_milvus, core_retrieve_relevant_lore, core_retrieve_recent_story_segments -
#      Full implementations from your previous correct versions, ensure they use session_state for Milvus objects
#      and check isinstance(obj, Collection) before using Collection methods.)

# --- LLM Generation Function ---
# ... (core_generate_with_llm - Full implementation from your previous correct version)
# Ensure all constants like OPENAI_LLM_MODEL_CORE etc. are used.

# --- Main Initialization Function (called by UI) ---
# This function definition must come AFTER all functions it calls (like core_init_milvus_collections_internal, core_seed_initial_lore)
def core_initialize_system(embedding_choice_key: str, llm_choice_key: str, api_keys_from_ui: dict):
    # ... (Full implementation from previous, ensuring it calls the defined functions above) ...
    # This will set up st.session_state and then call core_init_milvus_collections_internal and core_seed_initial_lore
    try:
        st.session_state.log_messages.insert(0, "[INFO][Core] 开始核心系统初始化...")
        logger.info("Core: Initializing system...")
        st.session_state.system_initialized_successfully = False # Default

        st.session_state.selected_embedding_provider_key = embedding_choice_key
        # ... (set other session_state variables from choices and api_keys_from_ui) ...
        # ... (load ST model if chosen and store in st.session_state.embedding_model_instance) ...
        # ... (API Key checks) ...

        core_init_milvus_collections_internal() # Call the corrected function
        core_seed_initial_lore() # Call the function defined above
        
        # ... (Story resume logic, ensuring segment numbers are int) ...
        # Example ensuring int:
        # st.session_state.current_segment_number = int(st.session_state.get('last_known_segment', 0))

        st.session_state.system_initialized_successfully = True
        logger.info("Core: System initialization successful.")
        # ...
        return True
    except Exception as e:
        st.session_state.system_initialized_successfully = False
        # ... (error logging and safe defaults for chapter/segment) ...
        raise


# --- UI Specific Core Functions ---
# ... (core_generate_segment_text_for_ui, core_adopt_segment_from_ui - Full implementations)
# Ensure they use the correctly initialized Milvus objects and other session_state variables.
# For example, core_adopt_segment_from_ui will use NOVEL_MD_OUTPUT_DIR_CORE.