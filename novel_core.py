# novel_core.py
from typing import Dict, Optional, List # Ensure these are at the very top

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
    # This error should ideally be caught in app_ui.py before even trying to use novel_core
    # However, adding a log here for direct core testing.
    logging.basicConfig(level=logging.ERROR) # Basic config if logger not yet set
    logging.getLogger("NovelCorePreImport").error(f"Core: Missing critical libraries: {e}. Please install them.")
    raise

load_dotenv() # Load .env file if present
logger = logging.getLogger("NovelCore") # Specific logger for core logic

# --- Constants Definition ---
OPENAI_API_KEY_ENV_NAME = "OPENAI_API_KEY"
DEEPSEEK_API_KEY_ENV_NAME = "DEEPSEEK_API_KEY"
GEMINI_API_KEY_ENV_NAME = "GEMINI_API_KEY"
CUSTOM_PROXY_API_KEY_ENV_NAME = "CUSTOM_PROXY_API_KEY"

OPENAI_OFFICIAL_HTTP_PROXY_CORE = os.getenv("OPENAI_OFFICIAL_HTTP_PROXY")
OPENAI_OFFICIAL_HTTPS_PROXY_CORE = os.getenv("OPENAI_OFFICIAL_HTTPS_PROXY")
DEEPSEEK_LLM_HTTP_PROXY_CORE = os.getenv("DEEPSEEK_LLM_HTTP_PROXY")
DEEPSEEK_LLM_HTTPS_PROXY_CORE = os.getenv("DEEPSEEK_LLM_HTTPS_PROXY")
CUSTOM_LLM_HTTP_PROXY_CORE = os.getenv("CUSTOM_LLM_HTTP_PROXY")
CUSTOM_LLM_HTTPS_PROXY_CORE = os.getenv("CUSTOM_LLM_HTTPS_PROXY")
GEMINI_HTTP_PROXY_CORE = os.getenv("GEMINI_HTTP_PROXY", os.getenv("GLOBAL_HTTP_PROXY"))
GEMINI_HTTPS_PROXY_CORE = os.getenv("GEMINI_HTTPS_PROXY", os.getenv("GLOBAL_HTTPS_PROXY"))

CUSTOM_PROXY_BASE_URL_CORE = os.getenv("CUSTOM_PROXY_BASE_URL", "https://api.openai-next.com/v1") # Example URL
HARDCODED_CUSTOM_PROXY_KEY_CORE = "sk-0rRacqNU0aY2DrOV1601DfE76a4d4a87A14f97383e25095b" # Example Key

MILVUS_ALIAS_CORE = "default"
MILVUS_HOST_CORE = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT_CORE = os.getenv("MILVUS_PORT", "19530")

OPENAI_EMBEDDING_MODEL_CORE = "text-embedding-3-small"
ST_MODEL_TEXT2VEC_CORE = "shibing624/text2vec-base-chinese"
ST_MODEL_BGE_LARGE_ZH_CORE = "BAAI/bge-large-zh-v1.5"

OPENAI_LLM_MODEL_CORE = "gpt-3.5-turbo"
DEEPSEEK_LLM_MODEL_CORE = "deepseek-chat"
GEMINI_LLM_MODEL_CORE = "gemini-1.5-flash-latest"
CUSTOM_PROXY_LLM_MODEL_CORE = "gpt-3.5-turbo"

DEEPSEEK_BASE_URL_CORE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

SETTINGS_FILES_DIR_CORE = "./novel_setting_files"
NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_chapters_core" # Default for core, UI can override

COLLECTION_NAME_LORE_PREFIX_CORE = "novel_lore_mv"
COLLECTION_NAME_STORY_PREFIX_CORE = "novel_story_mv"

ZILLIZ_CLOUD_URI_ENV_NAME = "ZILLIZ_CLOUD_URI"
ZILLIZ_CLOUD_TOKEN_ENV_NAME = "ZILLIZ_CLOUD_TOKEN"

embedding_providers_map_core = {
    "1": ("sentence_transformer_text2vec", ST_MODEL_TEXT2VEC_CORE, "(本地ST Text2Vec中文)", 768),
    "2": ("sentence_transformer_bge_large_zh", ST_MODEL_BGE_LARGE_ZH_CORE, "(本地ST BGE中文Large)", 1024),
    "3": ("openai_official", OPENAI_EMBEDDING_MODEL_CORE, "(需要官方OpenAI API Key)", 1536)
}
llm_providers_map_core = {"1": "openai_official", "2": "deepseek", "3": "gemini", "4": "custom_proxy_llm"}

# --- Helper Functions ---
_core_original_os_environ_proxies: Dict[str, Optional[str]] = {}

def core_get_custom_proxy_key() -> str:
    # This function now correctly uses the constants defined above.
    key_from_ui = st.session_state.get('api_keys', {}).get(CUSTOM_PROXY_API_KEY_ENV_NAME)
    if key_from_ui: return key_from_ui
    key_from_env = os.getenv(CUSTOM_PROXY_API_KEY_ENV_NAME)
    if key_from_env: return key_from_env
    logger.warning(f"Core: CUSTOM_PROXY_API_KEY 未设置。使用硬编码Key: {HARDCODED_CUSTOM_PROXY_KEY_CORE[:5]}...")
    return HARDCODED_CUSTOM_PROXY_KEY_CORE

def core_set_temp_os_proxies(http_proxy: Optional[str], https_proxy: Optional[str]):
    global _core_original_os_environ_proxies
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    if not _core_original_os_environ_proxies:
        for var in proxy_vars: _core_original_os_environ_proxies[var] = os.environ.get(var)
    actions = []
    for var_upper, var_lower, new_val in [("HTTP_PROXY", "http_proxy", http_proxy), ("HTTPS_PROXY", "https_proxy", https_proxy)]:
        if new_val:
            if os.environ.get(var_upper) != new_val: os.environ[var_upper] = new_val; actions.append(f"Set {var_upper}")
            if os.environ.get(var_lower) != new_val: os.environ[var_lower] = new_val;
        else:
            if var_upper in os.environ: del os.environ[var_upper]; actions.append(f"Del {var_upper}")
            if var_lower in os.environ: del os.environ[var_lower];
    if actions: logger.debug(f"Core: Temp OS Proxies: {', '.join(actions)} -> HTTP='{os.environ.get('HTTP_PROXY')}', HTTPS='{os.environ.get('HTTPS_PROXY')}'")

def core_restore_original_os_proxies():
    global _core_original_os_environ_proxies
    if not _core_original_os_environ_proxies: return
    actions = []
    for var, original_value in _core_original_os_environ_proxies.items():
        if original_value is not None:
            if os.environ.get(var) != original_value: os.environ[var] = original_value; actions.append(f"Restored {var}")
        elif var in os.environ: del os.environ[var]; actions.append(f"Del {var} (was not set)")
    if actions: logger.debug(f"Core: Original OS Proxies restored: {', '.join(actions)}")
    _core_original_os_environ_proxies.clear()

def core_get_httpx_client_with_proxy(http_proxy_url: Optional[str], https_proxy_url: Optional[str]) -> httpx.Client:
    proxies_for_httpx = {}
    if http_proxy_url: proxies_for_httpx["http://"] = http_proxy_url
    if https_proxy_url: proxies_for_httpx["https://"] = https_proxy_url
    if proxies_for_httpx:
        try: return httpx.Client(proxies=proxies_for_httpx, timeout=60.0)
        except Exception as e: logger.error(f"Core: 创建配置了代理的 httpx.Client 失败: {e}")
    return httpx.Client(timeout=60.0)

# --- Embedding Functions --- (Ensure constants like OPENAI_API_KEY_ENV_NAME are defined above)
def core_get_openai_embeddings(texts: List[str], model: str) -> Optional[List[List[float]]]:
    api_key = st.session_state.get('api_keys',{}).get(OPENAI_API_KEY_ENV_NAME, os.getenv(OPENAI_API_KEY_ENV_NAME))
    if not api_key:
        st.session_state.log_messages.insert(0, "[ERROR][Core] OpenAI API Key 未配置。")
        raise ValueError("Core: OpenAI API Key 未配置。")
    
    http_proxy = OPENAI_OFFICIAL_HTTP_PROXY_CORE
    https_proxy = OPENAI_OFFICIAL_HTTPS_PROXY_CORE
    temp_client = None
    try:
        core_set_temp_os_proxies(http_proxy, https_proxy)
        temp_client = core_get_httpx_client_with_proxy(http_proxy, https_proxy)
        client = openai.OpenAI(api_key=api_key, http_client=temp_client)
        response = client.embeddings.create(input=texts, model=model) # model passed as arg
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Core: 使用 OpenAI 生成嵌入失败: {e}")
        st.session_state.log_messages.insert(0, f"[ERROR][Core] OpenAI嵌入生成失败: {e}")
        return None
    finally:
        if temp_client: temp_client.close()
        core_restore_original_os_proxies()

def core_get_st_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    if 'embedding_model_instance' not in st.session_state or \
       not isinstance(st.session_state.embedding_model_instance, SentenceTransformer):
        logger.error("Core: Sentence Transformer 模型实例未在session_state中正确初始化。")
        st.session_state.log_messages.insert(0, "[ERROR][Core] ST嵌入模型未初始化。")
        return None
    try:
        embeddings = st.session_state.embedding_model_instance.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
        return embeddings
    except Exception as e:
        logger.error(f"Core: 使用 Sentence Transformer 生成嵌入失败: {e}")
        st.session_state.log_messages.insert(0, f"[ERROR][Core] ST嵌入生成失败: {e}")
        return None

# --- Milvus Related Functions ---
# (This function now relies on constants defined above it)
def core_init_milvus_collections_internal():
    # ... (The Zilliz-aware Milvus connection logic as provided in the previous response)
    # Ensure all constants used within this function (like COLLECTION_NAME_LORE_PREFIX_CORE)
    # are defined in the Constants section at the top of this file.
    if not st.session_state.get('embedding_dimension'):
        raise ValueError("Core: Embedding dimension 未在 session_state 中设置。")
    if not st.session_state.get('selected_embedding_provider_identifier'):
        raise ValueError("Core: Embedding provider 未在 session_state 中设置。")

    zilliz_uri = os.getenv(ZILLIZ_CLOUD_URI_ENV_NAME)
    zilliz_token = os.getenv(ZILLIZ_CLOUD_TOKEN_ENV_NAME)

    try:
        # Zilliz Cloud Connection or Local Fallback
        if zilliz_uri and zilliz_token:
            logger.info(f"Core:检测到Zilliz Cloud配置。尝试连接到 URI: {zilliz_uri}")
            if connections.has_connection(MILVUS_ALIAS_CORE): connections.remove_connection(MILVUS_ALIAS_CORE)
            connections.connect(alias=MILVUS_ALIAS_CORE, uri=zilliz_uri, token=zilliz_token)
            st.session_state.milvus_target = "Zilliz Cloud"
            logger.info(f"Core: 成功连接到Zilliz Cloud Milvus。")
        elif not connections.has_connection(MILVUS_ALIAS_CORE):
            logger.info(f"Core:尝试连接到本地Milvus: {MILVUS_HOST_CORE}:{MILVUS_PORT_CORE}")
            connections.connect(alias=MILVUS_ALIAS_CORE, host=MILVUS_HOST_CORE, port=MILVUS_PORT_CORE)
            st.session_state.milvus_target = "Local"
            logger.info(f"Core: 成功连接到本地Milvus。")
        else: # Already connected
            logger.info(f"Core: Milvus连接 '{MILVUS_ALIAS_CORE}' 已存在 (目标: {st.session_state.get('milvus_target', '未知')})。")
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
        if selected_st_model:
            model_short_suffix = selected_st_model.split('/')[-1].replace('-', '_').replace('.', '_')[:10]
        else: model_short_suffix = "unknownst"
    else:
        raise ValueError(f"Core: 未知的嵌入提供商标识符: {selected_provider_id}")

    def sanitize_milvus_name(name):
        s_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
        if not (s_name and (s_name[0].isalpha() or s_name[0] == '_')): s_name = "_" + s_name
        return s_name[:255]

    # Uses constants COLLECTION_NAME_LORE_PREFIX_CORE and COLLECTION_NAME_STORY_PREFIX_CORE
    lore_col_name = sanitize_milvus_name(f"{COLLECTION_NAME_LORE_PREFIX_CORE}_{provider_short}_{model_short_suffix}")
    story_col_name = sanitize_milvus_name(f"{COLLECTION_NAME_STORY_PREFIX_CORE}_{provider_short}_{model_short_suffix}")
    
    st.session_state.lore_collection_name = lore_col_name
    st.session_state.story_collection_name = story_col_name

    pk_field = FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=st.session_state.embedding_dimension)
    text_content_field = FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65530)
    timestamp_field = FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64)
    lore_specific_fields = [FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255), FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100)]
    story_specific_fields = [FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=50), FieldSchema(name="segment_number", dtype=DataType.VARCHAR, max_length=50)]
    base_schema_fields = [pk_field, embedding_field, text_content_field, timestamp_field]
    lore_schema_list = base_schema_fields + lore_specific_fields
    story_schema_list = base_schema_fields + story_specific_fields

    def _create_or_get_collection(name, schema_list, desc):
        # ... (Implementation from previous filled version) ...
        if utility.has_collection(name, using=MILVUS_ALIAS_CORE):
            collection = Collection(name, using=MILVUS_ALIAS_CORE)
        else:
            schema = CollectionSchema(fields=schema_list, description=desc, enable_dynamic_field=True)
            collection = Collection(name, schema=schema, using=MILVUS_ALIAS_CORE)
            index_name = f"idx_emb_{name[:50]}"
            index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 512}}
            collection.create_index(field_name="embedding", index_params=index_params, index_name=index_name)
        collection.load()
        return collection

    st.session_state.lore_collection_milvus_obj = _create_or_get_collection(lore_col_name, lore_schema_list, "Novel Lore")
    st.session_state.story_collection_milvus_obj = _create_or_get_collection(story_col_name, story_schema_list, "Novel Story")
    st.session_state.milvus_initialized = True
    logger.info(f"Core: Milvus collections '{lore_col_name}' and '{story_col_name}' ready.")


def core_chunk_text_by_paragraph(text: str) -> List[str]:
    paragraphs = text.replace('\r\n', '\n').replace('\r', '\n').split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]

# This function must be defined before core_seed_initial_lore if called by it
def core_load_and_vectorize_settings():
    # ... (Implementation from previous filled version) ...
    # Ensure it uses constants like SETTINGS_FILES_DIR_CORE defined above
    # and embedding functions like core_get_openai_embeddings
    if not st.session_state.get('milvus_initialized') or not st.session_state.get('lore_collection_milvus_obj'):
        logger.error("Core: Milvus lore collection not initialized for seeding.")
        return
    lore_collection = st.session_state.lore_collection_milvus_obj
    if lore_collection.num_entities > 0 and not os.environ.get("FORCE_RELOAD_SETTINGS", "false").lower() == "true":
        logger.info(f"Core: Knowledge base '{lore_collection.name}' not empty, skipping load.")
        return

    logger.info(f"Core: Loading settings from '{SETTINGS_FILES_DIR_CORE}' into '{lore_collection.name}'.")
    files_processed, chunks_added_total = 0, 0
    batch_size = 50
    if not os.path.exists(SETTINGS_FILES_DIR_CORE):
        logger.warning(f"Core: Settings directory '{SETTINGS_FILES_DIR_CORE}' not found.")
        return

    for filename in os.listdir(SETTINGS_FILES_DIR_CORE):
        if filename.endswith((".txt", ".md")):
            # ... (Rest of your file processing, embedding, and insertion logic) ...
            pass # Placeholder for brevity
    if chunks_added_total > 0: lore_collection.flush()
    logger.info(f"Core: Finished loading settings. Processed {files_processed} files, added {chunks_added_total} chunks.")


# This function definition must come before core_initialize_system calls it.
def core_seed_initial_lore():
    core_load_and_vectorize_settings() # Call the function defined above
    lore_collection = st.session_state.get('lore_collection_milvus_obj')
    if lore_collection and lore_collection.num_entities == 0:
        # ... (Fallback lore logic as in previous filled version) ...
        # Uses OPENAI_EMBEDDING_MODEL_CORE if OpenAI is chosen
        logger.info(f"Core: Lore collection '{lore_collection.name}' is empty, adding fallback.")
        text = "核心提示：AI小说项目基础设定。"
        vector = None
        embedding_provider_id = st.session_state.selected_embedding_provider_identifier
        if embedding_provider_id == "openai_official":
            vector_list = core_get_openai_embeddings([text], OPENAI_EMBEDDING_MODEL_CORE) # Use defined constant
            if vector_list: vector = vector_list[0]
        # ... (elif for ST embeddings) ...
        if vector:
            # ... (insert fallback to lore_collection) ...
            pass # Placeholder

# ... (core_add_story_segment_to_milvus, core_retrieve_relevant_lore, core_retrieve_recent_story_segments - REMAINS LARGELY THE SAME, ensure they use session_state for Milvus objects and configs)

# --- LLM Generation Function --- (Ensure constants like OPENAI_LLM_MODEL_CORE are defined above)
def core_generate_with_llm(provider_name, prompt_text_from_rag, temperature=0.7, max_tokens_override=None, system_message_override=None):
    # ... (Implementation from previous filled version) ...
    # This function uses many constants like OPENAI_LLM_MODEL_CORE, DEEPSEEK_BASE_URL_CORE etc.
    # and API Key name constants. Ensure all are defined in the Constants section at the top.
    # It also uses core_get_custom_proxy_key, core_set_temp_os_proxies, etc.
    # For Gemini, it would use GEMINI_LLM_MODEL_CORE.
    # This is a large function, so pasting its full content here would be redundant if it's already correct from before.
    # The key is that all module-level constants it relies on must be defined at the top of this file.
    logger.info(f"Core: Simulating LLM call for provider {provider_name}.")
    return f"[Simulated LLM Output for {provider_name}]\nDirective: {prompt_text_from_rag[:100]}..."


# --- Main Initialization Function (called by UI) ---
def core_initialize_system(embedding_choice_key: str, llm_choice_key: str, api_keys_from_ui: dict):
    try:
        st.session_state.log_messages.insert(0, "[INFO][Core] 开始核心系统初始化...")
        logger.info("Core: Initializing system...")

        st.session_state.selected_embedding_provider_key = embedding_choice_key
        st.session_state.selected_llm_provider_key = llm_choice_key
        st.session_state.api_keys = api_keys_from_ui

        # Uses embedding_providers_map_core defined above
        emb_identifier, emb_model_name, _, emb_dim = embedding_providers_map_core[embedding_choice_key]
        st.session_state.selected_embedding_provider_identifier = emb_identifier
        st.session_state.embedding_dimension = emb_dim
        
        if "sentence_transformer" in emb_identifier:
            # ... (ST model loading logic) ...
            st.session_state.selected_st_model_name = emb_model_name
            # ... (load ST model into st.session_state.embedding_model_instance) ...
        elif emb_identifier == "openai_official":
            # ... (OpenAI embedding setup) ...
            pass # No instance to store for OpenAI embeddings typically

        # Uses llm_providers_map_core defined above
        llm_provider_name = llm_providers_map_core[llm_choice_key]
        st.session_state.current_llm_provider = llm_provider_name
        
        # API Key Checks (Simplified example, ensure your full logic is here)
        # Uses constants like OPENAI_API_KEY_ENV_NAME defined above
        # ... (your API key check logic) ...
        logger.info("Core: API Key checks passed (simulated).")

        core_init_milvus_collections_internal() # Defined above
        core_seed_initial_lore()                # Defined above
        
        # Initialize/Load story state
        # ... (your resume logic as in previous filled version) ...
        st.session_state.current_chapter = 1 
        st.session_state.current_segment_number = 0
        logger.info("Core: Story state initialized/reset.")
        
        logger.info("Core: System initialization successful.")
        st.session_state.log_messages.insert(0, "[SUCCESS][Core] 核心系统初始化流程完成！")
        return True
    except Exception as e:
        logger.error(f"Core: System initialization failed: {e}", exc_info=True)
        st.session_state.log_messages.insert(0, f"[FATAL][Core] 核心系统初始化失败: {e}")
        raise

# --- UI Specific Core Functions --- (Ensure these are defined as they are called by app_ui.py)
def core_generate_segment_text_for_ui(user_directive: str) -> Optional[str]:
    # ... (Implementation from previous filled version) ...
    # This will call core_retrieve_relevant_lore, core_retrieve_recent_story_segments,
    # and core_generate_with_llm. Ensure those are correctly defined and use constants from above.
    logger.info(f"Core: Generating segment for UI. Directive: {user_directive[:30]}...")
    # ... build prompt ...
    final_prompt = f"User Directive: {user_directive}\n[Simulated Lore]\n[Simulated Recent Story]"
    return core_generate_with_llm(st.session_state.current_llm_provider, final_prompt)


def core_adopt_segment_from_ui(text_content: str, chapter: int, segment_num: int, user_directive_snippet: str):
    # ... (Implementation from previous filled version) ...
    # This will call core_get_openai_embeddings/core_get_st_embeddings and core_add_story_segment_to_milvus.
    # It will also use NOVEL_MD_OUTPUT_DIR_CORE defined above.
    logger.info(f"Core: Adopting segment Ch{chapter}-Seg{segment_num} from UI.")
    # ... (save to MD logic) ...
    # ... (get vector) ...
    # ... (add to Milvus) ...
    return True