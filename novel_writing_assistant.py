# novel_writing_assistant.py

# --- 1. Python 标准库和第三方库的 Import ---
import openai
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import httpx
import logging
from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection
import uuid
from datetime import datetime
import os
import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# --- 2. 加载 .env 文件 ---
load_dotenv()

# --- 3. 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 4. 全局常量和配置 ---
OPENAI_API_KEY_ENV_NAME = "OPENAI_API_KEY"; DEEPSEEK_API_KEY_ENV_NAME = "DEEPSEEK_API_KEY"; GEMINI_API_KEY_ENV_NAME = "GEMINI_API_KEY"; CUSTOM_PROXY_API_KEY_ENV_NAME = "CUSTOM_PROXY_API_KEY"
OPENAI_API_KEY = os.getenv(OPENAI_API_KEY_ENV_NAME); DEEPSEEK_API_KEY = os.getenv(DEEPSEEK_API_KEY_ENV_NAME); GEMINI_API_KEY = os.getenv(GEMINI_API_KEY_ENV_NAME); CUSTOM_PROXY_API_KEY_FROM_ENV = os.getenv(CUSTOM_PROXY_API_KEY_ENV_NAME)
GEMINI_HTTP_PROXY = os.getenv("GEMINI_HTTP_PROXY", os.getenv("GLOBAL_HTTP_PROXY")); GEMINI_HTTPS_PROXY = os.getenv("GEMINI_HTTPS_PROXY", os.getenv("GLOBAL_HTTPS_PROXY"))
DEEPSEEK_LLM_HTTP_PROXY = os.getenv("DEEPSEEK_LLM_HTTP_PROXY"); DEEPSEEK_LLM_HTTPS_PROXY = os.getenv("DEEPSEEK_LLM_HTTPS_PROXY")
CUSTOM_LLM_HTTP_PROXY = os.getenv("CUSTOM_LLM_HTTP_PROXY"); CUSTOM_LLM_HTTPS_PROXY = os.getenv("CUSTOM_LLM_HTTPS_PROXY")
OPENAI_OFFICIAL_HTTP_PROXY = os.getenv("OPENAI_OFFICIAL_HTTP_PROXY"); OPENAI_OFFICIAL_HTTPS_PROXY = os.getenv("OPENAI_OFFICIAL_HTTPS_PROXY")
CUSTOM_PROXY_BASE_URL = os.getenv("CUSTOM_PROXY_BASE_URL", "https://api.openai-next.com/v1"); HARDCODED_CUSTOM_PROXY_KEY = "sk-0rRacqNU0aY2DrOV1601DfE76a4d4a87A14f97383e25095b"
MILVUS_ALIAS = "default"; MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost"); MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"; ST_MODEL_TEXT2VEC = "shibing624/text2vec-base-chinese"; ST_MODEL_BGE_LARGE_ZH = "BAAI/bge-large-zh-v1.5"
OPENAI_LLM_MODEL = "gpt-3.5-turbo"; DEEPSEEK_LLM_MODEL = "deepseek-chat"; GEMINI_LLM_MODEL = "gemini-2.5-flash" ; CUSTOM_PROXY_LLM_MODEL = "gpt-3.5-turbo"
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
SETTINGS_FILES_DIR = "./novel_setting_files"; NOVEL_MD_OUTPUT_DIR = "./novel_markdown_chapters"
COLLECTION_NAME_LORE_PREFIX = "novel_lore_mv" ; COLLECTION_NAME_STORY_PREFIX = "novel_story_mv"

# --- 5. 全局变量声明 ---
llm_client: Optional[openai.OpenAI] = None
gemini_llm_client: Optional[genai.GenerativeModel] = None
current_llm_provider: Optional[str] = None
selected_embedding_provider_identifier: Optional[str] = None
embedding_model_instance: Optional[SentenceTransformer] = None
embedding_dimension: Optional[int] = None
lore_collection_milvus: Optional[Collection] = None
story_collection_milvus: Optional[Collection] = None
_original_os_environ_proxies: Dict[str, Optional[str]] = {}
selected_st_model_name: Optional[str] = None

# --- 6. 所有函数定义放在这里 ---
def get_custom_proxy_key():
    key = CUSTOM_PROXY_API_KEY_FROM_ENV
    if key: return key
    logger.warning(f"环境变量 CUSTOM_PROXY_API_KEY 未设置。使用硬编码Key。")
    return HARDCODED_CUSTOM_PROXY_KEY

def _set_temp_os_proxies(http_proxy: Optional[str], https_proxy: Optional[str]):
    global _original_os_environ_proxies
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    # Initialize _original_os_environ_proxies only if it's empty
    if not _original_os_environ_proxies:
        for var in proxy_vars:
            _original_os_environ_proxies[var] = os.environ.get(var)

    actions = []
    # Set HTTP_PROXY
    if http_proxy:
        if os.environ.get("HTTP_PROXY") != http_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
            actions.append(f"Set HTTP_PROXY='{http_proxy}'")
        if os.environ.get("http_proxy") != http_proxy: # some libs use lowercase
            os.environ["http_proxy"] = http_proxy
            # actions.append(f"Set http_proxy='{http_proxy}'") # Avoid duplicate log
    else:
        if "HTTP_PROXY" in os.environ:
            del os.environ["HTTP_PROXY"]
            actions.append("Del HTTP_PROXY")
        if "http_proxy" in os.environ:
            del os.environ["http_proxy"]
            # actions.append("Del http_proxy")

    # Set HTTPS_PROXY
    if https_proxy:
        if os.environ.get("HTTPS_PROXY") != https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
            actions.append(f"Set HTTPS_PROXY='{https_proxy}'")
        if os.environ.get("https_proxy") != https_proxy: # some libs use lowercase
            os.environ["https_proxy"] = https_proxy
            # actions.append(f"Set https_proxy='{https_proxy}'")
    else:
        if "HTTPS_PROXY" in os.environ:
            del os.environ["HTTPS_PROXY"]
            actions.append("Del HTTPS_PROXY")
        if "https_proxy" in os.environ:
            del os.environ["https_proxy"]
            # actions.append("Del https_proxy")

    if actions: logger.debug(f"Temp OS Proxies updated: {', '.join(actions)}")


def _restore_original_os_proxies():
    global _original_os_environ_proxies
    if not _original_os_environ_proxies:
        logger.debug("No original OS proxies to restore or already restored.")
        return
    actions = []
    for var, original_value in _original_os_environ_proxies.items():
        current_value = os.environ.get(var)
        if original_value is not None:
            if current_value != original_value:
                os.environ[var] = original_value
                actions.append(f"Restored {var}='{original_value}'")
        elif current_value is not None: # Original was None, but current is set
            del os.environ[var]
            actions.append(f"Del {var} (was not originally set)")

    if actions: logger.debug(f"Restored OS Proxies: {', '.join(actions)}")
    _original_os_environ_proxies.clear()


def _get_httpx_client_with_proxy(http_proxy_url: Optional[str], https_proxy_url: Optional[str]) -> Optional[httpx.Client]:
    proxies_for_httpx = {}
    if http_proxy_url: proxies_for_httpx["http://"] = http_proxy_url
    if https_proxy_url: proxies_for_httpx["https://"] = https_proxy_url
    if proxies_for_httpx:
        try: return httpx.Client(proxies=proxies_for_httpx, timeout=60.0) # Added timeout
        except Exception as e: logger.error(f"创建配置了代理的 httpx.Client 失败: {e}"); return None
    return httpx.Client(timeout=60.0) # Return default client if no proxies

def get_openai_embeddings(texts: List[str], model: str = OPENAI_EMBEDDING_MODEL) -> Optional[List[List[float]]]:
    if not OPENAI_API_KEY: raise ValueError("OpenAI API Key 未配置")
    temp_client = None
    try:
        _set_temp_os_proxies(OPENAI_OFFICIAL_HTTP_PROXY, OPENAI_OFFICIAL_HTTPS_PROXY)
        temp_client = _get_httpx_client_with_proxy(OPENAI_OFFICIAL_HTTP_PROXY, OPENAI_OFFICIAL_HTTPS_PROXY)
        client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=temp_client)
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e: logger.error(f"使用 OpenAI 生成嵌入失败: {e}"); return None
    finally:
        if temp_client: temp_client.close()
        _restore_original_os_proxies()

def get_st_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    global embedding_model_instance
    if embedding_model_instance is None or not isinstance(embedding_model_instance, SentenceTransformer):
        logger.error("错误：Sentence Transformer 模型实例未正确初始化。")
        return None
    try:
        # logger.debug(f"使用已加载的 ST 模型生成 {len(texts)}条文本的嵌入。")
        embeddings = embedding_model_instance.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist() # Added normalize_embeddings
        return embeddings
    except Exception as e: logger.error(f"使用 Sentence Transformer 生成嵌入失败: {e}"); return None

def initialize_embedding_function(provider_identifier_arg):
    global selected_embedding_provider_identifier
    selected_embedding_provider_identifier = provider_identifier_arg
    logger.info(f"全局 selected_embedding_provider_identifier 设置为: {selected_embedding_provider_identifier.upper()}")

def init_milvus_collections():
    global lore_collection_milvus, story_collection_milvus, embedding_dimension
    global selected_embedding_provider_identifier, selected_st_model_name
    if not embedding_dimension: raise ValueError("Embedding dimension 未设置")
    if not selected_embedding_provider_identifier: raise ValueError("selected_embedding_provider_identifier 未设置")
    try:
        if not connections.has_connection(MILVUS_ALIAS):
             connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
             logger.info(f"成功连接到 Milvus ({MILVUS_HOST}:{MILVUS_PORT})")
        else:
             logger.info(f"已存在到 Milvus ({MILVUS_HOST}:{MILVUS_PORT}) 的连接。")
    except Exception as e: logger.error(f"连接 Milvus 失败: {e}"); raise

    provider_short, model_short_suffix = "", ""
    if selected_embedding_provider_identifier == "openai_official":
        provider_short, model_short_suffix = "oai", OPENAI_EMBEDDING_MODEL.split('-')[-1][:6]
    elif selected_embedding_provider_identifier.startswith("sentence_transformer_"):
        provider_short = selected_embedding_provider_identifier.replace("sentence_transformer_", "st")[:6]
        if selected_st_model_name:
            model_short_suffix = selected_st_model_name.split('/')[-1].replace('-', '_').replace('.', '_')[:10] # Replace . for Milvus
        else:
            model_short_suffix = "unknownst"

    def sanitize_milvus_name(name):
        s_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
        # Milvus collection names must start with a letter or underscore
        if not (s_name[0].isalpha() or s_name[0] == '_'):
            s_name = "_" + s_name
        return s_name[:255]

    lore_col_name = sanitize_milvus_name(f"{COLLECTION_NAME_LORE_PREFIX}_{provider_short}_{model_short_suffix}")
    story_col_name = sanitize_milvus_name(f"{COLLECTION_NAME_STORY_PREFIX}_{provider_short}_{model_short_suffix}")
    logger.info(f"Milvus Lore Collection Name: {lore_col_name}")
    logger.info(f"Milvus Story Collection Name: {story_col_name}")

    # Define fields (consistent for both collections for simplicity, specific fields can be added if needed)
    # Increased max_length for doc_id to accommodate longer UUIDs or descriptive IDs
    pk_field = FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dimension)
    text_content_field = FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65530) # Max for VARCHAR in many DBs
    timestamp_field = FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64) # ISO 8601 with timezone can be long
    
    # Fields specific to lore
    source_file_field = FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255)
    document_type_field = FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100)
    
    # Fields specific to story
    chapter_field = FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=50) # Can be string like "外传1"
    segment_number_field = FieldSchema(name="segment_number", dtype=DataType.VARCHAR, max_length=50) # Can be string like "番外1.2"

    # Inside init_milvus_collections function

    def create_or_get_milvus_collection(collection_name_str, schema_fields_list, description_str):
        if utility.has_collection(collection_name_str, using=MILVUS_ALIAS):
            logger.info(f"Milvus 集合 '{collection_name_str}' 已存在，获取...")
            collection = Collection(collection_name_str, using=MILVUS_ALIAS)
        else:
            logger.info(f"Milvus 集合 '{collection_name_str}' 不存在，创建...")
            schema = CollectionSchema(fields=schema_fields_list, description=description_str, enable_dynamic_field=True)
            collection = Collection(collection_name_str, schema=schema, using=MILVUS_ALIAS)
            logger.info(f"Milvus 集合 '{collection_name_str}' 创建成功。")
            
            has_index_on_embedding = False
            if collection.has_index():
                for index in collection.indexes:
                    if index.field_name == "embedding":
                        has_index_on_embedding = True
                        logger.info(f"集合 '{collection_name_str}' 的 'embedding' 字段已存在索引: {index.index_name}")
                        break
            
            if not has_index_on_embedding: # Create index only if embedding field doesn't have one
                logger.info(f"为 '{collection_name_str}' 的 'embedding' 字段创建 HNSW 索引...")
                index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 512}}
                try:
                    collection.create_index(field_name="embedding", index_params=index_params, index_name=f"idx_emb_{collection_name_str[:50]}") # Added index_name
                    logger.info(f"为 '{collection_name_str}' 的 'embedding' 创建了 HNSW 索引。")
                except Exception as e_create_idx:
                    logger.error(f"为 '{collection_name_str}' 创建索引失败: {e_create_idx}")
            # else: # Already logged above if index exists
            #     logger.info(f"集合 '{collection_name_str}' 已有索引或 'embedding' 字段索引已存在。")

        # --- MODIFICATION START ---
        # Always attempt to load. If already loaded, this is usually a no-op or quick check.
        # The `is_loaded` attribute is not reliable across all PyMilvus versions.
        try:
            # Check current replicas before loading (optional, for info)
            # release_info = utility.get_query_segment_info(collection_name_str, using=MILVUS_ALIAS) # This API might change
            # logger.debug(f"Query segment info for '{collection_name_str}': {release_info}")

            logger.info(f"尝试加载集合 '{collection_name_str}'...")
            collection.load() # This will load the collection to memory for searching.
            logger.info(f"集合 '{collection_name_str}' 已成功加载 (或已加载)。")
        except Exception as e_load:
            logger.error(f"加载集合 '{collection_name_str}' 失败: {e_load}", exc_info=True)
            # Depending on the error, you might want to raise it or handle it.
            # For now, we log and continue, but search/query might fail later.
        # --- MODIFICATION END ---
        return collection

    lore_schema_fields = [pk_field, embedding_field, text_content_field, timestamp_field, source_file_field, document_type_field]
    story_schema_fields = [pk_field, embedding_field, text_content_field, timestamp_field, chapter_field, segment_number_field]
    
    lore_collection_milvus = create_or_get_milvus_collection(lore_col_name, lore_schema_fields, "Novel Lore Context Collection")
    story_collection_milvus = create_or_get_milvus_collection(story_col_name, story_schema_fields, "Novel Story Segments Collection")


def chunk_text_by_paragraph(text: str) -> List[str]:
    # Handles various newline combinations and empty paragraphs
    paragraphs = text.replace('\r\n', '\n').replace('\r', '\n').split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]


def load_and_vectorize_settings_from_files(directory_path):
    global selected_embedding_provider_identifier
    if not lore_collection_milvus:
        logger.error("错误：Lore Milvus Collection 未初始化。")
        return
    # Simplified reload check: if FORCE_RELOAD_SETTINGS is true, reload. Otherwise, skip if not empty.
    if lore_collection_milvus.num_entities > 0 and not os.environ.get("FORCE_RELOAD_SETTINGS", "false").lower() == "true":
        logger.info(f"知识库 '{lore_collection_milvus.name}' 已包含 {lore_collection_milvus.num_entities} 条设定。跳过加载。设置 FORCE_RELOAD_SETTINGS=true 以强制重新加载。")
        return

    logger.info(f"\n--- 从目录 '{directory_path}' 加载设定文件到 Milvus (知识库: {lore_collection_milvus.name}) ---")
    if not os.path.exists(directory_path):
        logger.error(f"错误：设定文件目录 '{directory_path}' 不存在。")
        return

    files_processed, chunks_added_to_milvus_session = 0, 0
    batch_size = 100 # Process N chunks at a time for embedding and insertion

    for filename in os.listdir(directory_path):
        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(directory_path, filename)
            logger.info(f"  处理文件: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
                text_chunks_from_file = chunk_text_by_paragraph(content)
                if not text_chunks_from_file:
                    logger.info(f"    文件 '{filename}' 内容为空或无有效段落。")
                    continue
                
                file_type_tag = os.path.splitext(filename)[0].lower().replace(' ', '_').replace('-', '_')
                
                for i in range(0, len(text_chunks_from_file), batch_size):
                    batch_texts = text_chunks_from_file[i:i+batch_size]
                    vectors = None
                    if selected_embedding_provider_identifier == "openai_official":
                        vectors = get_openai_embeddings(batch_texts)
                    elif selected_embedding_provider_identifier.startswith("sentence_transformer_"):
                        vectors = get_st_embeddings(batch_texts)
                    
                    if not vectors or len(vectors) != len(batch_texts):
                        logger.error(f"未能为 '{filename}' 的部分块生成有效向量。跳过此批次。")
                        continue

                    entities_to_insert = []
                    for j, chunk_text in enumerate(batch_texts):
                        # More robust ID generation
                        chunk_hash = uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{i+j}_{chunk_text[:100]}").hex[:16]
                        doc_id = f"setting_{file_type_tag}_{chunk_hash}"
                        
                        entities_to_insert.append({
                            "doc_id": doc_id, "embedding": vectors[j], "text_content": chunk_text,
                            "timestamp": datetime.now().isoformat(), "source_file": filename,
                            "document_type": file_type_tag
                        })
                    
                    if entities_to_insert:
                        try:
                            insert_result = lore_collection_milvus.insert(entities_to_insert)
                            chunks_added_to_milvus_session += len(insert_result.primary_keys)
                            logger.info(f"    成功将 {len(insert_result.primary_keys)} 个块从 '{filename}' (批次 {i//batch_size + 1}) 插入 Milvus。")
                        except Exception as e_insert:
                            logger.error(f"    插入 Milvus 失败 (文件: {filename}, 批次: {i//batch_size + 1}): {e_insert}")

                files_processed += 1
            except Exception as e_file_proc:
                logger.error(f"    处理文件 '{filename}' 时发生错误: {e_file_proc}")
    
    if files_processed > 0:
        lore_collection_milvus.flush() # Ensure data is written to disk
        logger.info(f"--- Milvus: 处理 {files_processed} 文件，本会话共添加 {chunks_added_to_milvus_session} 片段到 '{lore_collection_milvus.name}' ---")


def seed_initial_lore():
    load_and_vectorize_settings_from_files(SETTINGS_FILES_DIR)
    # Check if the fallback lore is truly needed
    if lore_collection_milvus and lore_collection_milvus.num_entities == 0:
        logger.info(f"Milvus 知识库 '{lore_collection_milvus.name}' 为空，添加基础后备设定...")
        text = "核心提示：AI小说项目。这是一个用于生成长篇小说的辅助工具，利用大语言模型和向量数据库进行创作。当前没有其他背景设定被加载。"
        vector = None
        if selected_embedding_provider_identifier == "openai_official":
            vector = get_openai_embeddings([text])[0] if get_openai_embeddings([text]) else None
        elif selected_embedding_provider_identifier.startswith("sentence_transformer_"):
            vector = get_st_embeddings([text])[0] if get_st_embeddings([text]) else None
        
        if vector:
            doc_id = "projmeta_fallback_001"
            entity = {
                "doc_id": doc_id, "embedding": vector, "text_content": text,
                "timestamp": datetime.now().isoformat(), "document_type": "project_meta_fallback",
                "source_file": "internal_default"
            }
            try:
                lore_collection_milvus.insert([entity])
                lore_collection_milvus.flush()
                logger.info(f"已添加基础后备设定到 Milvus (ID: {doc_id})。")
            except Exception as e_insert_fallback:
                 logger.error(f"添加后备设定到 Milvus 失败: {e_insert_fallback}")


def add_story_segment_to_milvus(text_content, chapter, segment_number, vector):
    if not story_collection_milvus:
        logger.error("错误: Story Milvus Collection 未初始化。")
        return None
    # Use a more robust ID for story segments as well
    segment_hash = uuid.uuid5(uuid.NAMESPACE_DNS, f"ch{chapter}_seg{segment_number}_{text_content[:100]}").hex[:16]
    doc_id = f"ch{chapter}_seg{segment_number}_{segment_hash}"

    entity = {
        "doc_id": doc_id, "embedding": vector, "text_content": text_content,
        "timestamp": datetime.now().isoformat(),
        "chapter": str(chapter), "segment_number": str(segment_number)
    }
    try:
        mr = story_collection_milvus.insert([entity])
        story_collection_milvus.flush() # Ensure write for subsequent resume logic
        logger.info(f"添加故事片段到 Milvus: '{doc_id}', PKs: {mr.primary_keys}")
        return doc_id
    except Exception as e:
        logger.error(f"添加故事片段到 Milvus 失败: {e}")
        return None

def retrieve_relevant_lore_from_milvus(query_text: str, n_results: int = 3) -> List[str]:
    if not lore_collection_milvus or not query_text: return []
    global selected_embedding_provider_identifier
    query_vector_list = None # Expecting a list of vectors
    if selected_embedding_provider_identifier == "openai_official":
        query_vector_list = get_openai_embeddings([query_text])
    elif selected_embedding_provider_identifier.startswith("sentence_transformer_"):
        query_vector_list = get_st_embeddings([query_text])
    
    if not query_vector_list or not query_vector_list[0]:
        logger.error("无法为查询文本生成向量。")
        return []
    
    query_vector = query_vector_list[0] # Use the first (and only) vector

    # Search parameters: ef should be between top_k and Nlist (if using IVF_FLAT, etc.)
    # For HNSW, ef should be >= n_results. Higher ef = better recall, slower search.
    search_params = {"metric_type": "L2", "params": {"ef": max(64, n_results * 8)}} # Dynamic ef
    
    try:
        results = lore_collection_milvus.search(
            data=[query_vector], anns_field="embedding", param=search_params,
            limit=n_results, output_fields=["text_content", "document_type", "source_file"],
            consistency_level="Bounded" # Bounded staleness for faster search if acceptable
        )
        retrieved_items = []
        for hits_for_one_query in results: # results is a list of lists of Hits
            for hit in hits_for_one_query:
                if hit.entity and "text_content" in hit.entity:
                    # More structured context
                    item_info = f"[知识库内容 - 相关度: {hit.distance:.4f} | 来源: {hit.entity.get('source_file', '未知')} ({hit.entity.get('document_type', '未知')})]"
                    item_text = f"{item_info}\n{hit.entity.get('text_content')}"
                    retrieved_items.append(item_text)
        return retrieved_items

    except Exception as e:
        logger.error(f"从 Milvus 检索知识片段失败: {e}", exc_info=True)
        return []

def retrieve_recent_story_segments_from_milvus(n_results: int = 1) -> List[str]:
    if not story_collection_milvus or story_collection_milvus.num_entities == 0:
        logger.debug("故事库为空或未初始化，无法检索最近片段。")
        return []
    try:
        # Fetch a bit more to ensure robust sorting by timestamp string, then take top N
        fetch_limit = max(n_results + 5, 10) # Fetch a few more for sorting
        
        # Query expression: Milvus requires a valid expression. "pk_field_name >= 0" or similar for numeric pk.
        # For VARCHAR PKs, a simple "doc_id != ''" works.
        query_expr = "doc_id != \"\""

        results = story_collection_milvus.query(
            expr=query_expr,
            output_fields=["text_content", "timestamp", "chapter", "segment_number"],
            limit=fetch_limit, # Milvus applies limit *before* ordering in some versions/setups if not using specific sort features
            consistency_level="Strong" # Crucial for getting the absolute latest for resume logic
        )
        
        if not results:
            logger.debug("故事库查询未返回任何结果。")
            return []
            
        # Sort in Python as Milvus string field sorting in query can be tricky
        # Timestamps are ISO strings, so lexicographical sort works.
        sorted_results = sorted(results, key=lambda x: x.get('timestamp', '0'), reverse=True)
        
        # Construct a more informative string for the recent story context
        final_segments = []
        for item in sorted_results[:n_results]:
            segment_info = f"[先前故事片段 - 章节 {item.get('chapter', '?')} 片段 {item.get('segment_number', '?')}]"
            final_segments.append(f"{segment_info}\n{item['text_content']}")
        
        return final_segments # Returns a list of formatted strings
    except Exception as e:
        logger.error(f"从 Milvus 检索最近故事片段失败: {e}", exc_info=True)
        return []

# --- LLM 调用函数定义 (generate_with_llm) ---
def generate_with_llm(provider, prompt_text_from_rag, temperature=0.7, max_tokens=1500, system_message_override=None):
    global llm_client, gemini_llm_client, current_llm_provider
    current_http_proxy_for_call, current_https_proxy_for_call = None, None
    if provider == "openai_official": current_http_proxy_for_call, current_https_proxy_for_call = OPENAI_OFFICIAL_HTTP_PROXY, OPENAI_OFFICIAL_HTTPS_PROXY
    elif provider == "deepseek": current_http_proxy_for_call, current_https_proxy_for_call = DEEPSEEK_LLM_HTTP_PROXY, DEEPSEEK_LLM_HTTPS_PROXY
    elif provider == "custom_proxy_llm": current_http_proxy_for_call, current_https_proxy_for_call = CUSTOM_LLM_HTTP_PROXY, CUSTOM_LLM_HTTPS_PROXY
    elif provider == "gemini": current_http_proxy_for_call, current_https_proxy_for_call = GEMINI_HTTP_PROXY, GEMINI_HTTPS_PROXY

    _set_temp_os_proxies(current_http_proxy_for_call, current_https_proxy_for_call)
    # logger.info(f"generate_with_llm for {provider}: Temp OS Proxy set to HTTP='{os.environ.get('HTTP_PROXY')}', HTTPS='{os.environ.get('HTTPS_PROXY')}'") # Logged in _set_temp_os_proxies

    default_system_message = "你是一位富有创意的小说作家助手，擅长撰写情节连贯、情感丰富、符合用户指令的中文网络小说。请严格遵循用户的具体指令进行创作。"
    final_system_message = system_message_override if system_message_override else default_system_message
    
    httpx_client_for_llm_call = None # For OpenAI compatible clients

    try:
        logger.info(f"\n--- Sending Prompt to {provider.upper()} LLM (Max Tokens: {max_tokens}) ---")
        if provider == "openai_official" or provider == "deepseek" or provider == "custom_proxy_llm":
            if current_llm_provider != provider or not llm_client: # Re-initialize if provider changed or client not set
                api_key_value, base_url_to_use = "", None
                # Use current OS proxies for the SDK client init
                httpx_client_for_llm_call = _get_httpx_client_with_proxy(os.environ.get("HTTP_PROXY"), os.environ.get("HTTPS_PROXY"))

                if provider == "openai_official": api_key_value = OPENAI_API_KEY
                elif provider == "deepseek": api_key_value, base_url_to_use = DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
                elif provider == "custom_proxy_llm": api_key_value, base_url_to_use = get_custom_proxy_key(), CUSTOM_PROXY_BASE_URL
                
                if not api_key_value: raise ValueError(f"API Key for {provider} not loaded.")
                
                llm_client = openai.OpenAI(api_key=api_key_value, base_url=base_url_to_use, http_client=httpx_client_for_llm_call)
                current_llm_provider = provider
                logger.info(f"LLM Client for {provider.upper()} initialized.")

            model_name_to_call = {"openai_official": OPENAI_LLM_MODEL, "deepseek": DEEPSEEK_LLM_MODEL, "custom_proxy_llm": CUSTOM_PROXY_LLM_MODEL}.get(provider)
            logger.info(f"--- OpenAI-compatible Prompt ({provider.upper()}) 发送中 (用户内容长度: {len(prompt_text_from_rag)} chars) ---")
            
            response = llm_client.chat.completions.create(
                model=model_name_to_call,
                messages=[
                    {"role": "system", "content": final_system_message},
                    {"role": "user", "content": prompt_text_from_rag}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()

        elif provider == "gemini":
            if current_llm_provider != provider or not gemini_llm_client:
                if not GEMINI_API_KEY: raise ValueError(f"GEMINI_API_KEY 未加载!")
                # Gemini SDK respects HTTP_PROXY and HTTPS_PROXY environment variables.
                # genai.configure() should be called only once ideally.
                try: # Check if already configured
                    genai.get_model(GEMINI_LLM_MODEL) # Throws if not configured
                except Exception:
                     genai.configure(api_key=GEMINI_API_KEY, client_options={"api_endpoint": os.getenv("GEMINI_API_ENDPOINT")})
                
                gemini_llm_client = genai.GenerativeModel(
                   GEMINI_LLM_MODEL, # Removed system_instruction from constructor
                   generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
                )
                current_llm_provider = provider
                logger.info(f"LLM Client for {provider.upper()} initialized with model {GEMINI_LLM_MODEL}, max_tokens={max_tokens}.")

            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            
            # Prepend system message for Gemini if not handled by a specific constructor param
            # This is a common way for models that take a single block of text
            prompt_with_system_message_for_gemini = f"{final_system_message}\n\n---\n\n{prompt_text_from_rag}"

            logger.info(f"--- Gemini Prompt 发送中 (总长度: {len(prompt_with_system_message_for_gemini)} chars) ---")
            # For debugging short prompts:
            # if len(prompt_with_system_message_for_gemini) < 3000: 
            #     logger.debug(f"DEBUG PROMPT for {provider.upper()}:\n{prompt_with_system_message_for_gemini}\n")

            if gemini_llm_client:
                try:
                    token_count_response = gemini_llm_client.count_tokens(prompt_with_system_message_for_gemini)
                    logger.info(f"--- Gemini Total Prompt Token 计数: {token_count_response.total_tokens} tokens ---")
                except Exception as e_count:
                    logger.warning(f"计算 Gemini Prompt Tokens 失败: {e_count}")

                response = gemini_llm_client.generate_content(prompt_with_system_message_for_gemini, safety_settings=safety_settings)
                generated_text_output = None

                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.warning(f"  Gemini Prompt 被阻止: {response.prompt_feedback.block_reason}")
                # Check new Gemini API response structure (candidates list)
                elif not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                     logger.warning("  Gemini 返回内容结构不符合预期 (candidates, content, or parts为空)。")
                     if hasattr(response, 'candidates') and response.candidates:
                        for i, cand in enumerate(response.candidates):
                            logger.warning(f"  候选 {i} 完成原因: {cand.finish_reason if hasattr(cand, 'finish_reason') else 'N/A'}")
                     else: logger.warning("  响应中也没有候选内容 (candidates)。")
                else:
                    try: # Try the convenient .text accessor first
                        generated_text_output = response.text.strip()
                        if not generated_text_output: logger.warning("  Gemini 最终生成的文本 (via .text) 为空。")
                    except (ValueError, AttributeError) as ve: # .text might fail if content is blocked or empty
                        logger.warning(f"  Gemini response.text 访问失败: {ve}. 尝试直接从 parts 组装。")
                        # Fallback to manually joining parts
                        try:
                            generated_text_output = "".join(part.text for part in response.candidates[0].content.parts).strip()
                            if not generated_text_output: logger.warning("  从 parts 组装后 Gemini 生成的文本仍为空。")
                        except Exception as e_parts:
                            logger.error(f"  从 parts 组装 Gemini 文本失败: {e_parts}")
                return generated_text_output
            else:
                logger.error("Gemini LLM client 未初始化。"); return None
        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

    except openai.RateLimitError as e: logger.error(f"{provider.upper()} API 速率限制: {e}. 等待后重试..."); time.sleep(20); return generate_with_llm(provider, prompt_text_from_rag, temperature, max_tokens, system_message_override)
    except (genai.types.generation_types.BlockedPromptException, genai.types.generation_types.StopCandidateException, genai.types.generation_types.BrokenResponseError, google_exceptions.RetryError, google_exceptions.ServiceUnavailable) as e:
        logger.error(f"Gemini 生成内容时发生问题或服务不可用: {e}"); return None
    except ValueError as e: logger.error(f"LLM 调用配置错误 ({provider.upper()}): {e}"); return None
    except Exception as e: logger.error(f"调用 {provider.upper()} LLM 时发生未知核心错误: {e}", exc_info=True); return None
    finally:
        if httpx_client_for_llm_call: httpx_client_for_llm_call.close() # Close the httpx client if it was created
        _restore_original_os_proxies()
        # logger.info(f"  已在 generate_with_llm for {provider} 后恢复原始代理。") # Logged in _restore_original_os_proxies

# --- 主写作循环定义 (writing_loop) ---
def writing_loop(current_chosen_llm_provider_arg):
    global llm_client, gemini_llm_client, current_llm_provider
    global lore_collection_milvus, story_collection_milvus 

    if current_llm_provider != current_chosen_llm_provider_arg:
        llm_client = None
        gemini_llm_client = None
        current_llm_provider = current_chosen_llm_provider_arg
        logger.info(f"LLM provider changed to {current_chosen_llm_provider_arg}. Clients reset.")

    current_chapter, current_segment_number = 1, 0

    if story_collection_milvus and story_collection_milvus.num_entities > 0:
        logger.info(f"故事库 '{story_collection_milvus.name}' 已包含 {story_collection_milvus.num_entities} 条数据。")
        try:
            # Query for the most recent segment based on timestamp
            # Milvus Python SDK v2.x.x query does not support direct sorting on non-primary key fields in the query call itself for now.
            # We fetch a slightly larger set and sort in Python.
            latest_segments_query_results = story_collection_milvus.query(
                expr="doc_id != \"\"", # Fetch all, then sort
                output_fields=["chapter", "segment_number", "timestamp"],
                limit=10, # Fetch last 10 to be safe for sorting
                consistency_level="Strong" # Important for resume logic
            )
            if latest_segments_query_results:
                # Sort by timestamp string (ISO format ensures correct lexicographical sorting)
                sorted_segments = sorted(latest_segments_query_results, key=lambda x: x.get('timestamp', '0'), reverse=True)
                if sorted_segments:
                    last_segment_meta = sorted_segments[0]
                    try:
                        last_chapter = int(last_segment_meta.get('chapter', '1'))
                        last_segment_num = int(last_segment_meta.get('segment_number', '0'))
                        
                        if last_chapter > 0 and last_segment_num > 0 : # Ensure valid numbers
                            resume_choice = input(f"检测到上次写作到 章节 {last_chapter}, 片段 {last_segment_num}。是否继续？(y/n，n则从新章节开始): ").lower()
                            if resume_choice == 'y':
                                current_chapter = last_chapter
                                current_segment_number = last_segment_num
                                logger.info(f"从章节 {current_chapter}, 片段 {current_segment_number} 后继续写作。")
                            else:
                                current_chapter = last_chapter + 1
                                current_segment_number = 0
                                logger.info(f"从新章节 {current_chapter} 开始写作。")
                        else:
                            logger.info("上次片段元数据无效，从新开始。")
                    except ValueError:
                        logger.error("解析上次章节/片段号失败，从新开始。")
                        current_chapter, current_segment_number = 1, 0
            else:
                logger.info("故事库中无历史片段，从新开始。")
        except Exception as e_resume:
            logger.error(f"尝试恢复上次写作状态失败: {e_resume}。将从新开始。", exc_info=True)
            current_chapter, current_segment_number = 1, 0


    logger.info(f"\n\n=== 小说写作助手启动 (LLM: {current_chosen_llm_provider_arg.upper()}, Embedding Provider: {selected_embedding_provider_identifier.upper()}) ===")
    if not os.path.exists(NOVEL_MD_OUTPUT_DIR):
        try:
            os.makedirs(NOVEL_MD_OUTPUT_DIR)
            logger.info(f"已创建小说输出目录: {NOVEL_MD_OUTPUT_DIR}")
        except OSError as e:
            logger.error(f"创建小说输出目录 {NOVEL_MD_OUTPUT_DIR} 失败: {e}")
            return # Exit if cannot create output dir
            
    logger.info(f"知识库 (Milvus): {lore_collection_milvus.name if lore_collection_milvus else 'N/A'}")
    logger.info(f"故事库 (Milvus): {story_collection_milvus.name if story_collection_milvus else 'N/A'}")
    print("输入 'quit' 退出写作。")

    max_tokens_for_llm = int(os.getenv("MAX_TOKENS_PER_LLM_CALL", "7800")) # Configurable max_tokens

    while True:
        print(f"\n--- 当前章节: {current_chapter}, 计划生成片段号: {current_segment_number + 1} ---")
        
        user_directive_lines = []
        print("请输入当前场景的写作指令/概要 (多行输入，以空行结束):")
        while True:
            line = input("> ")
            if not line.strip(): # Empty line signifies end of input
                break
            user_directive_lines.append(line)
        user_directive = "\n".join(user_directive_lines)

        if user_directive.lower() == 'quit': break
        if not user_directive.strip(): logger.warning("指令不能为空。"); continue

        # Input length check
        MAX_DIRECTIVE_LENGTH = 15000 # Increased, as detailed prompts can be long
        if len(user_directive) > MAX_DIRECTIVE_LENGTH:
            confirm_long_input = input(f"警告：您输入的指令长度 ({len(user_directive)}字符) 似乎过长，可能包含非预期内容。是否继续？(y/n): ").lower()
            if confirm_long_input != 'y':
                logger.info("用户取消了过长的指令输入。")
                continue

        relevant_lore_query = f"关于以下指令的核心人物、世界设定、相关背景和写作风格参考: '{user_directive[:200]}...'"
        retrieved_lore_list = retrieve_relevant_lore_from_milvus(relevant_lore_query, n_results=3) # More lore
        retrieved_lore_text = "\n\n---\n\n".join(retrieved_lore_list) if retrieved_lore_list else "无特定相关的背景知识补充。"

        num_recent_segments_to_fetch = 2
        recent_segments_data = retrieve_recent_story_segments_from_milvus(n_results=num_recent_segments_to_fetch)
        
        recent_story_text = ""
        if recent_segments_data: # This is now a list of formatted strings
            recent_story_text = "\n\n---\n\n".join(reversed(recent_segments_data)) # [older, newest]
        else:
            recent_story_text = "这是故事的开端，尚无先前的故事片段。"
        
        # logger.debug(f"最近的故事进展送入Prompt (前500字符):\n----\n{recent_story_text[:500]}...\n----")
        # logger.debug(f"检索到的知识送入Prompt (前500字符):\n----\n{retrieved_lore_text[:500]}...\n----")

        contextual_emotional_bridge = ""
        # Keywords can be expanded
        trigger_keywords_previous_event = ["退婚", "羞辱", "战斗结束", "重大变故", "牺牲", "惨败", "失去", "背叛"]
        trigger_keywords_current_directive_followup = ["回家", "安慰", "之后", "醒来", "疗伤", "反思", "继承", "复仇计划", "振作"]

        if recent_story_text != "这是故事的开端，尚无先前的故事片段。": # Only apply bridge if there's prior story
            previous_event_detected = any(keyword in recent_story_text.lower() for keyword in trigger_keywords_previous_event)
            current_directive_is_followup = any(keyword in user_directive.lower() for keyword in trigger_keywords_current_directive_followup)

            if previous_event_detected and current_directive_is_followup:
                extracted_event_hint = "先前发生的冲突、对话或重大决定" # Generic
                # Try to be more specific
                if any(k in recent_story_text.lower() for k in ["退婚", "羞辱"]):
                     extracted_event_hint = "先前苏家当众退婚带来的羞辱和主角的复仇誓言"
                elif any(k in recent_story_text.lower() for k in ["战斗结束", "惨败"]):
                    extracted_event_hint = "先前的激烈战斗及其结果"
                
                contextual_emotional_bridge = (
                    f"**重要情境回顾与情感指引**：\n"
                    f"角色可能刚刚经历了 {extracted_event_hint}。当前场景应是该事件的延续。\n"
                    f"请确保新生成内容中的人物情绪、对话和行为逻辑，都与先前发生的事件及其带来的情感冲击（如愤怒、不甘、悲伤、决心等）高度一致且自然承接。\n"
                    f"避免出现与先前事件及其后续情感状态不符的突兀转变或遗忘。"
                )
            elif previous_event_detected:
                 contextual_emotional_bridge = (
                        "**情境回顾提示**：\n"
                        "先前似乎发生了重要事件或情绪转折。请确保新内容与这些潜在的上下文在逻辑和情感上有所关联或自然发展。"
                    )
        # if contextual_emotional_bridge: logger.debug(f"应用了上下文情感桥梁:\n----\n{contextual_emotional_bridge}\n----")

        prompt_parts = [
            "---参考背景知识与设定（请结合这些信息进行创作）---", retrieved_lore_text,
            "---必须严格承接的先前故事情节（如果存在）---", recent_story_text,
            (contextual_emotional_bridge if contextual_emotional_bridge else ""),
            "---当前核心写作任务 (请严格从“先前故事情节”结尾处继续（如果“先前故事情节”不是开端），并高度重视任何“情境回顾与情感指引”中的信息，以确保情节和情感的无缝衔接。请全力完成用户的具体写作指令。如果用户指令中包含对章节名、爽点、钩子、情感线、篇幅引导等创作要求，请尽力满足。)---",
            f"用户具体写作指令如下：\n{user_directive}", # User directive is the primary instruction
            "---请基于以上所有信息，直接开始撰写故事正文（不要重复指令或做额外解释）：---"
        ]
        initial_prompt_for_llm = "\n\n".join(filter(None, prompt_parts))
        
        logger.info(f"\n正在生成初始片段 (总Prompt长度估算: {len(initial_prompt_for_llm)} chars)...")
        current_segment_text = generate_with_llm(current_chosen_llm_provider_arg, initial_prompt_for_llm, max_tokens=max_tokens_for_llm)

        segment_generation_complete = False
        final_user_choice_for_segment = None # To store 'y' or 'n' after iterations

        if current_segment_text:
            while not segment_generation_complete:
                print("\n--- AI 生成的故事片段 ---")
                print(current_segment_text)
                print(f"--- (当前片段字数: {len(current_segment_text)}) ---")
                print("------------------------")

                action = input("操作：[y]采纳此片段, [n]丢弃此片段并为本片段号重写指令, "
                               "[r]要求AI重写此片段(用相同指令), [e]要求AI扩写/补充细节, [q]退出写作: ").lower()

                if action == 'q': 
                    current_segment_text = None; final_user_choice_for_segment = 'n'; break # Treat as quit
                if action == 'y':
                    final_user_choice_for_segment = 'y'
                    segment_generation_complete = True
                elif action == 'n':
                    logger.info("用户选择丢弃当前片段并为本片段号重新输入指令。")
                    current_segment_text = None 
                    final_user_choice_for_segment = 'n' # Mark as not adopted
                    segment_generation_complete = True # Break this inner loop to re-enter outer loop for new directive
                elif action == 'r':
                    logger.info("要求AI基于原始指令重写此片段...")
                    current_segment_text = generate_with_llm(current_chosen_llm_provider_arg, initial_prompt_for_llm, max_tokens=max_tokens_for_llm)
                    if not current_segment_text:
                        logger.warning("重写失败，片段内容为空。")
                        # Keep current_segment_text as None, loop will ask again or user can choose 'n'
                elif action == 'e':
                    extra_directive_lines = []
                    print("请输入对AI扩写/补充的具体要求 (多行输入，以空行结束):")
                    while True:
                        line = input("补充指令> ")
                        if not line.strip(): break
                        extra_directive_lines.append(line)
                    extra_directive = "\n".join(extra_directive_lines)

                    if not extra_directive.strip():
                        logger.warning("补充要求不能为空。")
                        continue 

                    continuation_prompt_parts = [
                        "---先前已生成内容如下---", current_segment_text,
                        "---当前任务：补充与扩展---", 
                        "请严格按照以下“补充与扩展指令”，在不改变“先前已生成内容”核心情节的前提下，进行自然的补充、细节扩展或内容延续。直接开始撰写需要补充或扩展的部分，使其能与原文流畅衔接，并力求达到指令中的篇幅或细节要求。",
                        f"用户补充与扩展指令：\n{extra_directive}"
                    ]
                    continuation_prompt = "\n\n".join(continuation_prompt_parts)
                    logger.info(f"\n正在生成补充内容 (补充Prompt长度估算: {len(continuation_prompt)} chars)...")
                    additional_text = generate_with_llm(
                        current_chosen_llm_provider_arg, 
                        continuation_prompt, 
                        max_tokens=max_tokens_for_llm, # Allow full token usage for expansion
                        system_message_override="你是一位优秀的小说作家，正在基于用户提供的已生成内容和新的补充指令进行内容的补充和扩展。请确保补充内容与原文自然融合。"
                    )
                    
                    if additional_text and additional_text.strip():
                        logger.info(f"\n--- AI 补充/扩展的内容 (字数: {len(additional_text)}) ---"); print(additional_text); print("------------------------")
                        merge_choice = input("如何合并补充内容? ([a]追加到末尾(默认), [p]添加到开头, [r]替换整个片段, [i]尝试智能插入提示(手动操作), [d]不合并): ").lower()
                        if merge_choice == 'a' or not merge_choice: # Default to append
                            current_segment_text += "\n\n" + additional_text.strip()
                            logger.info("补充内容已追加到原片段末尾。")
                        elif merge_choice == 'p':
                            current_segment_text = additional_text.strip() + "\n\n" + current_segment_text
                            logger.info("补充内容已添加到原片段开头。")
                        elif merge_choice == 'r':
                            current_segment_text = additional_text.strip()
                            logger.info("原片段已用补充内容替换。")
                        elif merge_choice == 'i':
                            print("智能插入提示：请复制上方AI补充内容，并在您认为合适的地方手动将其插入到“AI生成的故事片段”中。完成编辑后，在下一个操作提示中选择[y]采纳最终版本。")
                            # No actual merge here, user does it mentally or externally
                        else: # 'd' or invalid
                            logger.info("补充内容未合并。")
                    else:
                        logger.warning("未能生成有效的补充内容。")
                else:
                    logger.warning("无效操作。")
            # End of inner while loop (segment_generation_complete)
            if action == 'q': break # Exit outer while loop if quit was chosen in inner
        
        else: # Initial generation failed
            logger.warning(f"未能生成故事片段 (generate_with_llm 返回 None 或空)。")
            final_user_choice_for_segment = 'n' # Treat as not adopted

        if final_user_choice_for_segment == 'y' and current_segment_text:
            actual_segment_number_to_save = current_segment_number + 1
            generated_chapter_title = ""
            text_content_for_md = current_segment_text
            
            # Extract chapter title if present
            if text_content_for_md.strip().startswith("##"):
                first_line_md = text_content_for_md.split('\n', 1)[0]
                if "章节名：" in first_line_md:
                    generated_chapter_title = first_line_md.replace("## 章节名：", "").strip()
                    text_content_for_md = text_content_for_md.split('\n', 1)[1] if '\n' in text_content_for_md else ""
            
            chapter_file_path = os.path.join(NOVEL_MD_OUTPUT_DIR, f"chapter_{current_chapter}.md")
            md_segment_header = ""
            is_first_segment_of_new_file = (actual_segment_number_to_save == 1 and 
                                           (not os.path.exists(chapter_file_path) or os.path.getsize(chapter_file_path) == 0))
            
            if is_first_segment_of_new_file: md_segment_header += f"# 第 {current_chapter} 章\n\n"
            if generated_chapter_title: md_segment_header += f"## {generated_chapter_title}\n\n"
            elif is_first_segment_of_new_file: md_segment_header += f"## (章节 {current_chapter} 未命名)\n\n" # Default if no title and first segment
            
            # Use a snippet of the original user directive for context in the MD file
            directive_snippet = user_directive.splitlines()[0][:100] if user_directive.splitlines() else "无"
            md_segment_header += f"\n### 片段 {actual_segment_number_to_save} (指令: {directive_snippet}...)\n\n"
            
            try:
                with open(chapter_file_path, 'a', encoding='utf-8') as f:
                    if md_segment_header: f.write(md_segment_header)
                    f.write(text_content_for_md.strip()); f.write("\n\n")
                logger.info(f"片段已追加到MD文件: {chapter_file_path}")
            except Exception as e_write: logger.error(f"错误：保存片段到文件失败: {e_write}", exc_info=True)
            
            text_to_store_in_db = text_content_for_md.strip()
            vector_to_store = None
            if selected_embedding_provider_identifier == "openai_official":
                vector_to_store_list = get_openai_embeddings([text_to_store_in_db])
                if vector_to_store_list: vector_to_store = vector_to_store_list[0]
            elif selected_embedding_provider_identifier.startswith("sentence_transformer_"):
                vector_to_store_list = get_st_embeddings([text_to_store_in_db])
                if vector_to_store_list: vector_to_store = vector_to_store_list[0]

            if vector_to_store:
                doc_id = add_story_segment_to_milvus(text_to_store_in_db, current_chapter, actual_segment_number_to_save, vector_to_store)
                if doc_id: logger.info(f"片段已添加到 Milvus 记忆库。ID: {doc_id}")
            else: logger.warning("错误：未能为采纳的片段生成向量，无法存入 Milvus。")
            
            current_segment_number = actual_segment_number_to_save # Increment only if adopted
            # Chapter progression logic
            if current_segment_number >= int(os.getenv("SEGMENTS_PER_CHAPTER_ADVANCE", "3")): # Configurable
                advance_choice = input(f"当前章节已采纳 {current_segment_number} 个片段。是否进入下一章节? (y/n): ").lower()
                if advance_choice == 'y':
                    current_chapter += 1
                    current_segment_number = 0 # Reset for new chapter
                    logger.info(f"进入新章节: {current_chapter}")
        elif final_user_choice_for_segment == 'n' and current_segment_text is None:
            logger.info("当前片段未被采纳，将为同一片段号重新征询指令。")
            # current_segment_number remains the same, so next loop iteration will be for the same segment number
        
        if action == 'q': break # Break from outer while if quit chosen

    logger.info(f"=== 小说写作助手 ({current_chosen_llm_provider_arg.upper()}) 关闭 ===")


# --- 12. 主程序入口 (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    main_chosen_embedding_provider_identifier_local: Optional[str] = None
    main_chosen_llm_provider_local: Optional[str] = None
    actual_st_model_for_init: Optional[str] = None

    embedding_providers_map = {
        "1": ("sentence_transformer_text2vec", ST_MODEL_TEXT2VEC, "(本地ST Text2Vec中文)"),
        "2": ("sentence_transformer_bge_large_zh", ST_MODEL_BGE_LARGE_ZH, "(本地ST BGE中文Large)"),
        "3": ("openai_official", OPENAI_EMBEDDING_MODEL, "(需要官方OpenAI API Key)")
    }
    print("请选择用于 Milvus 向量嵌入的提供商:")
    for key, (identifier, model_disp_name, desc) in embedding_providers_map.items():
        print(f"{key}. {identifier.upper()} {desc} - 模型: {model_disp_name}")
    emb_choice_input = ""
    while emb_choice_input not in embedding_providers_map:
        emb_choice_input = input(f"输入嵌入提供商选项 (1-{len(embedding_providers_map)}): ").strip()

    main_chosen_embedding_provider_identifier_local, model_name_associated_with_choice, _ = embedding_providers_map[emb_choice_input]
    selected_embedding_provider_identifier = main_chosen_embedding_provider_identifier_local

    try:
        if main_chosen_embedding_provider_identifier_local.startswith("sentence_transformer_"):
            actual_st_model_for_init = model_name_associated_with_choice
            selected_st_model_name = actual_st_model_for_init
            logger.info(f"准备初始化 Sentence Transformer，模型: {actual_st_model_for_init}")
            # Proxy for ST model download is tricky. Usually, it's better to download manually or ensure direct access.
            # _set_temp_os_proxies(None, None) # This might break if user *needs* proxy for HF
            try:
                embedding_model_instance = SentenceTransformer(actual_st_model_for_init) # device='cpu' or device=None for auto
                logger.info(f"Sentence Transformer 模型 '{actual_st_model_for_init}' 已在 main 中准备。")
                # More robust dimension detection
                try:
                    test_emb = embedding_model_instance.encode(["test"])
                    embedding_dimension = test_emb.shape[1]
                    logger.info(f"自动检测到 ST 模型 '{actual_st_model_for_init}' 的维度为: {embedding_dimension}")
                except Exception as e_dim:
                    logger.warning(f"自动检测 ST 模型维度失败: {e_dim}. 将使用预设值。")
                    if "base" in actual_st_model_for_init.lower(): embedding_dimension = 768
                    elif "large" in actual_st_model_for_init.lower(): embedding_dimension = 1024
                    elif ST_MODEL_TEXT2VEC in actual_st_model_for_init : embedding_dimension = 768
                    elif ST_MODEL_BGE_LARGE_ZH in actual_st_model_for_init: embedding_dimension = 1024
                    else: raise ValueError(f"无法确定ST模型维度: {actual_st_model_for_init}")
            finally:
                pass # _restore_original_os_proxies() # Only if _set_temp_os_proxies was used reliably
        elif main_chosen_embedding_provider_identifier_local == "openai_official":
            if not OPENAI_API_KEY: raise ValueError("OpenAI API Key 未配置 (用于嵌入)。")
            logger.info(f"OpenAI 嵌入模型 '{OPENAI_EMBEDDING_MODEL}' 将被使用。")
            if OPENAI_EMBEDDING_MODEL == "text-embedding-3-small": embedding_dimension = 1536
            elif OPENAI_EMBEDDING_MODEL == "text-embedding-3-large": embedding_dimension = 3072
            elif OPENAI_EMBEDDING_MODEL == "text-embedding-ada-002": embedding_dimension = 1536 # Legacy
            else: raise ValueError(f"未知的 OpenAI Embedding 模型维度: {OPENAI_EMBEDDING_MODEL}")
        else:
            raise ValueError(f"主程序中未处理的嵌入提供商: {main_chosen_embedding_provider_identifier_local}")

        initialize_embedding_function(main_chosen_embedding_provider_identifier_local) # Sets global var
        init_milvus_collections() # Uses global vars
    except Exception as e:
        logger.error(f"初始化嵌入模型或Milvus失败: {e}", exc_info=True)
        _restore_original_os_proxies() # Ensure restoration on error
        exit(1)

    seed_initial_lore()

    llm_providers_map = {"1": "openai_official", "2": "deepseek", "3": "gemini", "4": "custom_proxy_llm"}
    print("\n请选择要使用的 LLM 提供商 (用于文本生成):")
    for key, value in llm_providers_map.items():
        desc = f"(自定义代理URL: {CUSTOM_PROXY_BASE_URL})" if value == "custom_proxy_llm" else ""
        print(f"{key}. {value.upper()} {desc}")
    llm_choice_input = ""
    while llm_choice_input not in llm_providers_map:
        llm_choice_input = input(f"输入 LLM 提供商选项 (1-{len(llm_providers_map)}): ").strip()
    main_chosen_llm_provider_local = llm_providers_map[llm_choice_input]
    logger.info(f"你选择了 LLM: {main_chosen_llm_provider_local.upper()}, 嵌入提供商: {selected_embedding_provider_identifier.upper()}")

    required_api_keys_env_vars = []
    if selected_embedding_provider_identifier == "openai_official":
        required_api_keys_env_vars.append(OPENAI_API_KEY_ENV_NAME)

    if main_chosen_llm_provider_local == "openai_official": required_api_keys_env_vars.append(OPENAI_API_KEY_ENV_NAME)
    elif main_chosen_llm_provider_local == "deepseek": required_api_keys_env_vars.append(DEEPSEEK_API_KEY_ENV_NAME)
    elif main_chosen_llm_provider_local == "gemini": required_api_keys_env_vars.append(GEMINI_API_KEY_ENV_NAME)
    elif main_chosen_llm_provider_local == "custom_proxy_llm":
        if not CUSTOM_PROXY_API_KEY_FROM_ENV and not HARDCODED_CUSTOM_PROXY_KEY: # Only required if neither is set
            required_api_keys_env_vars.append(CUSTOM_PROXY_API_KEY_ENV_NAME)

    main_all_keys_valid = True
    unique_keys_to_check = list(set(required_api_keys_env_vars))
    if unique_keys_to_check: # Only check if there are keys to check
        for key_env_name_str in unique_keys_to_check:
            if not os.getenv(key_env_name_str):
                # Special handling for CUSTOM_PROXY_API_KEY if hardcoded one exists
                is_custom_proxy_key_missing_but_hardcoded_exists = (
                    key_env_name_str == CUSTOM_PROXY_API_KEY_ENV_NAME and
                    main_chosen_llm_provider_local == "custom_proxy_llm" and
                    HARDCODED_CUSTOM_PROXY_KEY
                )
                if not is_custom_proxy_key_missing_but_hardcoded_exists:
                     logger.error(f"错误：必要API Key的环境变量 {key_env_name_str} 未设置。")
                     main_all_keys_valid = False
                else: # Env var not set, but hardcoded key exists and is for the selected custom_proxy_llm
                     logger.warning(f"环境变量 {key_env_name_str} 未设置，但将使用硬编码的 CUSTOM_PROXY_API_KEY。")
    
    if main_all_keys_valid:
        try:
            writing_loop(main_chosen_llm_provider_local)
        except KeyboardInterrupt:
            logger.info("\n用户中断了写作。程序退出。")
        except Exception as e_main_loop:
            logger.error(f"写作循环中发生未捕获的错误: {e_main_loop}", exc_info=True)
        finally:
            _restore_original_os_proxies() # Ensure proxies are restored on any exit
            logger.info("写作助手已关闭。")
    else:
        print("请设置必要的 API Key 后重试。")