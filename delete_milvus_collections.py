from pymilvus import connections, utility

# --- 配置 ---
# Milvus 服务器连接参数 (与你的主脚本一致)
MILVUS_ALIAS = "default"
MILVUS_HOST = "localhost" # 或者你 Milvus 服务器的实际 IP 地址
MILVUS_PORT = "19530"   # 或者你 Milvus 服务器的实际端口

# !!! 重要：在这里填写你想要删除的集合的准确名称 !!!
# 你需要根据你之前运行主脚本时选择的嵌入提供商和模型来确定这些名称。
# 例如，如果用 Sentence Transformer "shibing624/text2vec-base-chinese" (简写 st_t2v_chinese)
# LORE_COLLECTION_NAME_TO_DELETE = "novel_lore_mv_st_t2v_chinese"
# STORY_COLLECTION_NAME_TO_DELETE = "novel_story_mv_st_t2v_chinese"

# 例如，如果用 Sentence Transformer "BAAI/bge-large-zh-v1.5" (简写 st_bge_large)
LORE_COLLECTION_NAME_TO_DELETE = "novel_lore_mv_stbge__bge_large_" 
STORY_COLLECTION_NAME_TO_DELETE = "novel_story_mv_stbge__bge_large_"

# 例如，如果用 OpenAI Official Embedding "text-embedding-3-small" (简写 oai_3sm)
# LORE_COLLECTION_NAME_TO_DELETE = "novel_lore_mv_oai_3sm"
# STORY_COLLECTION_NAME_TO_DELETE = "novel_story_mv_oai_3sm"

# 如果你只想删除其中一个，可以将另一个设置为空字符串 "" 或 None
# LORE_COLLECTION_NAME_TO_DELETE = "" # 示例：不删除 lore 集合


def delete_collection_if_exists(collection_name: str):
    """检查集合是否存在，如果存在则删除。"""
    if not collection_name:
        print(f"未提供集合名称，跳过删除。")
        return

    try:
        if utility.has_collection(collection_name, using=MILVUS_ALIAS):
            print(f"找到集合: '{collection_name}'。准备删除...")
            
            # --- 添加用户确认步骤，防止误删 ---
            confirm = input(f"你确定要永久删除集合 '{collection_name}' 吗？这个操作无法撤销！(输入 'yes' 确认): ")
            if confirm.lower() == 'yes':
                utility.drop_collection(collection_name, using=MILVUS_ALIAS)
                print(f"集合 '{collection_name}' 已成功删除。")
            else:
                print(f"删除操作已取消。集合 '{collection_name}' 未被删除。")
        else:
            print(f"集合 '{collection_name}' 不存在，无需删除。")
    except Exception as e:
        print(f"处理集合 '{collection_name}' 时发生错误: {e}")

if __name__ == "__main__":
    print("--- Milvus 集合删除脚本 ---")
    print(f"将尝试连接到 Milvus 服务器: {MILVUS_HOST}:{MILVUS_PORT}")

    try:
        connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
        print("成功连接到 Milvus。")

        # 删除 Lore Collection
        if LORE_COLLECTION_NAME_TO_DELETE:
            delete_collection_if_exists(LORE_COLLECTION_NAME_TO_DELETE)
        else:
            print("Lore 集合名称未配置，跳过删除 Lore 集合。")

        # 删除 Story Collection
        if STORY_COLLECTION_NAME_TO_DELETE:
            delete_collection_if_exists(STORY_COLLECTION_NAME_TO_DELETE)
        else:
            print("Story 集合名称未配置，跳过删除 Story 集合。")

    except Exception as e:
        print(f"连接到 Milvus 或执行操作时发生错误: {e}")
    finally:
        try:
            # 尝试断开所有连接，或者特定别名的连接
            # PyMilvus 的连接管理可能因版本而异，通常不需要显式对每个别名断开
            if connections.has_connection(MILVUS_ALIAS):
                connections.disconnect(MILVUS_ALIAS)
                print(f"已断开与 Milvus (别名: {MILVUS_ALIAS}) 的连接。")
        except Exception as e_disconnect:
            print(f"断开 Milvus 连接时发生错误: {e_disconnect}")
            
    print("脚本执行完毕。")