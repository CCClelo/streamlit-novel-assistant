
```markdown
# AI 小说写作助手 (AI Novel Writing Assistant)

本项目是一个实验性的 Python 应用程序，旨在利用大型语言模型 (LLM) 和向量数据库 (Milvus) 来辅助用户进行长篇网络小说的构思和创作。它基于检索增强生成 (RAG) 的原理，通过从知识库中检索相关上下文信息，为 LLM 提供创作基础，以期生成更连贯、更符合设定的故事情节。

## 核心功能

*   **多 LLM 支持**: 可通过配置切换使用 OpenAI (GPT 系列)、Google Gemini、DeepSeek 以及任何兼容 OpenAI API 格式的自定义代理服务作为内容生成引擎。
*   **灵活的文本嵌入**: 支持使用本地开源的 Sentence Transformers 模型 (如 `shibing624/text2vec-base-chinese`, `BAAI/bge-large-zh-v1.5`) 或 OpenAI 官方的 Embedding API 进行文本向量化。
*   **Milvus 向量数据库集成**: 利用 Milvus 存储和高效检索小说设定、世界观、人物小传以及已生成并被采纳的故事片段，作为 AI 的“长期记忆”。
*   **外部设定加载**: 允许用户将小说设定、大纲等信息保存在本地文本文件 (`./novel_setting_files` 目录)，脚本会自动加载并将其向量化存入知识库。
*   **交互式创作流程**: 用户通过命令行输入写作指令，AI 基于指令和上下文生成故事片段。
*   **内容采纳与输出**:
    *   用户可以评估 AI 生成的片段，选择是否采纳。
    *   采纳的片段会自动追加到按章节组织的 Markdown 文件中 (`./novel_markdown_chapters` 目录)，方便后续编辑和阅读。
    *   采纳的片段（其纯文本内容）也会被向量化并存入 Milvus 的故事片段库，作为后续 AI 生成的“上文回顾”上下文。
*   **动态代理管理**: 脚本尝试根据当前操作（如 Sentence Transformer 模型下载、特定 LLM API 调用）动态调整 Python 进程内的代理环境变量，以适应不同的网络访问需求。

## 系统架构（简要）

1.  **用户输入**: 写作指令、小说设定文件。
2.  **设定加载与向量化**: 从文件读取设定，使用选定的嵌入模型生成向量。
3.  **Milvus 知识库**: 存储设定文本及其向量。
4.  **Milvus 故事片段库**: 存储已采纳的 AI 生成的故事片段及其向量。
5.  **RAG 核心**:
    *   根据用户当前指令，从知识库检索相关设定。
    *   从故事片段库检索最新的已采纳内容作为“上文”。
    *   构建包含设定、上文和用户指令的 Prompt。
6.  **LLM 调用**: 将 Prompt 发送给选定的 LLM 服务。
7.  **内容输出**:
    *   在控制台显示 AI 生成的内容。
    *   如果用户采纳，保存到 Markdown 文件和故事片段库。

## 环境要求

*   **Python**: 3.9 或更高版本。
*   **Docker**: 最新稳定版 Docker Desktop (Windows/macOS) 或 Docker Engine (Linux)。
*   **Docker Compose**: 最新稳定版 (通常随 Docker Desktop 安装，Linux 上可能需要单独安装)。
*   **Git**: 用于从 GitHub 克隆项目（如果项目托管在 GitHub）。
*   **网络连接**:
    *   需要能够访问 Hugging Face Hub (以下载 Sentence Transformer 模型，除非模型已缓存且你希望脚本直连)。
    *   需要能够访问你选择使用的 LLM API 服务 (如 OpenAI, Google Gemini, DeepSeek API)。
    *   **可能需要配置 HTTP/HTTPS 代理**以访问上述服务，具体取决于你的网络环境。

## 安装与部署详细步骤

### 1. 获取项目代码

*   **如果项目已在 GitHub：**
    ```bash
    git clone https://github.com/your_username/your_project_repository.git
    cd your_project_repository
    ```
    请将 `your_username/your_project_repository.git` 替换为实际的仓库地址。
*   **如果代码已在本地：** 直接进入项目根目录。

### 2. 创建并激活 Python 虚拟环境 (强烈推荐)

这有助于隔离项目依赖，避免与系统全局 Python 环境冲突。
```bash
python -m venv .venv 
# 或者使用你喜欢的名称，如 venv, env

# 激活虚拟环境:
# Windows CMD:
# .\.venv\Scripts\activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# macOS / Linux (bash/zsh):
# source .venv/bin/activate 
```
激活后，你的命令行提示符通常会显示虚拟环境的名称。

### 3. 安装 Python 依赖库

项目依赖多个 Python 库。在激活虚拟环境后，通过以下命令安装：
```bash
pip install openai pymilvus sentence-transformers transformers "google-generativeai>=0.3.0" python-dotenv httpx
```
或者，如果项目提供了 `requirements.txt` 文件：
```bash
pip install -r requirements.txt
```
**主要依赖说明：**
*   `openai`: OpenAI 和兼容 OpenAI API 的服务 (如 DeepSeek, Custom Proxy) 的官方 SDK。
*   `pymilvus`: Milvus 向量数据库的 Python SDK。
*   `sentence-transformers`: 用于加载和使用本地 Sentence Transformer 嵌入模型。
*   `transformers`: `sentence-transformers` 的核心依赖，提供 Transformer 模型架构。
*   `google-generativeai`: Google Gemini API 的官方 Python SDK (确保版本 >= 0.3.0 以支持较新的 API 特性)。
*   `python-dotenv`: 用于从 `.env` 文件加载环境变量。
*   `httpx`: 一个现代的 HTTP 客户端，`openai` SDK v1.0.0+ 使用它，也用于我们脚本中为 OpenAI SDK 配置特定代理。

### 4. 部署 Milvus Standalone 服务

本项目使用 Milvus 作为向量数据库。推荐使用 Docker Compose 部署 Milvus Standalone 版本进行本地开发和测试。

*   **确保 Docker Desktop 已启动并正常运行。**
*   **获取 `docker-compose.yml` 文件：**
    *   从 [Milvus 官方 GitHub 仓库](https://github.com/milvus-io/milvus/tree/master/deployments/docker/standalone) 下载适用于 Milvus Standalone 的最新稳定版 `docker-compose.yml` 文件。
    *   将其保存到你的项目根目录下。
*   **配置数据卷持久化 (重要)：**
    打开你下载的 `docker-compose.yml` 文件。找到所有 `volumes:` 部分。为了将 Milvus 数据（元数据、向量索引、对象存储）持久化到你的项目目录下的一个子文件夹（例如 `milvus_data`），而不是 Docker 的默认位置，你可以：
    *   **使用环境变量 `DOCKER_VOLUME_DIRECTORY` (推荐)：**
        在启动 Docker Compose **之前**，在你的命令行终端中设置此环境变量指向当前项目目录。
        ```bash
        # Windows CMD (在项目根目录下):
        set DOCKER_VOLUME_DIRECTORY=%CD% 
        # Windows PowerShell (在项目根目录下):
        # $env:DOCKER_VOLUME_DIRECTORY = (Get-Location).Path
        ```
        然后，确保 `docker-compose.yml` 中的 `volumes` 路径使用了这个环境变量，例如：
        ```yaml
        services:
          etcd:
            volumes:
              - ${DOCKER_VOLUME_DIRECTORY:-.}/milvus_data/volumes/etcd:/etcd 
          minio:
            volumes:
              - ${DOCKER_VOLUME_DIRECTORY:-.}/milvus_data/volumes/minio:/minio_data
          standalone:
            volumes:
              - ${DOCKER_VOLUME_DIRECTORY:-.}/milvus_data/volumes/milvus:/var/lib/milvus
        ```
        将 `${DOCKER_VOLUME_DIRECTORY:-.}` 替换为你的实际配置。如果你的项目根目录就是 `DOCKER_VOLUME_DIRECTORY`，那么路径会是 `./milvus_data/...`。
    *   **或者，直接修改 YAML 文件中的路径**为绝对路径或相对于项目根目录的相对路径。
*   **启动 Milvus 服务：**
    在包含 `docker-compose.yml` 的项目根目录下，打开命令行终端，运行：
    ```bash
    docker-compose up -d
    ```
    首次运行会下载镜像，可能需要一些时间。
*   **检查 Milvus 服务状态：**
    等待几分钟，然后运行：
    ```bash
    docker-compose ps
    ```
    确保 `milvus-standalone`, `milvus-etcd`, 和 `milvus-minio` 三个服务的 `State` 都是 `Up` 并且 `Status` 显示 `healthy`（如果定义了健康检查）。Milvus 的 gRPC 服务默认监听在主机的 `19530` 端口。
*   **查看日志 (如果启动失败)：**
    ```bash
    docker-compose logs standalone
    # 或 docker-compose logs -f (查看所有并跟踪)
    ```

### 5. 配置 API Keys 和代理服务器

*   在项目根目录下创建一个名为 `.env` 的文本文件。
*   **非常重要：将此 `.env` 文件添加到你的 `.gitignore` 文件中，以防止将敏感凭证提交到版本控制系统！**
*   根据你计划使用的服务，在 `.env` 文件中添加以下内容，并替换为你的实际值：

    ```env
    # --- API Keys ---
    # 替换为你的真实 API Key
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    GEMINI_API_KEY="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    # CUSTOM_PROXY_API_KEY="your_custom_proxy_key" # 如果你的自定义代理LLM需要特定Key

    # --- 代理配置 (如果需要) ---
    # 示例：为 Gemini API 设置代理 (如果它需要通过 http://127.0.0.1:7890 访问)
    # 如果不需要，请将这些行注释掉或删除
    GEMINI_HTTP_PROXY="http://127.0.0.1:7890"
    GEMINI_HTTPS_PROXY="http://127.0.0.1:7890"

    # 示例：如果 DeepSeek LLM 也需要代理 (根据实际情况配置)
    # DEEPSEEK_LLM_HTTP_PROXY="http://127.0.0.1:7890"
    # DEEPSEEK_LLM_HTTPS_PROXY="http://127.0.0.1:7890"

    # 示例：如果 OpenAI Official 服务需要代理 (不常见)
    # OPENAI_OFFICIAL_HTTP_PROXY="http://your_corp_proxy_ip:port"
    # OPENAI_OFFICIAL_HTTPS_PROXY="http://your_corp_proxy_ip:port"

    # 示例：如果 Custom Proxy LLM 本身也需要通过另一个系统代理访问其 base_url
    # CUSTOM_LLM_HTTP_PROXY="http://127.0.0.1:7890"
    # CUSTOM_LLM_HTTPS_PROXY="http://127.0.0.1:7890"
    
    # --- 其他可选配置 (脚本中已有默认值，如果需要覆盖则在此处设置) ---
    # CUSTOM_PROXY_BASE_URL="https://your_custom_proxy_domain.com/v1/"
    # DEEPSEEK_BASE_URL="https://api.deepseek.com"
    # MILVUS_HOST="localhost"
    # MILVUS_PORT="19530"
    ```
*   **代理说明：**
    *   **Sentence Transformer 模型下载：** 脚本会尝试在下载时临时清除 Python 进程内的代理环境变量，以促使其直连 Hugging Face Hub。如果你的网络环境必须通过代理才能访问 Hugging Face，你需要调整脚本中 `initialize_embedding_function` 内清除代理的逻辑，或者在运行脚本前就在操作系统级别设置好能访问 Hugging Face 的代理。
    *   **Gemini API：** 会使用 `.env` 文件中定义的 `GEMINI_HTTP_PROXY` 和 `GEMINI_HTTPS_PROXY`。
    *   **OpenAI SDK (用于 Official OpenAI, DeepSeek LLM, Custom Proxy LLM)：** 会使用 `.env` 文件中为它们各自配置的特定代理（如 `DEEPSEEK_LLM_HTTPS_PROXY`）。如果未配置特定代理，它们会尝试遵循脚本启动时操作系统级的代理环境变量，若也无，则尝试直连。

### 6. 准备小说设定文件

*   在项目根目录下创建一个名为 `novel_setting_files` 的文件夹（如果脚本配置中是这个名称）。
*   将你的小说核心设定、世界观、主要人物小传、重要情节大纲等，分别保存为 `.txt` 或 `.md` 文件放入此文件夹。
*   脚本会在首次针对某个嵌入配置运行时（或者当对应的 Milvus 知识库为空时）自动加载这些文件，进行分块、向量化，并存入 Milvus。

### 7. 创建 Markdown 输出目录 (可选)

*   脚本配置了 `NOVEL_MD_OUTPUT_DIR = "./novel_markdown_chapters"`。
*   如果此目录不存在，脚本在首次尝试写入时会自动创建它。这里会存放 AI 生成并被用户采纳的小说章节。

## 运行 AI 小说写作助手

1.  **确保 Milvus 服务正在运行** (通过 `docker-compose ps` 检查)。
2.  **确保你的本地代理程序（如果 Gemini 等服务需要）正在运行并配置正确，** 以匹配你在 `.env` 文件中为相应服务指定的代理。
3.  **激活 Python 虚拟环境。**
4.  **在项目根目录下，通过命令行运行主脚本：**
    ```bash
    python novel_writing_assistant.py
    ```
5.  **按照脚本的命令行提示进行操作：**
    *   **选择嵌入提供商：**
        *   `SENTENCE_TRANSFORMER_TEXT2VEC` 或 `SENTENCE_TRANSFORMER_BGE_LARGE_ZH`：使用本地模型，首次运行会下载模型（注意此时的网络和代理需求）。后续从本地缓存加载。
        *   `OPENAI_OFFICIAL`：使用 OpenAI API 进行嵌入，需要有效的 `OPENAI_API_KEY` 和配额。
    *   **选择 LLM 提供商：**
        *   选择 `GEMINI`, `OPENAI_OFFICIAL`, `DEEPSEEK`, 或 `CUSTOM_PROXY_LLM`。确保对应的 API Key 已在 `.env` 文件中配置。
    *   **输入写作指令：** 根据提示，输入你希望 AI 续写或创作的场景、情节或章节核心内容。可以尝试在指令中加入对章节名和期望篇幅的要求。
    *   **评估与采纳：** AI 生成内容后，脚本会询问你是否采纳。
        *   输入 `y` (yes)：内容将被追加到对应的章节 Markdown 文件，并存入 Milvus 故事片段库作为后续的记忆上下文。
        *   输入 `n` (no)：内容将不会被保存。
    *   **章节管理：** 每当采纳了若干片段后（默认为5个），脚本会询问是否进入下一章节。

## 注意事项与提示

*   **API Keys 和配额：** 所有外部 API 服务（OpenAI, Gemini, DeepSeek）都需要有效的 API Key，并且通常有免费层级的使用配额限制。请密切关注你的用量，避免超出限制导致服务中断。
*   **代理配置是关键：** 如果你所在网络环境访问外部 API (包括 Hugging Face 模型下载) 需要代理，请务必正确配置 `.env` 文件中的代理 URL，并确保你的本地代理程序工作正常且规则正确。
*   **模型下载：** 首次使用 Sentence Transformer 模型时，会从 Hugging Face Hub 下载，可能需要较长时间，并消耗一定的磁盘空间。下载后会缓存本地。
*   **Milvus 存储：** Milvus 的数据通过 Docker 数据卷持久化。确保你有足够的磁盘空间。
*   **长文本连贯性：** AI 生成极长篇幅且保持高度连贯性仍然是一个挑战。你需要通过清晰的指令、合理的上下文管理（脚本已包含基础的 RAG）以及可能的多次迭代来引导 AI。
*   **错误处理：** 脚本包含了一些基本的错误处理，但实际使用中可能会遇到各种网络、API 或配置问题。请留意控制台的日志输出，它们通常包含排错的关键信息。
*   **迭代优化：** 本项目是一个起点。你可以根据自己的需求，不断优化 Prompt 工程、上下文检索策略、用户交互方式等，以提升创作效率和内容质量。

## 未来的可能改进 (TODO)

*   **更智能的章节摘要和管理：** 实现自动生成章节摘要并将其用于更长期的记忆回顾。
*   **细粒度的上下文选择界面：** 允许用户在生成前更精确地选择哪些背景知识或历史片段作为上下文。
*   **迭代编辑与反馈循环：** 允许用户对 AI 生成的片段进行修改，并将修改后的版本反馈给 AI 或记忆库。
*   **多轮对话式生成：** 将长章节的生成分解为多个更小、更聚焦的生成步骤。
*   **Web UI / GUI：** 构建图形用户界面以提升易用性。
*   **更高级的 Milvus 查询：** 利用 Milvus 更复杂的过滤和排序功能来优化检索。

希望这份文档能帮助你成功部署和使用这个 AI 小说写作助手！祝创作愉快！
