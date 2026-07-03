# Recorder AI Studio

Recorder AI Studio 是一个面向长录音场景的本地录音转文字与智能整理工作台。项目目标是把会议、访谈、课程、销售沟通等长音频转化为可编辑、可检索、可导出的结构化知识资产。

当前版本已经形成“真实转写 + Agent 工具化 + 可视化报告 + WorkBuddy 智能体校准”的完整闭环：前端页面、FastAPI 后端、本地 FunASR / SenseVoiceSmall 转写、异步 MCP 任务、HTML/Markdown/JSON 导出、校准包生成、智能体校准结果写回、自动化测试与真实样例端到端验证。

> 重要原则：项目不使用 mock / fallback 伪造转写结果。模型未就绪或识别失败时会直接返回错误，不生成假文本。

## 核心功能

- 长录音项目管理：创建项目、维护场景、术语表、音频与转写结果。
- 音频上传与播放：支持本地上传音频，并通过后端提供音频播放接口。
- 本地真实转写：默认使用 FunASR 的 SenseVoiceSmall 模型进行本机识别。
- 远程 FunASR 兼容：可通过 `FUNASR_ENDPOINT` 接入远程 OpenAI Whisper 风格接口。
- 说话人/段落结构：转写结果按段落保存，包含开始时间、结束时间、说话人、置信度、标签、原始文本与清洗文本。
- 术语与标签：基于关键词自动提取标签，可覆盖芯片、工具链、编译器、算力、大模型等技术场景。
- 智能整理：生成简述、重点摘要、核心议题、详情整理、决策结论、风险点、线索追问、行动指导等结构化信息。
- 可视化报告：自动生成跟随系统明暗色的 HTML 报告，适合直接预览和分享。
- WorkBuddy 智能体校准：通过 `recorder_prepare_review` / `recorder_apply_review` 支持口语词清理、错别字修正、上下文语义校准和最终 calibrated 报告写回。
- 导出能力：支持 Markdown、HTML、project JSON、report JSON，以及校准后的 calibrated 版本导出。
- 明暗主题：前端支持跟随系统暗色 / 亮色，也可手动切换。
- 状态可观测：提供 `/api/asr/status`，可查看本地模型是否下载完成。

## 技术栈

- 前端：HTML、CSS、Vanilla JavaScript
- 后端：FastAPI、Uvicorn、Pydantic
- 转写：FunASR、ModelScope、SenseVoiceSmall
- 音频处理：librosa、soundfile、torchaudio、pydub
- 测试：pytest、httpx
- 存储：本地 JSON 文件与上传目录

## 项目结构

```text
recorder-ai-studio/
├── index.html                         # 前端入口
├── styles.css                         # 明暗主题与页面样式
├── app.js                             # 前端交互与 API 调用
├── requirements.txt                   # Python 依赖
├── recorder_agent.py                  # CLI Agent 入口
├── recorder_mcp_server.py             # MCP Server 入口
├── run_real_asr_test.py               # 真实音频转写测试脚本
├── continue_real_asr_after_model_ready.py
├── server/
│   ├── app.py                         # FastAPI 接口与静态服务
│   ├── agent_tools.py                 # CLI/API/MCP 共享工具层
│   └── core.py                        # 项目存储、FunASR 调用、保活引擎、整理与导出逻辑
└── tests/
    ├── test_api.py                    # API 测试
    ├── test_core.py                   # 核心逻辑测试
    ├── test_agent_tools.py            # Agent 工具层测试
    ├── test_asr_engine.py             # 模型闲置保活策略测试
    └── test_audio_chunks.py           # 长音频分片相关测试
```

## 环境准备

建议使用 Python 3.11+。本项目已经在本地隔离环境中验证过 FunASR、ModelScope、Torch、FastAPI 等依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如需使用 ModelScope 本地缓存，建议显式设置缓存路径，避免污染系统环境：

```bash
export HOME="$(pwd)/../.cache/home"
export MODELSCOPE_CACHE="$(pwd)/../.cache/modelscope"
export MODELSCOPE_CREDENTIALS_PATH="$(pwd)/../.cache/modelscope/credentials"
```

## 启动服务

默认使用本地 FunASR / SenseVoiceSmall：

```bash
export FUNASR_MODEL="SenseVoiceSmall"
export FUNASR_VAD_MODEL=""
export FUNASR_PUNC_MODEL=""
export FUNASR_CHUNK_SECONDS="60"
export FUNASR_BATCH_SIZE_S="60"
export FUNASR_KEEPALIVE_SECONDS="600"

python -m uvicorn server.app:app --host 127.0.0.1 --port 8876
```

启动后访问：

```text
http://127.0.0.1:8876/
```

健康检查：

```text
GET http://127.0.0.1:8876/api/health
```

模型状态：

```text
GET http://127.0.0.1:8876/api/asr/status
```

## Agent / CLI / MCP 使用方式

这个项目不仅可以作为 Web 应用运行，也可以作为 Agent 工具被 WorkBuddy 或其他自动化系统调用。推荐分三层使用：

1. **CLI 工具**：适合脚本、定时任务、本地批处理。
2. **HTTP API**：适合前端页面或其他服务调用。
3. **MCP Server**：适合接入 WorkBuddy 自定义连接器，让 WorkBuddy 直接调用录音转写工具。

### CLI：查询模型状态

```bash
PYTHONPATH=. python recorder_agent.py status
```

返回示例：

```json
{
  "funasr": {
    "model": "iic/SenseVoiceSmall",
    "ready": true,
    "state": "ready"
  },
  "runtime": {
    "loaded": false,
    "keepaliveSeconds": 600,
    "remainingKeepaliveSeconds": null
  }
}
```

### CLI：真实转写音频

```bash
PYTHONPATH=. python recorder_agent.py transcribe /path/to/audio.mp3 \
  --title "meeting-demo" \
  --scene meeting \
  --glossary "芯片,工具链,编译器,AI" \
  --output-dir ../outputs/agent-runs
```

输出包括：

- `*-project.json`：完整结构化项目结果。
- `*-transcript.md`：Markdown 纪要。
- `*-report.html`：可视化智能报告。
- `*-report.json`：运行报告。

CLI 同样遵守真实转写约束：模型未就绪、音频不存在、FunASR 失败或输出为空时直接失败，不生成假结果。

### 模型闲置保活策略

项目默认采用“短时间保活、长期空闲释放”的策略，而不是永久常驻模型：

```bash
export FUNASR_KEEPALIVE_SECONDS="600"
```

含义：

- 第一次真实转写会加载 SenseVoiceSmall。
- 之后 10 分钟内有新任务，会复用已加载模型，避免频繁冷启动。
- 超过 10 分钟没有新任务，下一次状态检查或任务执行前会自动释放模型。
- 如需立即释放，可使用 CLI：

```bash
PYTHONPATH=. python recorder_agent.py release
```

`status` / `/api/asr/status` / MCP `recorder_asr_status` 会返回 `runtime` 字段，包括 `loaded`、`keepaliveSeconds`、`idleSeconds`、`remainingKeepaliveSeconds`、`loadCount`、`releaseCount` 等运行时信息。

### MCP：接入 WorkBuddy

项目提供 MCP Server 入口：

```bash
PYTHONPATH=/path/to/recorder-ai-studio python /path/to/recorder-ai-studio/recorder_mcp_server.py
```

可暴露七个工具：

| 工具 | 说明 |
| --- | --- |
| `recorder_asr_status` | 查询本地 FunASR / SenseVoiceSmall 模型和运行时保活状态 |
| `recorder_release_model` | 立即释放当前进程中已加载的本地模型 |
| `recorder_transcribe` | 提交非阻塞真实转写任务，立即返回 `jobId` |
| `recorder_job_status` | 查询转写任务状态、日志路径和输出路径 |
| `recorder_job_result` | 任务完成后获取 HTML 可视化报告、Markdown、project JSON、report JSON 输出 |
| `recorder_prepare_review` | 为 WorkBuddy 智能体生成待校准包和校准提示词，不直接调用云端模型 |
| `recorder_apply_review` | 将 WorkBuddy 智能体产出的校准 JSON 写回为 calibrated Markdown/HTML/JSON 报告 |

WorkBuddy 的用户级 MCP 配置示例：

```json
{
  "mcpServers": {
    "recorder-ai-studio": {
      "command": "python",
      "args": ["/path/to/recorder-ai-studio/recorder_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/recorder-ai-studio",
        "FUNASR_MODEL": "SenseVoiceSmall",
        "FUNASR_VAD_MODEL": "",
        "FUNASR_PUNC_MODEL": "",
        "FUNASR_CHUNK_SECONDS": "60",
        "FUNASR_BATCH_SIZE_S": "60",
        "FUNASR_KEEPALIVE_SECONDS": "600"
      }
    }
  }
}
```

配置后，在 WorkBuddy 连接器管理中启用该自定义 MCP，即可让 WorkBuddy 调用本地真实转写工具。

MCP 转写采用非阻塞任务模式，避免长录音转写时 WorkBuddy 一直占用单次 JSON-RPC 调用：

1. 调用 `recorder_transcribe` 提交任务并获取 `jobId`。
2. 调用 `recorder_job_status(jobId)` 查询 `queued` / `running` / `completed` / `failed`。
3. 完成后调用 `recorder_job_result(jobId)` 获取 HTML 可视化报告、Markdown、project JSON、report JSON 路径。

默认任务文件和日志写入：

```text
../outputs/mcp-jobs/
```

每次真实转写会同时生成：

- `*-report.html`：适合在浏览器中打开的可视化报告，包含重点摘要、核心议题、详情整理、线索追问、关键词热度和转写后工作指导。
- `*-transcript.md`：适合二次编辑和同步给团队的 Markdown 纪要。
- `*-project.json`：完整结构化项目数据。
- `*-report.json`：给 Agent / MCP 消费的轻量结果索引。

### WorkBuddy 智能体校准工作流

本项目不在 MCP 服务内直接硬接云端大模型。推荐由 MCP 做本地转写和文件落盘，由 WorkBuddy 当前智能体读取校准包并使用对话环境的 AI 能力完成语义优化。

1. `recorder_transcribe` 提交真实转写任务。
2. `recorder_job_status` 等待任务完成。
3. `recorder_prepare_review(jobId)` 生成：
   - `*-review-package.json`：包含原始分段、当前文本、低置信片段、术语表和上下文。
   - `*-review-prompt.md`：指导 WorkBuddy 智能体输出校准 JSON。
4. WorkBuddy 智能体根据 review package 完成：
   - 口语词清理
   - 错别字/同音误识别修正
   - 上下文语义校准
   - 专有名词统一
   - 核心议题、重点、详情、线索、追问、待办提取
5. 将智能体产出的 JSON 传给 `recorder_apply_review(review_json, jobId)`。
6. 工具写回最终产物：
   - `*-calibrated-project.json`
   - `*-calibrated-transcript.md`
   - `*-calibrated-report.html`
   - `*-calibrated-report.json`

这样可以复用 WorkBuddy 当前智能体的模型能力和用户上下文，同时保持 MCP 服务本身稳定、可测试、无额外云模型鉴权。

## API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 服务健康检查 |
| `GET` | `/api/asr/status` | 本地 FunASR 模型状态 |
| `GET` | `/api/projects` | 项目列表 |
| `POST` | `/api/projects` | 创建项目 |
| `GET` | `/api/projects/{project_id}` | 获取项目详情 |
| `PUT` | `/api/projects/{project_id}` | 更新项目 |
| `POST` | `/api/projects/{project_id}/upload` | 上传音频 |
| `GET` | `/api/projects/{project_id}/audio` | 播放音频 |
| `POST` | `/api/projects/{project_id}/transcribe` | 执行真实转写 |
| `POST` | `/api/projects/{project_id}/insights` | 生成摘要、待办、脑图等整理结果 |
| `GET` | `/api/projects/{project_id}/export.md` | 导出 Markdown |

## 真实转写约束

本项目刻意避免“看起来像完成”的伪结果：

1. 模型权重未下载完成时，转写接口返回 `409 Conflict`。
2. FunASR 失败时返回 `502`，不会自动降级到示例文本。
3. 没有音频时返回 `400`。
4. FunASR 产出为空时返回 `502`。
5. `transcriptionSource` 只接受真实来源：`local_funasr` 或 `remote_funasr`。

这保证前端展示的文字一定来自真实识别链路。

## 运行测试

```bash
pytest -q --rootdir=. tests
```

当前已验证：

- API 流程测试通过。
- 核心逻辑测试通过。
- 长音频分片准备逻辑通过。
- SenseVoice 控制标记清洗逻辑通过。
- 完整真实样例测试通过。

## 真实样例验证

项目内置了真实音频端到端测试脚本，但不会提交任何私有音频。运行前请通过环境变量指定测试音频：

```bash
export RECORDER_AI_TEST_AUDIO="/path/to/your/meeting-audio.mp3"
export RECORDER_AI_TEST_TITLE="meeting-audio"
python run_real_asr_test.py
```

验证目标：

- 转写来源必须是 `local_funasr`。
- 不允许使用 mock / fallback。
- FunASR 产出为空时直接失败。
- 可同时验证后端健康检查、项目创建、上传、导出链路。
- 输出目录默认是 `../outputs/real-asr-test/`，可通过 `RECORDER_AI_OUTPUT_DIR` 自定义。

在本地开发机上，曾使用一段约 91 分钟的真实会议录音完成端到端验证：本地 SenseVoiceSmall 识别耗时约 119.38 秒，生成 92 段转写，回归测试 7 passed。该音频和完整转写内容不包含在仓库中。

## 当前性能基准

以下数据来自本地真实长录音测试，具体耗时会受机器性能、模型冷启动状态、音频采样率、音频质量、是否启用智能体校准等因素影响。

| 场景 | 典型耗时 | 说明 |
| --- | ---: | --- |
| 纯 ASR 转写 | 约 47 倍实时速度 | 约 1 小时 MP3 可在 75～80 秒完成纯转写。 |
| 1 小时 MP3 → 自动 HTML | 约 1.5～2 分钟 | 包含本地转写、结构化初步整理、Markdown/HTML/JSON 落盘。 |
| 1 小时 MP3 → 高质量整理 HTML | 约 3～6 分钟 | 包含 WorkBuddy 智能体校准、口语词清理、专名修正、结构化纪要和 calibrated HTML 写回。 |

当前效率已经满足大多数会议/访谈场景的快速周转。后续真正值得持续优化的是：**专名校准、上下文语义修正、结构化纪要质量、音频分析专业度和自动化程度**。

## 性能调优方向

下一阶段建议从以下方向优化：

1. 模型闲置保活：已支持 10 分钟默认保活，活跃任务复用模型，长期空闲自动释放。
2. 分片并行化：对长音频按时间片并发识别，再按时间戳归并。
3. VAD / 标点策略分层：快速预览使用轻量识别，精修模式再启用 VAD、标点和说话人辅助模型。
4. 进度事件流：为长音频转写增加实时进度、当前分片、预计剩余时间。
5. 断点续跑：已完成分片落盘缓存，失败后只重跑未完成片段。
6. 任务队列：长录音通过 submit/status/result 三段式工具调用，避免 MCP/API 长时间阻塞。
7. 导出增强：增加 SRT、VTT、JSONL、Docx 等格式。
8. 智能体校准自动化：已支持 WorkBuddy 智能体校准包与写回工具，后续可继续优化长文本分批校准、差异审阅和人工确认界面。

## 后续质量与专业度优化方向

从当前测试结果看，纯转写效率已经比较理想，下一阶段更应该围绕“识别可信度、专名校准、结构化纪要、音频洞察”继续增强。

### 1. 专名校准与术语库

- 建立项目/团队级 glossary：公司名、产品名、芯片型号、工具链名称、人名、客户名、缩写词。
- 在 `recorder_prepare_review` 中自动提取疑似专名和低置信片段，交给 WorkBuddy 智能体集中校准。
- 支持“用户确认后的术语”回写到本地 glossary，下次同类会议自动优先使用。
- 对同音误识别做候选建议，例如把“新新型投资”提示为“新型投资/新兴投资/新型投入”等待确认项。

### 2. 高质量结构化纪要

- 按场景生成不同纪要模板：内部会议、客户访谈、销售沟通、课程培训、研发评审、投研访谈。
- 从单纯摘要升级为多层结构：背景、目标、讨论过程、决策、分歧、风险、待办、依赖、待确认问题。
- 为每个结论保留原文证据和时间戳，方便回听核验。
- 输出“老板版摘要”“执行版待办”“知识库版沉淀”三种视角。

### 3. 音频分析专业度

- 增加说话人维度统计：发言时长、发言占比、主要观点、待跟进事项。
- 增加会议节奏分析：长沉默、频繁打断、重点密集时间段、争议片段。
- 基于音频质量输出诊断：噪声、音量过低、重叠讲话、远场收音、可能影响识别的区间。
- 输出可回听索引：把关键议题、风险、决策定位到具体时间戳。

### 4. HTML 报告体验

- 增加“原文 vs 校准后”差异视图，突出删除口语词、修正错别字、补全语义的位置。
- 增加按议题过滤、按说话人过滤、按风险/待办/决策过滤。
- 增加关键时间线：按会议进程展示议题切换、关键决策和行动项。
- 增加可复制块：一键复制摘要、待办、风险、对外纪要。

### 5. 自动化工作流

- 支持一键流程：提交音频后自动转写、生成 review package、调用 WorkBuddy 智能体校准、写回 calibrated HTML。
- 对超长录音做分批校准：按议题或时间窗口分块，最后合并全局摘要与行动项。
- 增加人工确认环节：低置信专名、关键决策、负责人和截止时间需要用户确认后再固化。
- 支持复用历史会议上下文：同一项目/客户/团队的过往纪要可作为术语和背景参考。

## 数据与隐私

`.gitignore` 已默认排除：

- 上传音频与本地运行数据：`uploads/`、`data/`、`data-*/`
- 模型权重与缓存：`.cache/`、`*.pt`、`*.onnx`、`*.safetensors`
- Python 缓存：`__pycache__/`、`.pytest_cache/`
- 本地环境：`.env`、`.venv/`、`node_modules/`

因此仓库默认只保存源码、测试和必要脚本，不提交用户音频、模型权重和本地数据。

## 当前状态

当前代码已经形成稳定基线：

- 前后端链路可运行。
- CLI/API/MCP 三种入口可用。
- MCP 支持非阻塞 submit/status/result 任务模式。
- 本地模型支持 10 分钟闲置保活，避免频繁冷启动。
- 短样本端到端测试通过。
- 完整样例真实转写测试通过。
- HTML 可视化报告已接入 Agent/MCP 输出。
- WorkBuddy 智能体校准包与 calibrated 报告写回已接入。
- 完整测试套件通过。
