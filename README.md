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
- 动态词库：按会议场景和内容自动加载部分分类词库，校准后可把高置信专名/术语写回本地 glossary，后续同类录音自动复用。
- 术语与标签：基于关键词、分类词库和会议上下文自动提取标签，可覆盖芯片、工具链、编译器、算力、大模型、Agent 等技术场景。
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
│   ├── glossary.py                    # 动态分类词库、术语候选提取和词库写回
│   └── core.py                        # 项目存储、FunASR 调用、保活引擎、整理与导出逻辑
├── data/glossary/                     # 可持续积累的本地分类词库
│   ├── global.json
│   ├── ai_chip.json
│   └── ai_agent.json
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

可暴露十二个工具，分为模型运行、异步转写、智能体校准、词库交互四组：

| 工具 | 说明 |
| --- | --- |
| `recorder_asr_status` | 查询本地 FunASR / SenseVoiceSmall 模型和运行时保活状态 |
| `recorder_release_model` | 立即释放当前进程中已加载的本地模型 |
| `recorder_transcribe` | 提交非阻塞真实转写任务，立即返回 `jobId` |
| `recorder_job_status` | 查询转写任务状态、日志路径和输出路径 |
| `recorder_job_result` | 任务完成后获取 HTML 可视化报告、Markdown、project JSON、report JSON 输出 |
| `recorder_prepare_review` | 为 WorkBuddy 智能体生成待校准包和校准提示词，不直接调用云端模型 |
| `recorder_apply_review` | 将 WorkBuddy 智能体产出的校准 JSON 写回为 calibrated Markdown/HTML/JSON 报告 |
| `recorder_glossary_list` | 查询分类词库，可按类别和关键词过滤 |
| `recorder_glossary_suggest` | 基于已完成 job 或 project JSON 生成词库候选、命中词和低置信术语 |
| `recorder_glossary_confirm` | 将用户确认后的术语写入分类词库 |
| `recorder_glossary_reject` | 记录用户拒绝的候选，避免后续重复推荐 |
| `recorder_glossary_update` | 直接新增或更新一个词库条目 |

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

### MCP 整体推荐流程

MCP 不直接承担复杂交互界面，交互由 WorkBuddy 智能体在对话层完成；MCP 负责稳定的结构化读写、转写任务、词库操作和报告落盘。

推荐完整流程：

```text
用户提供音频
  → recorder_transcribe 提交异步转写
  → recorder_job_status 等待 completed
  → recorder_job_result 获取原始 HTML/Markdown/JSON
  → recorder_prepare_review 生成校准包
  → WorkBuddy 智能体做文本校准、纪要整理、词库候选判断
  → 必要时 recorder_glossary_suggest/list 辅助用户确认词库
  → recorder_glossary_confirm/reject/update 写回词库
  → recorder_apply_review 生成 calibrated HTML/Markdown/JSON
```

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
   - `*-review-package.json`：包含原始分段、当前文本、低置信片段、动态加载的分类词库、词库命中、疑似专名候选和上下文。
   - `*-review-prompt.md`：指导 WorkBuddy 智能体输出校准 JSON，并把高置信术语放入 `confirmedTerms`。
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
7. 若智能体返回 `confirmedTerms` / `glossaryUpdates` / `termCorrections`，工具会按分类自动写回 `data/glossary/*.json`。

这样可以复用 WorkBuddy 当前智能体的模型能力和用户上下文，同时保持 MCP 服务本身稳定、可测试、无额外云模型鉴权。

### 动态分类词库机制

为了避免把所有历史术语一次性塞进模型上下文，系统采用“按场景 + 按内容 + 显式类别”的动态词库加载策略：

1. **分类存储**：词库按类别保存到 `data/glossary/*.json`，例如 `global`、`ai_chip`、`ai_agent`、`nas`、`business`、`people`、`product`。
2. **按需加载**：`recorder_prepare_review` 会根据 `scene`、标题、转写内容和可选 `glossary_categories` 只加载相关类别，避免上下文臃肿。
   - 如需把用户个人/团队词库放到仓库外，可设置 `RECORDER_AI_GLOSSARY_DIR=/path/to/glossary`。
3. **候选提取**：review package 会输出 `glossaryMatches`、`termCandidates`、`lowConfidenceTerms`，帮助智能体优先处理专名和低置信片段。
4. **确认写回**：`recorder_apply_review` 会读取智能体返回的 `confirmedTerms`，仅把高置信且不需要人工确认的词写入对应分类词库。
5. **持续学习**：同一用户/项目反复使用后，专有词、人名、项目名、产品名会逐步沉淀，后续同类会议识别和校准提示会更准。

词库条目示例：

```json
{
  "term": "昇腾编译器",
  "aliases": ["Ascend Compiler", "升腾编译器"],
  "category": "ai_chip",
  "priority": 8,
  "frequency": 3,
  "source": "workbuddy_review",
  "notes": "由 WorkBuddy 智能体校准后确认写入。"
}
```

`confirmedTerms` 示例：

```json
{
  "confirmedTerms": [
    {
      "term": "昇腾编译器",
      "raw": "升腾编译器",
      "aliases": ["Ascend Compiler"],
      "category": "ai_chip",
      "confidence": 0.94,
      "needHumanConfirm": false,
      "reason": "会议明确讨论芯片工具链。"
    }
  ]
}
```

如果 `needHumanConfirm=true` 或 `confidence < 0.72`，系统不会自动写入词库，只会保留为待确认项。

### 词库交互工具使用方式

MCP 本身不是复杂交互 UI，词库确认应由 WorkBuddy 智能体在对话层组织。推荐交互是：MCP 给出候选，WorkBuddy 展示给用户确认，用户确认后 WorkBuddy 再调用 MCP 写回。

#### 1. 查询已有词库：`recorder_glossary_list`

用途：查看某些分类下已有术语，或按关键词搜索。

参数：

```json
{
  "categories": "ai_chip,ai_agent",
  "keyword": "编译器",
  "limit": 50
}
```

返回重点字段：

```json
{
  "ok": true,
  "glossaryDir": ".../data/glossary",
  "categories": ["ai_chip", "ai_agent"],
  "terms": [
    {
      "term": "昇腾编译器",
      "aliases": ["Ascend Compiler", "升腾编译器"],
      "category": "ai_chip",
      "priority": 8,
      "frequency": 3
    }
  ]
}
```

#### 2. 从录音结果生成候选：`recorder_glossary_suggest`

用途：基于已完成转写 job 或 project JSON，生成词库命中、疑似专名候选、低置信术语。

参数二选一：

```json
{
  "job_id": "rec-xxxx",
  "categories": "ai_chip,ai_agent",
  "limit": 40
}
```

或：

```json
{
  "project_json": "/path/to/project.json",
  "categories": "ai_chip,ai_agent",
  "limit": 40
}
```

返回重点字段：

```json
{
  "glossaryMatches": [],
  "termCandidates": [
    {
      "term": "昇腾编译器",
      "count": 1,
      "suggestedCategory": "ai_chip",
      "reason": "高频或形态上疑似专名/术语，建议由智能体结合上下文确认后写入词库。"
    }
  ],
  "lowConfidenceTerms": []
}
```

WorkBuddy 应把这些候选整理成自然语言给用户确认，而不是自动全部写入。

#### 3. 确认写入词库：`recorder_glossary_confirm`

用途：用户确认某些候选后，把它们写入对应分类词库。

参数：

```json
{
  "terms_json": "{\"terms\":[{\"term\":\"昇腾编译器\",\"raw\":\"升腾编译器\",\"aliases\":[\"Ascend Compiler\"],\"category\":\"ai_chip\",\"priority\":8,\"notes\":\"AI 芯片工具链相关术语\"}]}",
  "default_category": "meeting"
}
```

返回：

```json
{
  "ok": true,
  "updatedTerms": [
    {
      "term": "昇腾编译器",
      "category": "ai_chip",
      "action": "created",
      "path": ".../ai_chip.json"
    }
  ]
}
```

#### 4. 拒绝候选：`recorder_glossary_reject`

用途：用户确认某些候选不应写入词库时，记录到 `_rejected.json`，避免后续反复推荐。

参数：

```json
{
  "items_json": "{\"items\":[{\"raw\":\"新新型投资\",\"suggested\":\"新型投资\",\"category\":\"business\"}]}",
  "reason": "上下文不足，暂不写入"
}
```

#### 5. 直接新增或更新词条：`recorder_glossary_update`

用途：用户明确要求“把某个词加入某类词库”时直接调用。

参数：

```json
{
  "term": "MCP",
  "category": "ai_agent",
  "aliases": "模型上下文协议,Model Context Protocol",
  "notes": "WorkBuddy 连接器和智能体工具场景常用术语",
  "priority": 9
}
```

### WorkBuddy 词库交互示例

用户可以自然说：

```text
把这次会议里的专名候选列出来，我确认后写入词库。
```

WorkBuddy 推荐执行：

1. 调用 `recorder_glossary_suggest(job_id)` 获取候选。
2. 将候选按“建议写入 / 需要确认 / 建议拒绝”展示给用户。
3. 用户确认后：
   - 对确认项调用 `recorder_glossary_confirm`。
   - 对拒绝项调用 `recorder_glossary_reject`。
   - 对用户手动补充项调用 `recorder_glossary_update`。
4. 再调用 `recorder_prepare_review` / `recorder_apply_review` 重新生成更准确的 calibrated 报告。

## MCP 客户端安装与使用说明

本项目的 MCP Server 是标准 stdio MCP 服务，只要客户端支持配置 `command + args + env`，就可以接入。核心原则是：

- MCP 进程只负责本地 ASR、任务状态、文件落盘、review package、词库读写。
- WorkBuddy / Claude / Cursor 等 Agent 客户端负责自然语言交互、提示用户确认、调用自身可用的大模型做语义校准。
- MCP 内部不直接绑定任何云端大模型，也不保存云端模型密钥。
- 长音频必须走 `recorder_transcribe` → `recorder_job_status` → `recorder_job_result`，不要把长任务做成一次阻塞调用。

### 1. WorkBuddy 安装配置

推荐把 MCP 配到用户级 WorkBuddy 配置文件：

```text
~/.workbuddy/mcp.json
```

示例配置：

```json
{
  "mcpServers": {
    "recorder-ai-studio": {
      "command": "/absolute/path/to/python",
      "args": ["/absolute/path/to/recorder-ai-studio/recorder_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/recorder-ai-studio",
        "FUNASR_MODEL": "SenseVoiceSmall",
        "FUNASR_VAD_MODEL": "",
        "FUNASR_PUNC_MODEL": "",
        "FUNASR_CHUNK_SECONDS": "60",
        "FUNASR_BATCH_SIZE_S": "60",
        "FUNASR_KEEPALIVE_SECONDS": "600",
        "RECORDER_AI_MCP_WORKERS": "1"
      }
    }
  }
}
```

如果使用本仓库在当前工作区的路径，可按实际环境写成：

```json
{
  "mcpServers": {
    "recorder-ai-studio": {
      "command": "python",
      "args": ["/Users/neoco/WorkBuddy/2026-07-02-01-30-36/recorder-ai-studio/recorder_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/Users/neoco/WorkBuddy/2026-07-02-01-30-36/recorder-ai-studio",
        "FUNASR_MODEL": "SenseVoiceSmall",
        "FUNASR_KEEPALIVE_SECONDS": "600"
      }
    }
  }
}
```

配置后：

1. 打开 WorkBuddy 的连接器管理页面。
2. 进入右上角自定义连接器入口。
3. 找到 `recorder-ai-studio`。
4. 点击信任 / 启用。
5. 新开一个对话或刷新连接器工具列表。
6. 让 WorkBuddy 调用 `recorder_asr_status` 检查是否可用。

### 2. Claude Desktop 安装配置

Claude Desktop 也使用 stdio MCP。配置示例：

```json
{
  "mcpServers": {
    "recorder-ai-studio": {
      "command": "python",
      "args": ["/absolute/path/to/recorder-ai-studio/recorder_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/recorder-ai-studio",
        "FUNASR_MODEL": "SenseVoiceSmall",
        "FUNASR_KEEPALIVE_SECONDS": "600"
      }
    }
  }
}
```

使用时建议向 Claude 明确说明：

```text
请使用 recorder-ai-studio MCP 工具处理音频。长音频必须先提交异步任务，再查询状态，完成后再获取结果，不要单次阻塞等待。
```

### 3. Cursor / 其他支持 MCP 的 Agent 客户端

对于 Cursor、Cline、Continue 或其他支持 MCP 的 Agent 客户端，配置思路一致：

```json
{
  "mcpServers": {
    "recorder-ai-studio": {
      "command": "python",
      "args": ["/absolute/path/to/recorder-ai-studio/recorder_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/recorder-ai-studio",
        "FUNASR_MODEL": "SenseVoiceSmall",
        "FUNASR_KEEPALIVE_SECONDS": "600",
        "RECORDER_AI_MCP_JOB_DIR": "/absolute/path/to/outputs/mcp-jobs"
      }
    }
  }
}
```

注意：

- 如果客户端只显示旧工具 schema，重启客户端或重新加载 MCP 连接器。
- 如果长任务导致客户端超时，说明没有按异步三段式调用，应改用 `submit/status/result` 流程。
- 如果 stdout 被污染导致 MCP JSON-RPC 失败，请确认没有在 `recorder_mcp_server.py` 主进程里 `print` 普通日志。本项目已经把转写日志重定向到 job log 文件。
- 如果运行环境找不到依赖，请先在本仓库环境安装 `requirements.txt`，并保证 MCP 配置的 `command` 指向同一个 Python 环境。

## Agent 与模型说明

### 模型分工

| 环节 | 使用模型 / 能力 | 运行位置 | 是否由本项目内置调用 |
| --- | --- | --- | --- |
| 语音转文字 | FunASR `SenseVoiceSmall` | 本地机器 | 是 |
| 初步结构化整理 | 规则 + 本地项目逻辑 | 本地机器 | 是 |
| HTML/Markdown/JSON 报告 | 本地模板与结构化渲染 | 本地机器 | 是 |
| 专名校准、错字修正、纪要重写 | WorkBuddy / Claude / Cursor 当前 Agent 使用的大模型 | Agent 客户端环境 | 否，由客户端调用 |
| 动态词库选择与写回 | 本地 glossary 逻辑 + 用户/Agent 确认 | 本地机器 + 对话层 | 是 |

### 为什么 MCP 不直接调用云端模型

本项目刻意不在 MCP Server 里硬编码云端模型调用，原因是：

1. MCP Server 更稳定：不需要额外处理云模型鉴权、限流、账单和网络错误。
2. Agent 更灵活：WorkBuddy、Claude、Cursor 可以用当前会话已配置的大模型能力处理语义校准。
3. 用户上下文更完整：Agent 对话层知道用户偏好、项目背景、术语确认结果，更适合做内容整理。
4. 工具边界更清晰：MCP 负责确定性文件和任务，Agent 负责非确定性的语言理解和生成。

因此推荐工作方式是：

```text
本地 MCP：真实 ASR + 文件落盘 + review package + glossary
当前 Agent：大模型语义校准 + 用户确认 + 结构化纪要生成
本地 MCP：写回 calibrated 报告 + 更新词库
```

### 关键环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FUNASR_MODEL` | `SenseVoiceSmall` | 本地 ASR 模型。当前推荐保持默认。 |
| `FUNASR_KEEPALIVE_SECONDS` | `600` | 模型加载后空闲保活秒数。10 分钟内新任务复用模型。 |
| `FUNASR_CHUNK_SECONDS` | `60` | 长音频分片长度。 |
| `FUNASR_BATCH_SIZE_S` | `60` | FunASR 批处理参数。 |
| `FUNASR_VAD_MODEL` | 空 | 默认关闭额外 VAD，减少依赖和冷启动成本。 |
| `FUNASR_PUNC_MODEL` | 空 | 默认关闭额外标点模型。 |
| `RECORDER_AI_MCP_JOB_DIR` | `../outputs/mcp-jobs` | MCP job、日志和结果输出目录。 |
| `RECORDER_AI_MCP_WORKERS` | `1` | MCP 后台转写 worker 数。长音频建议先保持 1，避免本地资源争抢。 |
| `RECORDER_AI_GLOSSARY_DIR` | `data/glossary` | 外部分类词库目录，可用于团队/个人私有词库。 |

## MCP 工具逐个使用说明

### `recorder_asr_status`

用途：检查本地 FunASR / SenseVoiceSmall 模型和运行时保活状态。

适合场景：

- 首次安装后检查 MCP 是否可用。
- 转写前确认模型权重是否准备好。
- 查看模型是否已加载、是否还在 10 分钟保活窗口内。

自然语言示例：

```text
请检查 recorder-ai-studio 的本地 ASR 模型状态，告诉我模型是否已加载、是否可用、当前是否在保活窗口内。
```

### `recorder_release_model`

用途：立即释放当前 MCP 进程中已加载的本地 ASR 模型。

自然语言示例：

```text
请释放 recorder-ai-studio 当前加载的本地 ASR 模型，然后再告诉我释放后的状态。
```

### `recorder_transcribe`

用途：提交非阻塞真实转写任务，立即返回 `jobId`。

关键参数：

```json
{
  "audio_path": "/path/to/audio.mp3",
  "title": "内部会议",
  "scene": "meeting",
  "glossary": "芯片,工具链,Agent,MCP",
  "output_dir": "/path/to/outputs"
}
```

自然语言示例：

```text
请使用 recorder-ai-studio 对这个音频文件发起异步转写任务：

/path/to/audio.mp3

要求：不要阻塞等待完整转写结束，只提交任务并返回 jobId。
```

### `recorder_job_status`

用途：查询转写任务状态。

状态值：

- `queued`：任务已提交，等待后台 worker。
- `running`：正在加载模型或转写。
- `completed`：已完成，可获取结果。
- `failed`：失败，可查看 `error` 和 `logPath`。

自然语言示例：

```text
请查询 recorder-ai-studio 里这个转写任务的进度：

jobId: rec-xxxx

请告诉我当前状态、是否完成、日志路径和已生成的输出。
```

### `recorder_job_result`

用途：任务完成后获取完整输出。

应重点检查：

- `outputs.projectJson`
- `outputs.markdown`
- `outputs.htmlReport`
- `outputs.report`

自然语言示例：

```text
请获取 recorder-ai-studio 里这个任务的完整结果：

jobId: rec-xxxx

请重点检查是否包含 projectJson、markdown、report 和 htmlReport，并告诉我 HTML 报告路径。
```

### `recorder_prepare_review`

用途：为当前 Agent 准备校准包，不直接调用大模型。

关键输入：

```json
{
  "job_id": "rec-xxxx",
  "max_segments": 120,
  "glossary_categories": "global,ai_chip,ai_agent"
}
```

输出：

- `reviewPackagePath`
- `reviewPromptPath`
- `reviewPackage`
- `activeGlossaryCategories`
- `termCandidateCount`
- `lowConfidenceCount`

自然语言示例：

```text
请基于这个 recorder-ai-studio 转写任务准备 WorkBuddy 校准包：

jobId: rec-xxxx

要求生成 reviewPackage 和 reviewPrompt，不要直接改写项目。
```

### `recorder_apply_review`

用途：将当前 Agent 输出的校准 JSON 写回项目，生成 calibrated 报告。

关键输入：

```json
{
  "job_id": "rec-xxxx",
  "review_json": "{...}",
  "suffix": "calibrated"
}
```

输出：

- `outputs.calibratedProjectJson`
- `outputs.calibratedMarkdown`
- `outputs.calibratedHtmlReport`
- `outputs.calibratedReport`

自然语言示例：

```text
请把刚才生成的 calibratedReview 写回 recorder-ai-studio 对应项目，并生成校准版 HTML、Markdown 和 JSON 报告。
```

### `recorder_glossary_list`

用途：查看当前分类词库。

自然语言示例：

```text
请查看 recorder-ai-studio 当前可用的动态词库分类，并列出 global、ai_chip、ai_agent 每类大概包含多少术语。
```

### `recorder_glossary_suggest`

用途：从已完成任务或 project JSON 中提取词库候选。

自然语言示例：

```text
请基于这个 jobId 提取会议里的专名候选和低置信术语，但不要自动写入词库，先让我确认。

jobId: rec-xxxx
```

### `recorder_glossary_confirm`

用途：把用户确认过的术语写入分类词库。

自然语言示例：

```text
请把以下术语确认加入 recorder-ai-studio 的 ai_agent 词库：MCP 异步任务、WorkBuddy 校准流程、HTML 校准报告。要求写入后返回结果。
```

### `recorder_glossary_reject`

用途：把用户拒绝的候选记录到拒绝列表，避免后续反复推荐。

自然语言示例：

```text
请将候选术语“随便测试词”标记为拒绝，不要加入 recorder-ai-studio 词库，后续相似内容不要反复推荐。
```

### `recorder_glossary_update`

用途：手动新增或更新一个词库条目。

自然语言示例：

```text
请更新 recorder-ai-studio 词库中的术语：

分类：ai_agent
术语：MCP 异步任务
别名：异步 job、异步转写任务、非阻塞转写
说明：用于避免长音频转写阻塞 WorkBuddy MCP 连接。
```

## WorkBuddy 可复制测试 Prompt

下面的测试 Prompt 可以直接复制到 WorkBuddy 中执行。请把示例音频路径替换成自己的真实文件：

```text
/Users/neoco/Documents/asr原始文件/04-24 内部会议. 芯片与工具链进展.mp3
```

### 基础连通性

#### Prompt 1：检查 ASR 服务状态

```text
请检查 recorder-ai-studio 的本地 ASR 模型状态，告诉我模型是否已加载、是否可用、当前是否在保活窗口内。
```

#### Prompt 2：释放本地模型

```text
请释放 recorder-ai-studio 当前加载的本地 ASR 模型，然后再告诉我释放后的状态。
```

### 异步转写任务

#### Prompt 3：提交转写任务

```text
请使用 recorder-ai-studio 对这个音频文件发起异步转写任务：

/Users/neoco/Documents/asr原始文件/04-24 内部会议. 芯片与工具链进展.mp3

要求：
1. 不要阻塞等待完整转写结束；
2. 只提交任务；
3. 返回 jobId；
4. 告诉我后续如何查询进度。
```

#### Prompt 4：查询任务进度

```text
请查询 recorder-ai-studio 里这个转写任务的进度：

jobId: 这里替换成刚才返回的 jobId

请告诉我：
1. 当前任务状态；
2. 是否完成；
3. 如果还没完成，大概已经到了哪个阶段；
4. 如果完成，输出里有哪些文件或结果。
```

#### Prompt 5：获取转写结果

```text
请获取 recorder-ai-studio 里这个任务的完整结果：

jobId: 这里替换成刚才返回的 jobId

请重点检查：
1. 是否有 projectJson；
2. 是否有 markdown；
3. 是否有 report；
4. 是否有 htmlReport；
5. 如果有 HTML 报告，请告诉我文件路径。
```

### 端到端自动转写 + HTML 报告

#### Prompt 6：完整跑一次并检查 HTML

```text
请使用 recorder-ai-studio 完成一次端到端测试：

音频文件：
/Users/neoco/Documents/asr原始文件/04-24 内部会议. 芯片与工具链进展.mp3

测试要求：
1. 提交异步转写任务；
2. 等任务完成后获取结果；
3. 检查是否生成 HTML 报告；
4. 检查报告里是否包含转写正文、摘要、议题、重点、待办；
5. 最后把生成的 HTML 报告路径告诉我。
```

### WorkBuddy 智能体校准流程

#### Prompt 7：准备校准包

```text
请基于这个 recorder-ai-studio 转写任务准备 WorkBuddy 校准包：

jobId: 这里替换成已完成的 jobId

要求：
1. 生成 reviewPackage；
2. 生成 reviewPrompt；
3. 告诉我校准包里包含哪些内容；
4. 不要直接改写项目，只准备校准材料。
```

#### Prompt 8：让 WorkBuddy 进行内容校准

```text
请基于刚才 recorder-ai-studio 生成的 reviewPackage，对会议内容进行智能校准。

校准要求：
1. 修正明显的 ASR 错字；
2. 统一专有名词；
3. 保留原意，不要编造；
4. 提取核心议题；
5. 提取关键结论；
6. 提取行动项；
7. 输出结构化的 calibratedReview JSON，方便后续写回 recorder-ai-studio。
```

#### Prompt 9：写回校准结果并生成校准版 HTML

```text
请把刚才生成的 calibratedReview 写回 recorder-ai-studio 对应项目，并生成校准版报告。

要求：
1. 写回 calibratedProjectJson；
2. 生成 calibratedMarkdown；
3. 生成 calibratedHtmlReport；
4. 告诉我校准版 HTML 报告路径；
5. 检查报告里是否有“原始转写”和“校准后内容”的对照或校准痕迹。
```

### 动态词库

#### Prompt 10：查看当前词库

```text
请查看 recorder-ai-studio 当前可用的动态词库分类。

要求：
1. 列出所有词库分类；
2. 每个分类显示大概包含多少个术语；
3. 告诉我全局词库和会议类别词库分别有哪些。
```

#### Prompt 11：根据会议内容建议词库候选

```text
请根据下面这段会议内容，让 recorder-ai-studio 给出可能需要加入词库的候选术语：

会议内容：
我们今天讨论了 SenseVoiceSmall、FunASR、MCP 异步任务、Agent 工具链、芯片工具链、模型保活策略和 HTML 校准报告。

要求：
1. 自动判断适合的词库分类；
2. 给出候选术语；
3. 说明每个术语为什么值得加入；
4. 不要直接写入词库，先让我确认。
```

#### Prompt 12：确认加入词库

```text
请把以下术语确认加入 recorder-ai-studio 的动态词库：

分类：ai_agent

术语：
1. MCP 异步任务
2. WorkBuddy 校准流程
3. HTML 校准报告

要求：
1. 写入词库；
2. 标记为 confirmed；
3. 返回写入后的结果；
4. 告诉我下次会议转写时这些词会如何被使用。
```

#### Prompt 13：拒绝词库候选

```text
请将下面这个词库候选标记为拒绝，不要加入 recorder-ai-studio 词库：

候选术语：随便测试词
分类：ai_agent

要求：
1. 标记为 rejected；
2. 后续相似内容不要反复推荐；
3. 返回处理结果。
```

#### Prompt 14：更新已有词库术语

```text
请更新 recorder-ai-studio 词库中的这个术语：

分类：ai_agent
术语：MCP 异步任务

更新内容：
1. 标准写法：MCP 异步任务
2. 别名：异步 job、异步转写任务、非阻塞转写
3. 说明：用于避免长音频转写阻塞 WorkBuddy MCP 连接。

请更新后返回该术语的完整词库记录。
```

### 分类词库加载

#### Prompt 15：测试按会议主题加载词库

```text
请模拟 recorder-ai-studio 对下面会议标题和内容进行词库选择，不要转写音频，只判断应该加载哪些词库分类。

会议标题：
芯片与 Agent 工具链进展会议

会议内容：
会议讨论了芯片工具链、SenseVoiceSmall、FunASR、MCP、WorkBuddy、异步转写任务、模型保活和 HTML 报告。

要求：
1. 告诉我会加载哪些词库分类；
2. 说明为什么加载这些分类；
3. 说明哪些词库不会加载，以及为什么不加载；
4. 重点确认是否避免了无关词库污染上下文。
```

### 完整高质量会议纪要验收

#### Prompt 16：从音频到高质量校准版 HTML

```text
请使用 recorder-ai-studio 对下面音频做一次完整高质量测试：

音频文件：
/Users/neoco/Documents/asr原始文件/04-24 内部会议. 芯片与工具链进展.mp3

完整流程要求：
1. 先检查本地 ASR 状态；
2. 提交异步转写任务；
3. 查询任务直到完成；
4. 获取原始转写结果；
5. 检查是否生成原始 HTML 报告；
6. 准备 WorkBuddy 校准包；
7. 基于校准包进行智能校准；
8. 写回校准结果；
9. 生成校准版 HTML 报告；
10. 检查动态词库候选；
11. 给出建议加入词库的专有名词，但不要自动确认；
12. 最后输出：
    - jobId
    - 原始 HTML 报告路径
    - 校准版 HTML 报告路径
    - 发现的专有名词候选
    - 会议摘要
    - 核心议题
    - 行动项
```

### 异常场景

#### Prompt 17：不存在的音频路径

```text
请使用 recorder-ai-studio 转写这个不存在的音频文件：

/Users/neoco/Documents/asr原始文件/not-exist.mp3

要求：
1. 不要崩溃；
2. 返回清晰错误原因；
3. 告诉我应该检查什么。
```

#### Prompt 18：查询不存在的 jobId

```text
请查询 recorder-ai-studio 中这个不存在的任务：

jobId: fake-job-id-123

要求：
1. 返回明确错误；
2. 不要创建新任务；
3. 不要返回旧任务结果。
```

#### Prompt 19：释放模型后查询历史结果

```text
请释放 recorder-ai-studio 的本地 ASR 模型，然后查询我之前完成的这个任务结果：

jobId: 这里替换成已完成的 jobId

要求：
1. 确认模型释放成功；
2. 确认历史任务结果仍然可以读取；
3. 确认不会因为释放模型而丢失 HTML 报告。
```

### 推荐测试顺序

```text
1. Prompt 1：检查状态
2. Prompt 2：释放模型
3. Prompt 3：提交转写
4. Prompt 4：查进度
5. Prompt 5：取结果
6. Prompt 7：准备校准包
7. Prompt 8：WorkBuddy 校准
8. Prompt 9：写回校准结果
9. Prompt 10-14：测试词库
10. Prompt 16：完整验收
11. Prompt 17-19：异常测试
```

如果只想快速验收，建议执行：

```text
Prompt 1：检查状态
Prompt 3：提交异步转写
Prompt 16：完整高质量流程
```

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
- 已支持“用户确认后的术语”回写到本地分类 glossary，下次同类会议自动优先使用。
- 已支持按 `scene`、标题和正文动态加载部分词库，避免把无关术语塞入上下文。
- 已在 `recorder_prepare_review` 输出词库命中、疑似专名候选和低置信术语，交给 WorkBuddy 智能体集中校准。
- 后续可继续增强同音误识别候选建议，例如把“新新型投资”提示为“新型投资/新兴投资/新型投入”等待确认项。

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

- 已增加“原文 vs 校准后”差异视图，突出删除口语词、修正错别字、补全语义的位置，并展示专名校准与词库写回状态。
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
- MCP 词库交互工具组已接入，可查询、建议、确认、拒绝、更新分类词库。
- 动态分类词库、词库命中、疑似专名候选、校准后术语写回已接入。
- HTML 报告已展示动态词库、专名校准、原文/校准后对比和人工复核片段。
- 完整测试套件通过。
