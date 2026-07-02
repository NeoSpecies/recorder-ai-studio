# Recorder AI Studio

Recorder AI Studio 是一个面向长录音场景的本地录音转文字与智能整理工作台。项目目标是把会议、访谈、课程、销售沟通等长音频转化为可编辑、可检索、可导出的结构化知识资产。

当前版本重点验证了“真实可用”的最小闭环：前端页面、FastAPI 后端、本地 FunASR / SenseVoiceSmall 转写、摘要与待办整理、Markdown 导出、自动化测试与真实样例端到端测试。

> 重要原则：项目不使用 mock / fallback 伪造转写结果。模型未就绪或识别失败时会直接返回错误，不生成假文本。

## 核心功能

- 长录音项目管理：创建项目、维护场景、术语表、音频与转写结果。
- 音频上传与播放：支持本地上传音频，并通过后端提供音频播放接口。
- 本地真实转写：默认使用 FunASR 的 SenseVoiceSmall 模型进行本机识别。
- 远程 FunASR 兼容：可通过 `FUNASR_ENDPOINT` 接入远程 OpenAI Whisper 风格接口。
- 说话人/段落结构：转写结果按段落保存，包含开始时间、结束时间、说话人、置信度、标签、原始文本与清洗文本。
- 术语与标签：基于关键词自动提取标签，可覆盖芯片、工具链、编译器、算力、大模型等技术场景。
- 智能整理：生成摘要、待办、脑图节点、风险点等结构化信息。
- 导出能力：支持 Markdown 导出，便于沉淀为会议纪要或知识库材料。
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
├── run_real_asr_test.py               # 真实音频转写测试脚本
├── continue_real_asr_after_model_ready.py
├── server/
│   ├── app.py                         # FastAPI 接口与静态服务
│   └── core.py                        # 项目存储、FunASR 调用、整理与导出逻辑
└── tests/
    ├── test_api.py                    # API 测试
    ├── test_core.py                   # 核心逻辑测试
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
pytest -q --rootdir=. tests/test_api.py tests/test_core.py tests/test_audio_chunks.py
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

## 性能调优方向

下一阶段建议从以下方向优化：

1. 模型常驻与预热：避免每次请求重复初始化 FunASR 模型。
2. 分片并行化：对长音频按时间片并发识别，再按时间戳归并。
3. VAD / 标点策略分层：快速预览使用轻量识别，精修模式再启用 VAD、标点和说话人辅助模型。
4. 进度事件流：为长音频转写增加实时进度、当前分片、预计剩余时间。
5. 断点续跑：已完成分片落盘缓存，失败后只重跑未完成片段。
6. 模型加载池：将 SenseVoiceSmall 加载为进程级单例，减少首段延迟。
7. 导出增强：增加 SRT、VTT、JSONL、Docx 等格式。
8. 云端校对接入：将本地 ASR 结果送入云端模型做术语校正、纪要结构化和任务提取。

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
- 短样本端到端测试通过。
- 完整样例真实转写测试通过。
- Git 初始提交已完成。
- 后续可在此基础上进行性能调优与产品功能增强。
