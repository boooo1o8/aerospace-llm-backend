# 航空测控微调平台后端

基于 FastAPI 和 LLaMAFactory 的训练执行服务。为项目自己的前端、中台提供数据上传、数据版本、训练任务、进度日志、取消、模型下载与批量答题接口。

训练和推理由 **LLaMAFactory** 完成；本项目新增的是围绕数据、任务和产物的后端管理。`/docs` 是 Swagger 接口调试页面，**不是自研操作前端**。当前版本 `0.2.0`，正式航天领域效果、完整用户权限及业务端联调尚未验收。

## 已实现的功能

| 部分 | 功能 |
|---|---|
| 文件与数据 | 文件上传、SHA-256、数据格式检查、固定训练/验证/测试文件的数据版本 |
| 模型管理 | 基础模型目录、成功训练适配器登记，以模型 ID 调用 |
| 训练管理 | 单卡 LoRA 训练、进度和指标、增量日志、取消后退出执行进程 |
| 结果管理 | 下载适配器 ZIP、训练参数和数据来源记录、日志 |
| 批量答题 | 指定模型处理问答集，逐题保存原始回答、用量和错误 |
| 运行控制 | 请求幂等、单任务准入、GPU 占用检查、SQLite 持久化、API 重启及异常进程回收 |
| 调用校验 | Bearer 服务密钥、绑定项目和操作者允许范围，按项目校验资源访问 |

调用关系：`自有前端 → 中台 → 本服务 → LLaMAFactory`。中台负责业务排队、用户权限、题库审核、评分与报告；本服务执行训练和答题，返回实际结果。

## 目录与代码职责

```text
.
├── aerospace_backend/        # 服务实现
│   ├── app.py                # HTTP 接口、请求校验及错误响应
│   ├── core.py               # 配置、数据库、鉴权、文件校验、进程与锁
│   ├── manager.py            # 任务调度检查、独立 worker 启动与异常回收
│   ├── worker.py             # 准备数据、启动引擎、更新进度、保存模型和产物
│   └── engine.py             # 实际调用 LLaMAFactory 训练与答题
├── contracts/v1/             # 执行 API 定义与原生推理 API 参考
├── scripts/                  # 初始化、服务管理、演示数据生成、验证脚本
├── examples/                 # 请求 JSON 与少量可运行演示数据
├── tests/                    # 不使用 GPU 的 HTTP/数据库/子进程集成测试
├── docs/                     # 接口分工、部署限制、Swagger 操作与验证说明
├── config.example.json       # 配置字段模板，不是可直接用于所有机器的配置
└── requirements.txt          # 后端依赖；模型引擎需另外安装 LLaMAFactory
```

## 环境要求

- **Linux**，建议 Ubuntu 22.04、Python 3.11；进程控制使用 `/proc`、`fcntl` 和 Linux 进程组，Windows 原生不支持运行，可维护源码或使用 WSL2。
- 实际训练需要 NVIDIA GPU、驱动及匹配的 CUDA/PyTorch 环境；当前实现为单卡 FP16 + SDPA LoRA。
- 有可读取的 Qwen2.5-1.5B-Instruct 模型目录。模型权重、正式数据和运行结果不包含在仓库中。
- 历史真实验证环境使用 RTX 2080 Ti 11GB、PyTorch 2.6.0+cu124、LLaMAFactory 0.9.6.dev0；换模型或环境需重新检查兼容性。

## 安装与启动

以下命令在 Linux 服务器执行，普通账号即可。仓库默认目录为 `aerospace-llm-backend`。

### 1. 下载本项目并建立环境

```bash
git clone https://github.com/boooo1o8/aerospace-llm-backend.git
cd aerospace-llm-backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

私有仓库需要先为 Git 配置有访问权限的 GitHub 账号或 SSH 凭据。

### 2. 安装 LLaMAFactory

```bash
mkdir -p "$HOME/projects"
git clone https://github.com/hiyouga/LLaMAFactory.git "$HOME/projects/LLaMAFactory"
git -C "$HOME/projects/LLaMAFactory" checkout c683e59edd8f893abc5e1403184a1d0644817d2a
python -m pip install -e "$HOME/projects/LLaMAFactory[torch,metrics]"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

这里使用历史验证过的上游提交。PyTorch 的 CUDA 包和驱动须与机器匹配；显示 `CUDA: False` 时先解决环境问题，再运行真实训练。

### 3. 准备基础模型

已有本地模型时直接使用其绝对路径。需要下载时：

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',
                  local_dir=str(Path.home() / 'models/Qwen2.5-1.5B-Instruct'))
PY
```

下载需要相应网络访问，模型使用遵守其上游许可。

### 4. 初始化私有配置

```bash
python scripts/provision.py \
  --directory "$HOME/.config/aerospace-backend" \
  --project demo-project --actor demo-user \
  --model-path "$HOME/models/Qwen2.5-1.5B-Instruct" \
  --llamafactory-root "$HOME/projects/LLaMAFactory"
```

生成 `config.json`、`credentials.json`、`service.key`、`cursor.key`，文件仅当前账号可读写，不输出密钥内容。已有配置时拒绝覆盖，避免意外换掉现有凭据。

生成配置自动使用当前 Python 解释器。默认数据在 `~/.local/share/aerospace-backend`，最终适配器在 `~/models/aerospace-backend`。如有大容量数据盘，可初始化时加 `--data-root /你的数据盘目录`、`--model-root /你的模型目录`。

启动前检查 `config.json` 的模型、存储路径、`gpu_id` 和 `min_free_gpu_mib`。所有文件路径必须是绝对路径；默认只接受 `node01 / GPU 0`，对应请求也必须一致。

### 5. 启动、检查和停止

```bash
export AEROSPACE_CONFIG="$HOME/.config/aerospace-backend/config.json"
python scripts/service.py start --config "$AEROSPACE_CONFIG"
python scripts/service.py status --config "$AEROSPACE_CONFIG"
curl http://127.0.0.1:8010/healthz
```

- Swagger：<http://127.0.0.1:8010/docs>
- OpenAPI：<http://127.0.0.1:8010/openapi.json>
- 服务日志：配置的 `data_root/api.log`。

```bash
python scripts/service.py restart --config "$AEROSPACE_CONFIG"
python scripts/service.py stop --config "$AEROSPACE_CONFIG"
```

服务只监听服务器回环地址，不向公网开放。API 停止或重启不会自动停止独立训练；要结束训练，先调用取消接口并等到 `cancelled`。不要启动多个 uvicorn worker。

### 6. 在自己电脑访问

在本机终端执行，替换成实际账号和服务器地址，保持终端打开：

```bash
ssh -N -L 127.0.0.1:18010:127.0.0.1:8010 YOUR_USER@YOUR_SERVER
```

然后打开 <http://127.0.0.1:18010/docs>。`8010` 是服务端口，`18010` 是本机转发端口。

## 接口怎么用

1. 在 `/docs` 右上角点 **Authorize**，填初始化生成的 `service.key` 内容，不加 `Bearer` 前缀，也不填 SSH 密码。
2. 每个接口填写 `X-Project-ID: demo-project`、`X-Actor-ID: demo-user`（初始化指定其他值时对应替换）。
3. 展开接口，点 **Try it out**，填写参数后点 **Execute**。
4. 实际结果在 **Server response → Response body**。`Example Value` 只是格式示例，`Schemas` 只是字段定义。

| 接口 | 用途 |
|---|---|
| `GET /internal/v1/capabilities` | 查询支持的方法、数据格式和限制 |
| `POST /internal/v1/files` | 上传文件，取得 file_id |
| `POST /internal/v1/dataset-versions` | 固定数据版本，取得 dataset_version_id |
| `GET /internal/v1/dataset-versions/{dataset_version_id}` | 查看数据文件、样本数和校验信息 |
| `GET /internal/v1/models` | 查询基础模型和成功训练的适配器 |
| `POST /internal/v1/training-runs` | 创建训练，取得 run_id |
| `GET /internal/v1/training-runs/{run_id}` | 查询状态、进度、指标、模型及产物编号 |
| `GET /internal/v1/training-runs/{run_id}/logs` | 读取训练日志，以 next_cursor 继续读取 |
| `POST /internal/v1/training-runs/{run_id}/cancel` | 取消训练，查询到 cancelled 才表示结束 |
| `POST /internal/v1/prediction-runs` | 指定模型批量答题 |
| `GET /internal/v1/prediction-runs/{run_id}` | 查询答题状态和结果文件编号 |
| `GET /internal/v1/artifacts/{artifact_id}/content` | 下载微调权重、来源记录、日志或回答文件 |

完整操作见 [Swagger 操作指南](docs/swagger-walkthrough.md)，前中后台边界见 [对接说明](docs/integration-guide.md)，参数限制及恢复边界见 [执行服务说明](docs/execution-v0.2.md)。

### 一次完整运行

```text
上传 train.json / validation.json → 注册 Alpaca 数据版本
→ 用模型 ID、数据版本 ID 创建训练 → 查询状态和日志
→ succeeded 后保存 output_model_id 与 artifact_ids → 下载微调权重
→ 上传 questions.json，注册 question_set_v1 的 test 数据版本
→ 引用刚训练的 output_model_id 创建答题 → 查询并下载逐题结果
```

请求模板在 `examples/`。所有示例编号都必须换成本次接口实际返回的编号；`training-run-accepted.json` 是响应示例，不能直接当成已有任务。

创建训练和答题必须带 `Idempotency-Key`。同一键和相同请求体返回同一任务，修改请求体要换新键。`202` 仅表示接收；训练成功以 `status=succeeded` 为准。逐题回答还须检查 `error`，不是看到批量任务成功就视为每题都成功。

`train` 是训练数据，`validation` 是验证数据，`test` 是答题测试数据。注册数据版本不会自动划分数据，当前上传单文件最多 8 MiB。训练使用 Alpaca，批量答题使用 question_set_v1。

## 测试和验证

### 不占用 GPU 的集成测试

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

测试使用隔离的假引擎，验证鉴权、数据校验、幂等、并发、日志、取消、重启和异常回收；不下载模型、不接触真实 GPU。**假引擎产物不是模型。**

### 真实 GPU 验证（可选）

先启动服务并确认指定 GPU 空闲：

```bash
python scripts/smoke_execution.py \
  --base-url http://127.0.0.1:8010 \
  --config "$HOME/.config/aerospace-backend/config.json" \
  --key-file "$HOME/.config/aerospace-backend/service.key" \
  --project demo-project --actor demo-user \
  --report reports/execution-smoke.json
```

该脚本会真实训练、检查权重、用新模型答题并验证取消，需要显卡资源。`--restart-api` 还会实际重启 API，仅在已安排的验证窗口使用。原生推理验证方法见 [验证说明](docs/validation.md)。

## 当前限制

- 同时最多1个训练或预测任务；活动执行冲突返回429，GPU被占用或显存不足返回503，不自动排队、不停止其他用户进程。
- 当前开放单卡 LoRA、FP16 + SDPA，尚未开放全量、QLoRA、多卡和国产硬件适配。
- 下载的是适配器权重，推理还需要对应基础模型；尚无 HTTP 模型合并导出。
- 新模型登记不会自动切换独立原生推理服务的模型。
- API 重启可保留独立执行，服务器重启后要手动启动服务；中断任务标记失败，不自动断点续训。
- 服务间鉴权和项目范围校验不等于完整用户 RBAC；业务评分、敏感输出检查、备份和灾难恢复待完善。
- 演示样本仅验证工程流程，尚未证明航天领域能力提升。

## 仓库内容说明

保留源码、接口定义、部署文档、少量演示数据、测试代码和可重复使用的验证脚本。不提交真实密钥、私有配置、内部服务器地址、数据库、模型权重、历史任务报告、缓存、截图及临时构建文件。

上游项目：[LLaMAFactory](https://github.com/hiyouga/LLaMAFactory)。本仓库不复制上游训练框架或模型权重。
