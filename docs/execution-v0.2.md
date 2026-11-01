# 执行服务 0.2.0：部署与联调

首次通过浏览器操作，请使用 [Swagger 逐步操作指南](swagger-walkthrough.md)，包含按钮、参数和完整 JSON。

本版本以演示数据验证后端执行流程，不代表航天领域效果达标。业务排队、题目审核、评分与报告由中台负责。

## 调用地址与认证

- 服务器：`http://127.0.0.1:8010`，仅监听服务器回环地址。
- OpenAPI：`GET /openapi.json`；Swagger：`GET /docs`；健康检查：`GET /healthz`。
- 业务接口统一要求 `Authorization: Bearer <服务密钥>`、`X-Project-ID`、`X-Actor-ID`。
- 当前演示授权绑定项目 `demo-project` 和操作者 `demo-user`，不能通过任意修改这两个请求头扩权。
- 私有配置：`~/.config/aerospace-backend/config.json`。
- 服务密钥：`~/.config/aerospace-backend/service.key`。密钥不进入源码、前端或验证报告；与原推理服务密钥独立。
- 中台在服务器端读取凭据并代理请求；前端不直接持有服务密钥。

需要在本机查看时，手动建立独立转发：

```powershell
ssh -N -L 127.0.0.1:18010:127.0.0.1:8010 YOUR_USER@YOUR_SERVER
```

之后访问 `http://127.0.0.1:18010/docs`。Swagger 的 Authorize 填服务密钥，每次业务调用填写项目和操作者头；创建执行另填 Idempotency-Key。

## 第一版支持范围

| 能力 | 当前行为 |
|---|---|
| 模型 | 注册的 Qwen2.5-1.5B 基础模型，以及本服务成功训练的适配器 |
| 训练 | 单卡、LoRA、FP16、SDPA；不开放 QLoRA/全量/多卡 |
| 资源 | 仅 node01 / GPU 0；全服务同时最多 1 个训练或预测执行 |
| 冲突 | 存在活动执行返回 429；GPU 已有计算进程或空闲显存不足返回 503；不自动停止其他服务 |
| 训练参数 | cutoff_len 32–1024，batch 1–2，梯度累积 1–64，rank 1–32，最多 1000 更新步 |
| 文件 | 每个文件最多 8 MiB，文件名不用于决定存储路径，返回 SHA-256 |
| 数据 | UTF-8 JSON 数组；Alpaca 或 question_set_v1；ShareGPT 尚不支持 |
| 产物 | 适配器 ZIP、运行参数与来源记录、经过过滤的日志、逐题预测 JSONL |
| 模型导出 | 本版本下载 LoRA 适配器；HTTP 模型合并导出接口尚未提供 |

GPU 空闲检测是本服务的准入检查，不能替代共享服务器上的跨用户资源调度。网页和原推理服务如果占用 GPU 0，需要按已批准的资源安排手动释放后再下发执行。

## 数据格式

Alpaca 每条记录必须有非空字符串 `instruction`、`output`，可选 `input`、`system`。第一版不接受 history 等额外字段。每个文本字段最多 32768 字符。示例：

```json
[{"instruction":"计算1加2，只返回数字。","input":"","output":"3"}]
```

批量问答使用 question_set_v1，必须有唯一 `question_id` 和非空 `prompt`，可选 `system`、`answer`、`question_type`。`answer` 只随数据保存，不用于后端评分：

```json
[{"question_id":"demo_q1","prompt":"计算17加25，只返回数字。","answer":"42"}]
```

每个数据文件至少 1 条，最多 100000 条（同时受文件大小限制）。注册时检查字段、空文本、超长文本、重复 ID、重复输入和跨 split 的相同输入；这不能替代中台的语义去重、数据审核与业务划分。预测单题超过本版本 1024 token 上下文预算时返回该题错误，继续处理其他题。

文件 purpose 与 split 对应：train 使用 training_data，validation/test 使用 evaluation_data 或 prediction_data。question_set_v1 不能注册为训练 split。

## 调用顺序

1. `GET /internal/v1/capabilities` 和 `GET /internal/v1/models`。
2. `POST /internal/v1/files`，multipart 字段为 `file`、`purpose`，取得 file_id。
3. `POST /internal/v1/dataset-versions`，固定 split、文件哈希、样本数。相同项目内 dataset_id + version 不可覆盖。
4. `POST /internal/v1/training-runs`，使用现有 `examples/create-training-run.json` 的结构，替换数据版本 ID。
5. 按 `status_url` 查询状态；`GET .../{run_id}/logs?cursor=...&limit=100` 读取增量日志。
6. 成功时读取 output_model_id、artifact_ids；`GET /internal/v1/artifacts/{artifact_id}/content` 下载，核对 `X-Content-SHA256`。
7. 上传并注册问答集，`POST /internal/v1/prediction-runs` 引用刚产生的 output_model_id，生成后按 result_artifact_id 下载逐题结果。

批量执行 succeeded 表示已生成逐题记录，不代表所有题都答题成功；调用方需要检查每条记录的 error。模型 output_model_id 是登记结果，不会自动替换原生 8000 端口正在服务的模型。

示例数据可由 `python scripts/generate_demo.py <目录>` 生成，包含训练、评估、问答，以及缺字段、空内容、超长内容 3 类错误文件。

## 状态、幂等与取消

`accepted → running → succeeded / failed`。取消流程为 `cancel_requested → cancelled`。执行进程退出后才能进入 cancelled；进入产物发布阶段的取消返回 409，防止已发布模型又被标成取消。

创建训练和预测必须提供 1–128 个可打印字符的 Idempotency-Key。同一服务、项目、操作者、执行类型内，相同 key 与相同请求体返回同一执行 ID（包含终态回放）；同 key 不同请求体返回 409。当前记录不自动清理，保留至少 24 小时。中台重试失败任务需创建新 key、新执行 ID，不复用旧目录。

业务排队由中台负责。后端不把资源冲突静默加入内部长队列。

## 持久化与恢复边界

- SQLite WAL 保存文件、数据版本、模型、任务、幂等记录和日志。
- 数据、中间结果与数据库：`~/.local/share/aerospace-backend`；最终适配器：`~/models/aerospace-backend`。
- API 重启不会终止独立执行进程，新 API 继续查询同一任务。
- 执行进程异常退出时，验证 PID、启动时间、账号及命令标识，再回收其引擎进程，任务标记 failed。不会静默重跑半成品。
- 服务器重启后需手动启动 API；中断任务会在启动后被识别为失败。尚未实现自动断点续训和开机自启动。
- 最终适配器保存前检查权重有限且确有 LoRA B 更新。非有限训练指标会导致失败，不向 JSON 输出 NaN。
- 尚未实现数据库与模型异地备份、完整灾难恢复；需作为后续独立交付。

## 运行命令

```bash
cd ~/projects/aerospace-llm-backend
PY=python
CFG="$HOME/.config/aerospace-backend/config.json"
$PY scripts/service.py status --config "$CFG"
$PY scripts/service.py start --config "$CFG"
$PY scripts/service.py restart --config "$CFG"
$PY scripts/service.py stop --config "$CFG"
```

stop 停止 API 接收请求，独立训练继续执行。需要取消训练时先调用 cancel 并等到 cancelled，再停服务。禁止同时使用多个 uvicorn worker；实例锁会拒绝第二个执行管理器。

## 验证与剩余工作

`python -m unittest discover -s tests -v` 使用显式测试引擎验证 HTTP、数据库和子进程生命周期，不消耗 GPU；测试产物均为 fixture，不是模型。

`scripts/smoke_execution.py` 使用真实 GPU/模型执行端到端验证，包含训练、适配器更新检查、通过新适配器答题、API 重启和真实训练取消。它会消耗 GPU，需要先安排资源；失败时会请求取消本次验证创建的活动任务。

服务间授权允许列表是第一版边界，仍需和中台对齐实际用户体系、角色、资源授权、审计与密钥轮换。完整敏感输出检测、航天领域题库效果、通用能力保持、模型合并 HTTP 接口、备份恢复和设备扩展均尚待实现或验收。
