# Swagger 操作指南

先按 README 安装、初始化并启动服务。服务器访问8010/docs，本机转发后访问18010/docs。

## 认证与按钮

Authorize 中填私有 service.key 的内容，不加 Bearer。每次业务请求填写初始化时绑定的 X-Project-ID 和 X-Actor-ID，默认 demo-project / demo-user。

点击带GET或POST的接口横条 → Try it out → 填参数 → Execute。Server response 下的 Response body 才是实际结果，Example Value 是示例，Schemas 是字段说明。未执行时通常只有示例。

## 1. 上传训练与验证文件

POST /internal/v1/files 中选择 examples/demo-data/train.json，purpose选择training_data。返回201后保存file_id。再选择validation.json，purpose选择evaluation_data，保存另一个file_id。

## 2. 注册训练数据版本

POST /internal/v1/dataset-versions 使用 examples/create-dataset-version.json；替换其中train和validation的文件ID。dataset_id为自己的数据名称，version首次填1，以后另建版本递增。同一名称和版本不可覆盖。

执行后保存dataset_version_id，检查train为32条、validation为8条。GET同路径加/{dataset_version_id}可重新查询。

## 3. 创建训练

POST /internal/v1/training-runs 使用 examples/create-training-run.json，替换dataset_version_id，确认base_model_id是模型目录中的实际编号。请求头Idempotency-Key为本次请求唯一名称，如demo-train-001。当前配置下资源使用node01 / GPU 0。

GPU必须空闲，参数示例为8步LoRA。返回202后保存run_id。相同幂等键与相同内容只返回原任务，不会重新训练；新的训练使用新键。

## 4. 查进度和日志

GET /internal/v1/training-runs/{run_id}输入刚才的编号，重复Execute直至succeeded、failed或cancelled。HTTP 200只是查询成功。

GET /internal/v1/training-runs/{run_id}/logs，第一次cursor留空，limit填100。后续将响应的next_cursor填到cursor继续读取。失败原因看failure和日志。

## 5. 下载模型和来源记录

训练成功后保存output_model_id和artifact_ids。GET /internal/v1/artifacts/{artifact_id}/content输入产物编号，Execute后点Download file。训练产物依次为适配器ZIP、provenance.json和日志。

provenance.json包含本次run_id、请求参数、数据版本和文件校验信息，可核对“用了什么数据、怎样训练”。ZIP是微调权重，仍需要基础模型。

## 6. 使用新模型批量答题

上传examples/demo-data/questions.json，purpose填prediction_data。注册另一份数据版本，format填question_set_v1，splits为{"test":"刚上传的file_id"}，使用新的dataset_id或version。

POST /internal/v1/prediction-runs使用examples/create-prediction-run.json，model_id替换为刚训练返回的output_model_id，dataset_version_id替换为刚注册的问答版本，split填test，并填写新的Idempotency-Key。

保存新的run_id，GET /internal/v1/prediction-runs/{run_id}查询至succeeded；用result_artifact_id下载JSONL。每行对应一道题，查看raw_output和error。业务评分由中台负责。

## 7. 可选：取消训练

另建一个max_steps=1000的训练，使用新幂等键，查询到running后POST /internal/v1/training-runs/{run_id}/cancel。继续查询到cancelled再确认停止。不要拿已成功完成的任务演示取消。

## 常见错误

| 状态 | 含义与处理 |
|---|---|
|401|检查当前执行服务密钥并重新Authorize|
|403|项目、操作者或资源不在允许范围|
|404|编号不存在、类型不对或资源不可见|
|409|数据版本重复、幂等内容冲突或任务不可取消|
|422|字段、格式或参数错误，查看error.details|
|429|已有活动任务，等待或按流程取消|
|503|GPU占用或显存不足，先协调资源|

历史任务编号不放进仓库；每个环境使用本次接口实际返回的编号。
