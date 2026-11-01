# 测试与验证

## 保留的测试

`python -m unittest discover -s tests -v` 使用隔离临时目录和明确开启的测试引擎，不使用GPU。覆盖鉴权、数据版本、错误数据、幂等、并发、日志、答题、取消、API重启、孤儿引擎回收和非有限指标处理。

`scripts/smoke_execution.py` 为真实模型/GPU验证脚本，完整命令见README。报告写到忽略的reports目录。测试引擎产生的文件不是模型。

`python scripts/generate_demo.py reports/demo-data` 可重新生成演示数据和缺字段、空文本、超长文本等错误样本。仓库只保留少量正常样本。

## 历史验证

2026年10月23日的环境已完成9项集成测试及19项网页/下载检查，包含8步真实LoRA训练、新适配器答题、下载校验与取消资源释放。这是历史工程验证记录，不代表新环境自动通过，也不代表航天领域能力达标。旧机器报告和任务编号不随仓库发布。

## 原生推理服务（可选）

此后端不包含原生聊天服务的启动器。先独立安装、配置并启动LLaMAFactory OpenAI风格API，再执行：

```bash
python scripts/smoke_inference.py --base-url http://127.0.0.1:8000 \
  --key-file /你的私有目录/inference.key --report reports/inference-smoke.json
```

也可用LLAMAFACTORY_API_KEY环境变量提供其密钥。`--snapshot contracts/v1/inference.openapi.json`会严格比较历史参考定义，仅在确认实际部署版本匹配时使用。

验证脚本会调用聊天和流式生成，消耗相应推理资源，不会打印密钥。
