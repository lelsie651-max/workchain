# WorkChain

WorkChain 是一个职场证据链工具，用于把聊天截图或文字中的工作要求整理为可追溯的事项线。

## 目录

```text
workchain/
  app/
  evidence_core/
  scripts/
  tests/
  verify.py
```

## 本地运行

```bash
pip install -r requirements.txt
python -m app.main
```

## 环境变量

- `DEEPSEEK_API_KEY=`: 仅用于 `/api/diag/llm` 连通性自检，帮助确认部署环境能否访问 DeepSeek API。绝不要提交真实 key。
