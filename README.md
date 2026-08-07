# WorkChain

WorkChain 是一个职场证据链工具，用于把聊天截图或文字中的工作要求整理为可追溯的事项线。

本次提交仅完成工程初始化与 `evidence_core/canonical.py`，不包含 UI、LLM、Web 框架、OCR、数据库或哈希逻辑。

## 目录

```text
workchain/
  evidence_core/
    __init__.py
    canonical.py
  tests/
    test_canonical.py
  blobs/.gitkeep
  .gitignore
  .env.example
  README.md
  requirements.txt
```

## 开发

```bash
pytest
```
