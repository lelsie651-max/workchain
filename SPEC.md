# WorkChain 项目说明书

> 版本 v1.0 · 最后更新 2026-08-07
> 本文档是项目的唯一事实来源。当 IDE 上下文丢失、记忆偏差或需要交接时,以本文档为准。

---

## 0. 一句话定义

**WorkChain 是一个职场证据链工具:你把聊天截图/文字粘进来,它自动读懂"谁在什么时候要求你做什么",生成不可篡改的证据链,并把散落的记录自动串成一条可追溯的"事项线"。**

演示话术(30 秒):
> 老板说"这不是我要的"。你点开事项线:3月5日他的原话、你的确认回复、3月20日他改需求的截图、你产出的文件——一条完整的链,哈希校验通过,时间戳齐全。

---

## 1. 产品定位

### 1.1 它不是什么
- 不是 todo 工具(Notion / 滴答清单 / 飞书任务)
- 不是笔记工具(印象笔记 / flomo)
- 不是会议纪要 / 周报生成器

### 1.2 它是什么
三个能力合为一体,共用同一条记录:

| 能力 | 用户感知 | 技术落点 |
|---|---|---|
| **可验证性** | "这份记录没被改过" | 哈希链 + 可信时间戳 + 独立验证器 |
| **事项线归并** | "这件破事的来龙去脉" | 实体对齐 + 语义归并 |
| **变更检测** | "需求改了 2 次,你没确认过" | 版本比对 + 风险标记 |

### 1.3 核心差异点
- 市面所有 todo 工具只记"我欠别人什么";WorkChain 同时记 **"别人欠我什么"**
- 市面所有笔记工具的记录可随意篡改;WorkChain 的记录 **可被第三方独立验证**

---

## 2. 关键设计决策(及其理由)

> 这一节最重要。后续任何人想改架构,先读这里。

### D1. 不接入任何 IM 平台 API
**决策**:不做飞书/钉钉/企微开放平台集成。输入只有三种:截图粘贴、文字粘贴、文件上传(txt/word/pdf)。

**理由**:
- 穷举集成是无底洞(QQ、微信、企微、Teams、邮件…)
- 敏感权限需企业管理员审批,不可控
- **截图是唯一的通用协议**,任何平台都能截图

**代价**:每次记录有约 8-10 秒的应用切换摩擦。已知,接受。这是留存率的天花板,后续优化方向为浏览器插件 / 全局热键。

### D2. AI 产物绝对不进哈希链
**决策**:`slot_*`、`slots_filled`、`plain_summary`、`caveats`、`thread_id`、`kind` 全部排除在摘要计算之外。

**理由**:这些字段会因模型升级或用户修正而变化。一旦进链,任何一次重新解析都会让全链断裂。
**链保护的是"某时刻收到过某份未经修改的原始内容"这一事实,不保护解读。**

**验证方式**:测试用例 5 —— 修改全部槽位后全链验证仍须通过。

### D3. 用"槽位填充"而非"二分类"判断是否为待办
**决策**:不训练/不 prompt 一个"这是不是 todo"的分类器。改为抽取五个槽位,按填充数判定。

五槽位:
1. **requester** 委托方
2. **owner** 受托方
3. **deliverable** 交付物
4. **due** 时限
5. **direction** 方向性(i_owe / owed_to_me / none)

**判定规则**:`slots_filled >= 3 且 direction != none` → 进待办;否则 `kind = reference`(参考信息,如八卦、通知)。

**理由**:
- 八卦天然填不满槽位(无受托方、无交付物),不需要判断
- **可解释**:用户看到的是"未识别到交付物和时限",而非"AI 觉得这不是待办"
- 用户可手动补槽,立刻转为待办

### D4. 归并不追求全自动
**决策**:高分自动归并,中间地带出建议由用户一键确认,低分开新线。

**理由**:准确率的锅甩掉一半,且用户每次确认都是标注数据。

### D5. AI 分两段,不用多模态大模型
**决策**:
- 图 → 文:本地 OCR(PaddleOCR,离线免费,中文识别好)
- 文 → 槽位:DeepSeek API(deepseek-chat,纯文本)

**理由**:DeepSeek 主力 API 不支持图像输入。两段式反而更可控,且 OCR 本地化符合"数据不出本机"的隐私卖点。

---

## 3. 数据结构(冻结)

> 字段一旦冻结不得擅自增删。需变更请先更新本文档并记入第 7 节变更日志。

### 3.1 actors(当事人)

| 字段 | 类型 | 说明 |
|---|---|---|
| actor_id | TEXT PK | `act_xxx` |
| canonical_name | TEXT NOT NULL | 归一后主名,如"张伟" |
| aliases | TEXT (JSON数组) | `["张总","老张","@zhangwei"]` |
| org | TEXT | 部门/公司,消歧用 |
| role_hint | TEXT | 上级 / 同级 / 下游 / 客户 |
| is_self | INTEGER NOT NULL | 是否本人,全库最多一条为 1 |
| confidence | REAL | 归一置信度 |
| created_at | INTEGER NOT NULL | Unix 毫秒 |

约束:`is_self = 1` 的行最多一条(partial unique index)。

> `is_self` 是关键字段,`slot_direction` 由它推导,无需额外存储逻辑。

### 3.2 threads(事项线)

| 字段 | 类型 | 说明 |
|---|---|---|
| thread_id | TEXT PK | `thr_xxx` |
| title | TEXT | AI 生成,用户可改 |
| status | TEXT NOT NULL | open / delivered / closed / disputed / abandoned |
| owner_actor_id | TEXT | 当前受托方 |
| requester_actor_id | TEXT | 委托方 |
| current_deliverable | TEXT | 最新版交付物描述 |
| current_due | INTEGER | 最新版时限 |
| version | INTEGER NOT NULL | 变更次数,初始 1(即"第几集") |
| risk_flags | TEXT (JSON数组) | 见 3.5 |
| first_seen_at | INTEGER NOT NULL | |
| last_activity_at | INTEGER NOT NULL | |

### 3.3 evidence(证据)

| 字段 | 类型 | 说明 |
|---|---|---|
| evidence_id | TEXT PK | `ev_xxx` |
| seq | INTEGER NOT NULL UNIQUE | 全局严格连续自增,哈希链用 |
| thread_id | TEXT | 未归并时为 NULL |
| kind | TEXT NOT NULL | request / confirm / change / deliver / dispute / reference |
| **原始载荷** | | |
| media_type | TEXT NOT NULL | image / text / file |
| blob_path | TEXT | 内容寻址路径 |
| raw_text | TEXT | OCR 结果或粘贴原文 |
| source_hint | TEXT | 如"飞书群-项目A" |
| **五槽位** | | 全部可空,空即未识别 |
| slot_requester | TEXT | actor_id |
| slot_owner | TEXT | actor_id |
| slot_deliverable | TEXT | |
| slot_due | INTEGER | 解析后时间戳 |
| slot_due_raw | TEXT | "下周五"原文,保留歧义 |
| slot_direction | TEXT | i_owe / owed_to_me / none |
| slots_filled | INTEGER NOT NULL | 0-4 |
| **解读层** | | |
| plain_summary | TEXT | 一句话:这段在要求你做什么 |
| caveats | TEXT (JSON数组) | 如 `["未指定具体日期,默认按周五"]` |
| **时间** | | |
| occurred_at | INTEGER | 事件发生时间(截图里显示的时间) |
| captured_at | INTEGER NOT NULL | 入库时间 |
| **完整性** | | |
| content_hash | TEXT NOT NULL | |
| prev_hash | TEXT NOT NULL | |
| chain_hash | TEXT NOT NULL | |

### 3.4 checkpoints(校验点)

| 字段 | 类型 | 说明 |
|---|---|---|
| checkpoint_id | TEXT PK | |
| at_seq | INTEGER NOT NULL | |
| chain_hash | TEXT NOT NULL | |
| created_at | INTEGER NOT NULL | |
| tsa_token | BLOB | 可信时间戳,可后补 |

打点时机:每 100 条 + 每次导出时。

### 3.5 risk_flags 取值

| 值 | 触发条件 | 演示价值 |
|---|---|---|
| `due_missing` | 有交付物但无时限 | 中 |
| `changed_Nx` | deliverable 变更 N 次 | 高 |
| `due_advanced` | due 被提前 | 高 |
| `overdue` | 超期未交付 | 中 |
| `unconfirmed` | 变更后 7 天内无 confirm | **最高(演示王牌)** |

---

## 4. 哈希链算法(核心)

### 4.1 三层计算

```
content_hash  = SHA256(原始载荷字节).hexdigest()
                image/file → 文件原始字节
                text       → raw_text 的 UTF-8 编码

record_digest = SHA256(canonical_json(D)).hexdigest()
                D 只含 7 个字段:
                evidence_id, seq, content_hash,
                occurred_at, captured_at, media_type, source_hint

chain_hash    = SHA256((prev_hash + record_digest).encode('ascii')).hexdigest()
                prev_hash = seq-1 那条的 chain_hash
                seq = 1 时,prev_hash = "0" * 64
```

### 4.2 canonical_json 规范(必须精确)

- 键按 Unicode 码点升序
- 分隔符无空格:`(',', ':')`
- `ensure_ascii=False`,最终以 UTF-8 编码为 bytes
- 所有时间戳为 int(Unix 毫秒),禁止浮点
- 7 个字段固定输出,缺失值输出 JSON `null`,不得省略键

> 此规范不写死,序列化会前后不一致,验证必然失败。

### 4.3 blob 存储

内容寻址:`blobs/<content_hash[:2]>/<content_hash>.bin`
写入前若已存在同 hash 文件则复用,不重复写。

### 4.4 导出举证包

```
export/
  manifest.json   # {version, generated_at, records:[...], checkpoints:[...]}
  blobs/
  verify.py       # 独立验证器,自包含
```

- manifest 每条 record 含:7 个摘要字段 + record_digest + prev_hash + chain_hash + blob 相对路径
- **不含任何 slot_* / plain_summary / caveats**(举证包不携带 AI 解读)

### 4.5 verify.py 要求

- **不得 import 项目任何模块**,必须单文件可分发
- 用法:`python verify.py --dir ./export`
- 逐条重算 content_hash / record_digest / chain_hash 并比对
- 检查 seq 严格连续无缺口
- 校验 checkpoints
- 通过 → 打印 OK 与总条数,exit 0
- 失败 → 打印首个失败的 seq 与原因,exit 1

> verify.py 是给评委当场跑的东西,优先级最高。

---

## 5. 技术栈

| 层 | 选型 | 备注 |
|---|---|---|
| 语言 | Python 3.11+ | |
| 存储 | SQLite | 无 ORM,直接 sqlite3 |
| 哈希 | hashlib 标准库 | |
| OCR | PaddleOCR | 本地离线,中文优先 |
| LLM | DeepSeek `deepseek-chat` | 纯文本,不支持图像 |
| 后端 | FastAPI | 第二阶段引入 |
| 前端 | 待定 | 第三阶段 |
| 测试 | pytest | |

密钥管理:`.env`(必须写入 `.gitignore`),变量名 `DEEPSEEK_API_KEY`。

---

## 6. 阶段规划

### 阶段一:数据层 + 哈希链(当前)
产出:`evidence_core/` + `verify.py` + 完整测试
**禁止**:任何 UI、任何 LLM 调用、任何 Web 框架、任何 OCR

拆分为 6 个 IDE 指令:
1. 工程初始化 + `canonical.py`
2. `db.py` schema
3. `chain.py`
4. `store.py`
5. `export.py` + `verify.py`
6. 全量测试

**必须通过的测试**:
1. 连续写入 100 条,verify_chain 返回 True
2. 篡改任意 blob 字节 → 验证失败且定位到正确 seq
3. 直接改库里某条 occurred_at → 验证失败
4. 删除中间一条 → 检出 seq 缺口
5. **改全部槽位与 plain_summary → 全链仍通过**(证明 D2 正确实现)
6. canonical_json 确定性:含中文/null/键序打乱的等价输入,输出 bytes 完全一致
7. 导出后在临时目录跑 verify.py,exit code 为 0

### 阶段二:AI 解析
OCR 接入 → DeepSeek 槽位抽取 → plain_summary 生成 → 槽位校正交互

### 阶段三:归并与检测
实体对齐(人名消解)→ 事项线归并三路打分 → 变更检测与 risk_flags

### 阶段四:界面与演示
粘贴入口 → 事项线时间轴("职场连续剧"视觉)→ 举证包导出 → 演示数据准备

---

## 7. 变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-08-07 | v1.0 初始版本 | — |

---

## 8. 名词表

| 术语 | 含义 |
|---|---|
| **证据 Evidence** | 一条不可变的原始记录(截图/文字/文件)及其解读 |
| **事项线 Thread** | 围绕同一件事的多条证据的有序集合,即"一部连续剧" |
| **当事人 Actor** | 归一后的人物实体,含别名集合 |
| **槽位 Slot** | 从证据中抽取的五个结构化字段 |
| **哈希链 Chain** | 每条记录的摘要包含前一条摘要,形成不可回溯篡改的序列 |
| **校验点 Checkpoint** | 定期对当前链头打的可信时间戳锚点 |
| **举证包 Export** | 可交给第三方独立验证的导出目录 |
