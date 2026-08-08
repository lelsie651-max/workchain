# WorkChain 项目说明书

> 版本 v1.1 · 最后更新 2026-08-08
> 本文档是项目的唯一事实来源。当 IDE 上下文丢失、记忆偏差或需要交接时,以本文档为准。

---

## 0. 一句话定义

**WorkChain 是一个职场证据链工具:你把聊天截图/文字粘进来,它中立记录"谁对谁表达了什么、后来发生了什么变化",生成不可篡改的证据链,并把散落的记录自动串成一条可追溯的"事项线"。**

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
- WorkChain 的核心对象不是"我的待办",而是**可验证的事实记录与变化过程**
- "我的待办"只是基于可选身份(`self_names` / `is_self`)生成的一种视图,不是底层事实模型
- 市面所有笔记工具的记录可随意篡改;WorkChain 的记录 **可被第三方独立验证**

---

## 2. 关键设计决策(及其理由)

> 这一节最重要。后续任何人想改架构,先读这里。

### D1. 不接入任何 IM 平台 API
**决策**:不做飞书/钉钉/企微开放平台集成。输入只接受用户主动提交的原始材料:文字粘贴、截图/图片上传、文件上传(`txt / pdf / docx`)。

**理由**:
- 穷举集成是无底洞(QQ、微信、企微、Teams、邮件…)
- 敏感权限需企业管理员审批,不可控
- **截图是唯一的通用协议**,任何平台都能截图
- 文件上传作为补充入口,复用同一条证据链,不另起第二套模型

**代价**:每次记录有约 8-10 秒的应用切换摩擦。已知,接受。这是留存率的天花板,后续优化方向为浏览器插件 / 全局热键。

### D2. AI 产物绝对不进哈希链
**决策**:`slot_*`、`slots_filled`、`plain_summary`、`caveats`、`thread_id`、`kind` 全部排除在摘要计算之外。

**理由**:这些字段会因模型升级或用户修正而变化。一旦进链,任何一次重新解析都会让全链断裂。
**链保护的是"某时刻收到过某份未经修改的原始内容"这一事实,不保护解读。**

**验证方式**:测试用例 5 —— 修改全部槽位后全链验证仍须通过。

### D3. 用槽位记录事实,而不是把底层记录直接做成"我的待办"
**决策**:不训练/不 prompt 一个"这是不是 todo"的分类器。底层先抽取结构化槽位,用于还原对话中的关系、交付物、时间与变化;任何"我的待办"都只是后续视图层的投影。

五槽位:
1. **requester** 委托方
2. **owner** 受托方
3. **deliverable** 交付物
4. **due** 时限
5. **direction** 面向当前身份视角的可选投影(i_owe / owed_to_me / none)

**当前规则**:
- 底层记录始终优先表达"发生了什么"
- `kind` 用于区分 `request / confirm / change / deliver / dispute / reference`
- `"我的待办"` 如需展示,应基于 `self_names / is_self + slot_* + kind` 在视图层派生,而不是反过来驱动事实层建模

**理由**:
- 先记录事实,可以同时兼容"我视角""对方视角""第三方回放视角"
- **可解释**:用户看到的是"系统识别到谁说了什么、谁确认了什么、哪里发生了变更",而不是"AI 觉得这是不是我的活"
- 身份缺失、身份变化或多人协作场景下,底层模型仍然稳定

### D4. 归并不追求全自动
**决策**:高分自动归并,中间地带出建议由用户一键确认,低分开新线。

**理由**:准确率的锅甩掉一半,且用户每次确认都是标注数据。

### D5. AI 分两段,复用同一条文本解析链
**决策**:
- 图 → 文:阿里云百炼 `vanchin/deepseek-ocr`(OpenAI 兼容接口)
- 文档 → 文:`pypdf` / `python-docx` / 文本解码
- 文 → 槽位:DeepSeek `deepseek-chat`(纯文本)

**理由**:
- OCR 模型只负责把图片转成文字,不负责理解业务语义
- 文档提取与 OCR 产物统一落到 `raw_text`,后续搜索、导出、详情页、LLM 槽位抽取全部复用同一链路
- 这样可以把"原始材料"与"AI 解读"严格分离,继续满足 D2

### D6. 访客沙箱优先于账号体系
**决策**:当前阶段不引入登录系统。每个访客通过 `wc_sid` cookie 获得一个独立沙箱(SQLite + blobs 副本),24 小时后自动清理。

**理由**:
- 演示与真实试用可以共存,且互不影响
- 避免在早期阶段把精力耗在账号、权限、找回、风控上
- 让"先试一下"的成本最低,适合演示和评审

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

> `is_self` 与 `self_names` 主要服务于"我的待办"这类视图层能力,不应反过来定义事实层本身。

### 3.2 threads(事项线)

| 字段 | 类型 | 说明 |
|---|---|---|
| thread_id | TEXT PK | `thr_xxx` |
| title | TEXT | AI 生成,用户可改 |
| status | TEXT NOT NULL | open / delivered / closed / disputed / abandoned |
| owner_actor_id | TEXT | 当前主要受托方 |
| requester_actor_id | TEXT | 当前主要委托方 |
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
| raw_text | TEXT | 原文 / 文档提取文本 / OCR 文本 / 人工校正后的 OCR 文本 |
| source_hint | TEXT | 如"飞书群-项目A" |
| **五槽位** | | 全部可空,空即未识别 |
| slot_requester | TEXT | actor_id |
| slot_owner | TEXT | actor_id |
| slot_deliverable | TEXT | |
| slot_due | INTEGER | 解析后时间戳 |
| slot_due_raw | TEXT | "下周五"原文,保留歧义 |
| slot_direction | TEXT | 面向当前身份视角的可选投影:i_owe / owed_to_me / none |
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

### 3.6 meta(KV 元信息)

`meta` 表除 `schema_version` 外,还承载运行态与配置态信息。当前工程中已实际使用以下键约定:

| key 模式 | value | 用途 |
|---|---|---|
| `schema_version` | `1` | schema 版本 |
| `parse_status:{evidence_id}` | `ocr_running / llm_running / done / failed / unsupported` | 解析状态机 |
| `parse_detail:{evidence_id}` | TEXT | 解析失败或降级说明 |
| `extract_note:{evidence_id}` | TEXT | 文档提取 / OCR 的具体失败原因 |
| `verified:{evidence_id}` | `1` | 用户人工确认标记 |
| `ocr_corrected:{evidence_id}` | `0/1` | OCR 文本是否被人工校正 |
| `parse_count:{YYYY-MM-DD}` | 整数 | 沙箱内每日 LLM 解析计数 |
| `global_parse_count:{YYYY-MM-DD}` | 整数 | 全站每日 LLM 解析计数 |
| `ocr_count:{YYYY-MM-DD}` | 整数 | 沙箱内每日 OCR 计数 |
| `global_ocr_count:{YYYY-MM-DD}` | 整数 | 全站每日 OCR 计数 |
| `settings:self_names` | JSON 数组 | 访客身份称呼 |
| `settings:glossary` | JSON 数组 | 访客私人词典 |

其中 `parse_status` 当前约定:
- `ocr_running`: 正在读取图片中的文字
- `llm_running`: 正在理解这段对话
- `done / failed / unsupported`: 含义保持不变

> `parse_status` 描述的是处理阶段,不是业务判断结果;业务展示应优先回到 `kind + slot_* + plain_summary + 变更链路` 这一层。

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
  manifest.json         # {version, generated_at, records:[...], checkpoints:[...]}
  blobs/
  verify.py             # 独立验证器,自包含
  workchain-记录-*.pdf   # 给人读的 PDF
  怎么验证这份材料.txt    # 大白话说明
```

- manifest 每条 record 含:7 个摘要字段 + record_digest + prev_hash + chain_hash + blob 相对路径
- **manifest 不含任何 slot_* / plain_summary / caveats**(举证包的机器校验部分不携带 AI 解读)
- PDF 允许包含 plain_summary / deliverable / due / caveats,因为 PDF 的定位是"给人读"

### 4.5 verify.py 要求

- **不得 import 项目任何模块**,必须单文件可分发
- 用法:`python verify.py --dir ./export`
- 逐条重算 content_hash / record_digest / chain_hash 并比对
- 检查 seq 严格连续无缺口
- 校验 checkpoints
- 通过 → 打印 OK 与总条数,exit 0
- 失败 → 打印首个失败的 seq 与原因,exit 1

> verify.py 是给评委当场跑的东西,优先级最高。

### 4.6 链尾截断与校验点
verify_chain 逐条遍历现有记录,无法自行发现链尾被截断
(删除末尾若干条后 seq 仍连续、哈希仍自洽)。
checkpoints 表用于封堵此漏洞:每 100 条自动打点,
记录当时的最大 seq 与 chain_hash。
verify_chain 校验 checkpoint.at_seq 是否仍存在、chain_hash 是否一致。

**残余风险**:最近一次 checkpoint 之后写入的记录若被整体删除,
仍无法检测。彻底封堵需可信时间戳(tsa_token),属阶段四范围。

---

## 5. 技术栈

| 层 | 选型 | 备注 |
|---|---|---|
| 语言 | Python 3.11+ | |
| 存储 | SQLite | 无 ORM,直接 sqlite3 |
| 哈希 | hashlib 标准库 | |
| OCR | DashScope `vanchin/deepseek-ocr` | OpenAI 兼容接口;仅做图转文 |
| 文档提取 | `pypdf` / `python-docx` | PDF / docx / txt |
| LLM | DeepSeek `deepseek-chat` | 纯文本槽位抽取 |
| 后端 | FastAPI | 已落地 |
| 前端 | Jinja2 Templates + 原生 JS + Tailwind CDN | 已落地 |
| 测试 | pytest | |
| PDF 导出 | reportlab + pypdf | 生成与中文可读回验证 |

密钥管理:`.env`(必须写入 `.gitignore`),当前变量:
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`

---

## 6. 当前实现进度

### 阶段一:数据层 + 哈希链
**已完成**
- `evidence_core/` 的 canonical / schema / chain / store / export 全链路
- 独立 `verify.py`
- checkpoint 校验

### 阶段二:AI 解析
**已完成首版**
- DeepSeek 文本槽位抽取
- PDF / docx / txt 文档提取
- 图片 OCR → 文本 → 现有槽位抽取链路
- 身份设置(`self_names`)与私人词典(`glossary`)
- 人工槽位修正
- OCR 文字人工校正后重新解析

### 阶段三:归并与检测
**已完成首版**
- Actor 归一与词典辅助别名回写
- 事项线展示
- 风险标签与变更提示
- 搜索覆盖 `raw_text` / `plain_summary` / actor 关联信息

### 阶段四:界面与导出
**已完成首版**
- FastAPI + Jinja 首页 / 详情页 / 帮助页 / 搜索页 / 事项线页
- 访客沙箱(`wc_sid`)与 24 小时过期清理
- 图片/文档上传、预览、Lightbox
- PDF 导出
- 完整举证包 zip 导出(全链)
- `/api/diag/llm` 与 `/api/diag/ocr` 连通性自检

### 下一阶段
- TSA/外部时间锚点,进一步补强 checkpoint 之后的链尾截断问题
- 更稳健的事项线自动归并策略
- 更细的权限/账号体系(若未来从访客模式转正)

---

## 7. 变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-08-07 | v1.0 初始版本 | — |
| 2026-08-07 | 简化 slot_direction CHECK 写法;新增 §9 | 原写法正确,等价简化为 IN 单条件;NULL 由 SQLite 三值逻辑天然放行 |
| 2026-08-07 | verify_chain 增加 checkpoint 校验;append_evidence 每 100 条自动打点 | 链尾截断此前无法检测 |
| 2026-08-07 | 新增 export.py 与独立 verify.py | 阶段一收尾 |
| 2026-08-07 | Web 应用、访客沙箱、搜索、帮助页与导出入口落地 | 项目已进入可演示可试用状态 |
| 2026-08-07 | 新增 PDF / docx / txt 提取链路 | 文件类证据需进入统一解析与搜索链路 |
| 2026-08-08 | 图片 OCR 改为 DashScope `vanchin/deepseek-ocr`;新增 `/api/diag/ocr` | 当前工程已不再使用本地 PaddleOCR 方案 |
| 2026-08-08 | 新增 `ocr_running` / `llm_running` 状态、OCR 文本人工校正、举证包附带 PDF 与说明文件 | 让处理过程可见、结果可追溯,并与现有导出物保持一致 |
| 2026-08-08 | 核心定位从"谁欠谁"调整为"中立记录谁对谁表达了什么、发生了什么变化" | `"我的待办"`降级为可选身份视图,不再作为底层事实模型的核心差异点 |

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

---

## 9. 已知限制(已评估,暂不修复)
- canonical.py:INT_FIELDS 字段若收到 dict/list 容器类型不会报错
  (容器分支提前 return,未走到类型校验)。
  风险低:store.py 以 dataclass 程序化构造记录,不会出现该输入。
- canonical.py:NFC 归一化后的键若发生碰撞会静默覆盖,造成摘要歧义。
  风险低:digest 的 7 个键均为 ASCII 字面量。
- 最近一次 checkpoint 之后的记录若被整体删除,当前无法检测;
  需 TSA 时间戳封堵,见 §4.6。
- checkpoints 表本身若被一并删除,verify_chain 无法检测。
  自持有数据无法自我封堵此风险;缓解方式是举证包一经导出交付,
  对方持有的 manifest 即成为外部锚点。彻底方案为 TSA 时间戳。
- OCR 与 LLM 当前都依赖外部 API,未配置 key 时系统会优雅降级,但不会自动解析图片或文本。
- `raw_text` 承担搜索、展示与后续解析输入;它允许文档提取文本、OCR 文本和人工校正文本覆盖展示层,但这类变化都不进入摘要计算,见 D2。
