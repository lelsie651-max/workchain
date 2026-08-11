# WorkChain 项目说明书

> 版本 v1.4 · 最后更新 2026-08-10
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

### D4. 归入事项不追求全自动
**决策**:
- 高置信度且语义单一时,允许自动归入已有事项
- 存在分叉、多事件候选或冲突语义时,只给建议并要求用户确认
- 低置信度或上下文不足时,进入"待确认归属",不得自动瞎建事件
- **是否 auto / confirm / needs_context 由确定性代码策略决定,不由模型决定**

**理由**:
- 事实层一旦错误归入事项,后续所有事件时间线都会被污染
- 把不确定性显式暴露出来,比假装全自动更可靠
- 用户确认本身就是高价值标注数据

### D5. AI 分阶段解析,复用同一条证据链
**决策**:
- 图 → 文:阿里云百炼 `vanchin/deepseek-ocr`(OpenAI 兼容接口)
- 生产图片提取:默认使用 DashScope OCR,可通过 `WORKCHAIN_IMAGE_EXTRACTION_PROVIDER=ark_vision` 切到 Ark Vision 优先,并在 Ark 失败时按配置与 OCR 配额决定是否回退到 DashScope OCR
- 文档 → 文:`pypdf` / `python-docx` / 文本解码
- `Evidence -> Extraction(transcript + visual observations) -> Fact -> Event`
- Extraction(`transcript + visual observations`) → Fact / Interpretation:DeepSeek OpenAI-compatible 接口,模型由 `DEEPSEEK_MODEL` 配置(默认 `deepseek-v4-flash`)
- Production Semantic Parsing 固定使用 DeepSeek V4 Flash 的 non-thinking 模式(`thinking: {"type":"disabled"}`),并使用独立 `DEEPSEEK_TEXT_TIMEOUT_SECONDS`(默认 60 秒,仅作用于文本语义链路)
- Fact → Event 建议:独立 Event Matcher,同样使用 `DEEPSEEK_MODEL`,只输出分组与归属建议
- Production 路由现为 `Semantic Parser succeeded -> Event Matcher -> deterministic routing`;仅当本次 Semantic Run 成功且生成了新 Fact 时才运行 Event Matcher
- Event 路由模式(`auto / confirm / needs_context`)由本地确定性策略计算,不交给模型
- 只有 `routing_mode=auto` 才会自动写入 `facts.event_id / event_assignment`; `confirm / needs_context` 会保存建议结果并进入用户确认归属流程

**理由**:
- OCR 模型只负责把图片转成文字,不负责理解业务语义
- 生产图片链统一通过受控 extraction provider 路由;默认值仍是 OCR,避免部署瞬间行为突变
- Ark Vision 可同时产出 `transcript + observations`;其中 transcript 进入 production semantic route,observations 只进入 Extraction 层与 Semantic Parser 输入,不得混入 `raw_text`
- `source` / `source_hint` 属于用户声明的 metadata(`declared_platform`),是 Extraction 输入上下文而非机器真值;它只帮助 Vision 阅读 UI,不是 Evidence 原文的一部分,也不得被用来改写原图文字
- `evidence.source_hint` 是原始提交时写入哈希链的 immutable provenance,后续不得直接改写;用户若核实或修正来源,必须写入独立 `source_reviews` 履历,并通过 `effective source` 解析当前生效来源
- 当 Ark Vision 处理聊天/IM 截图时,优先返回 platform-aware structured conversation extraction,并将 `declared_platform` 与 `observed_platform` 分离:平台必须先由截图 UI 独立观察得到;若 `observed_platform` 与用户声明冲突,最终 transcript / observation / warning 必须保留 provenance,不得静默覆盖任一方
- 当 `observed_platform=unknown` 时,不得仅因用户声明了来源就伪造成机器已观察到该平台;可在 transcript / observation 中保留 `declared_platform` 与 `source_consistency=unknown`
- 当 Ark Vision 处理聊天/IM 截图时,structured contract 必须支持 `system_events`;至少覆盖 `message_recalled / system_notice / unknown`,并保留 `type / visible_text / actor_display_name`;system event 里的名字不得单独推断出新的 participant
- 聊天 transcript 中的 `speaker_ref` 必须是 stable neutral identity:direct chat 在 side 明确时使用 `left_account / right_account`,side 无法安全判断时只能保留 `unknown_account` / unknown side 或显式 warning;group chat 使用 `right_account / participant_n`;昵称只放 `display_name`,不得让 `left_戴雯 / left_饭之 / right_用户` 这类 ref 直接进入最终 transcript
- `chat_header` 只表示顶部直接可见 UI 文本,不等于 participant identity;只有明确的平台特定 UI 规则才允许把它映射到 participant display_name,且这种规则不得跨平台套用(例如微信单聊可映射 left_account.display_name,其它平台没有规则时不得套用)
- group_chat / direct_chat 的接受与纠偏必须基于结构证据,而不是标题语义:可用强证据包括 `visible_sender_label`、`avatar_ref` 或等价布局身份;若模型给出 `group_chat` 但看不到两个以上可区分的非右侧发送者,代码层必须降级为 `direct_chat` 或 `unknown`,不得凭 `chat_header` 中的“群/组/group”、成员数、system event 里的名字或正文提到的人名来伪造群聊参与者
- quote / reply / reaction 属于直接可观察 conversation structure,必须绑定在对应 message 上保留;actor 不可见时只能显式记为 `unknown`,不得推断含义
- emoji 属于 Evidence transcript 边界:能可靠辨认具体 Unicode emoji 时必须原样保留;无法可靠区分时只能写 `[emoji_unknown]` 并附 warning,不得用语义相近 emoji 替换,也不得翻译成解释性文字;sticker 不得伪装成 emoji 或 reaction
- 当 Ark Vision 处理聊天/IM 截图时,最终 `transcript` 必须保留消息视觉顺序、稳定 neutral speaker_ref 与左右/昵称布局关系,不得把聊天压平成无说话人的普通 OCR 段落,也不得在 Vision 层自行改写"打错了/改成/更正"这类原句
- 聊天类图片 Extraction 与 Semantic Projection 必须严格分层:`structured_payload / transcript / observations / warnings` 仍属于 Extraction provenance;Projection 是进入 Semantic Parser 之前由代码确定性生成的只读派生物,不得回写 Evidence 哈希链,也不得让模型代替代码生成
- Schema V11 必须为图片 Extraction 持久化 normalized `structured_payload`;text / doc Extraction 可继续为 `NULL`;后续语义链路读取该 payload 时不得通过反向解析 transcript 标签来重建聊天结构
- Semantic Projection 只允许包含:稳定 `speaker_ref` + 原文 message、可见 `date/time marker`、`quote / reply`、`reaction`、可可靠识别的 inline Unicode emoji;必须排除 `platform / scene`、`chat_header`、participant 声明、`system_event`(含撤回)、菜单/人数/UI noise 与无文字 sticker
- `system_events` 可以保留在 Extraction 中作为机器观测 provenance,但绝不能进入 Semantic Projection,也不能直接参与 Semantic Parser 的事实判断
- Ark Vision 的 structured contract 优先使用 Responses API 官方 structured output(`text.format -> json_schema`, beta);本地 normalize / validation 继续作为第二层代码保护,用于处理模型输出偏差与 provenance 补强
- 当 Ark 失败且 DashScope OCR 已配置且 OCR 配额允许时,允许自动 fallback 到 OCR;fallback 必须保留真实 provider/model 与 warning provenance
- Ark 实验 Extraction 在 diagnostics-only 场景下使用 disabled thinking,并区分 text probe 与 vision 请求的独立 timeout
- 所有新 Evidence 的 production semantic parse 都从 latest Extraction 驱动:text 证据会先生成 builtin machine extraction,图片/文档复用已有 machine extraction,OCR 人工校正生成新的 user extraction
- 图片链路中的 pre-semantic gate 顺序固定为 `Extraction -> Source Gate -> Semantic Projection -> (多图时) Context Assembly / ambiguity review -> Pre-Semantic Context Review -> Semantic Parser`;其中 `Pre-Semantic Context Review` 合并 required temporal context 与 optional speaker context;Source Gate 优先于其它 gate,只要来源仍待核实,就不得进入 Projection、Context Assembly 或 DeepSeek
- 图片链路中的 Source Gate 固定放在 `Extraction -> Semantic Parser` 之间:当 declared platform 明确、`observed_platform` 明确且与 declared 不同、并且 `platform_confidence >= 0.75` 时,当前 Evidence 进入 `clarification_required`,禁止创建 Semantic Run、禁止调用 DeepSeek / Event Matcher,直到用户确认或修正来源
- Source Gate 只要求用户核实,不会自动覆盖用户来源;一旦用户确认 declared source,该 Extraction 的同一 mismatch 不得再次阻断;若用户修正来源,必须基于 `effective source` 重新跑 Vision,而不是直接拿旧 Extraction 继续 Semantic Parse
- Temporal Gate 使用确定性代码检查 Extraction transcript / observations 中是否存在依赖记录日期才能解释的相对/不完整时间;至少包括 `今天/明天/后天/昨天/前天`、`周X/星期X`、`本周/这周/下周/下下周`、`这个月/本月/下个月/今年/明年`、`这个月16号`、缺少年份的 `8月16日` 等
- 若命中 Temporal Gate 且当前没有可靠 `anchor_date`,当前 Evidence 必须进入 `clarification_required` 且 `clarification_reason=temporal_context`;禁止创建 Semantic Run、禁止调用 DeepSeek、禁止调用 Event Matcher;UI 复用现有 `record_date` 输入并提示“这份记录包含相对时间，需要先确认记录发生日期，再进行 AI 整理。”
- 对于尚未创建过 Semantic Run 的 gated Evidence,用户补充 `record_date` 后必须先重跑 Source Gate / Temporal Gate;全部通过后才首次进入 `llm_running` 并后台启动 `_run_parse_pipeline`;这一路径不得误走“已有结果只修 due_at”的兼容逻辑
- direct-style Projection 若出现 `left_account / right_account` 且对应 display_name 缺失,必须进入一次 Pre-Semantic Context Review;`record_date` 仍是 required context,但左/右发言者称呼只属于 optional semantic context,必须明确提示“不填写也可继续”,且用户填写结果只能以独立 provenance 绑定到当前 Extraction,不得修改 Evidence 或 machine Extraction
- speaker context 本轮只覆盖 `left_account / right_account`;source correction 或其它导致新的 Extraction 产生后,旧 speaker review 不得静默套用到结构不同的新 Extraction
- Multi-image Context Assembly 只负责判断哪些 Evidence 应一起阅读以及组内 `analysis_order`;`Context Group != Event`,不得生成 Fact / Event / 标题 / 责任判断,也不依赖 `anchor_date` 做日期换算
- `submission.position` 永远表示用户上传顺序,不得被 AI 修改;`analysis_order` 是独立保存的 AI 阅读顺序,允许与上传顺序不同
- 同一 Evidence 可以属于多个 Context Group,但每个 Evidence 至少必须进入一个 group;assembler 低置信或存在 ambiguity 时必须先进入用户确认,不得直接开始 Semantic Parse;assembler 失败时只能 fallback 为 singleton groups,并显式保留 warning provenance
- 图片的真正 Semantic Parse 必须按 Context Group 发起 multi-evidence Semantic Run;同一组内全部 Semantic Projection 必须同时发送给 DeepSeek,并保留 `evidence_id` 边界;Fact provenance 允许关联一份或多份 Evidence
- 单图图片可跳过 Context Assembly,但仍必须使用 Semantic Projection 与同一套 Pre-Semantic Context Review 边界
- text / doc 当前流程保持不变:本轮不引入 Context Assembly,也不把其 Extraction 输入重构为 Semantic Projection
- Semantic Parser 的输入来自 text/doc 的 Extraction transcript + visual observations,以及图片链路的 Semantic Projection / grouped projections;这些都属于不可信待分析数据,只能放在 user payload,不得混入 system prompt
- production semantic parse 固定使用 `semantic_llm.extract_semantics(...)`,只传 `glossary / source_hint / anchor_date`;其中 `source_hint` 必须取 `effective source`,不传 `self_names`,也不使用 `counterpart` 参与事实判断
- `anchor_date` 优先级固定为:1) 用户明确填写的 `record_date`;2) `infer_reliable_anchor_date(...)` 从 transcript / observation 中确定性识别出的同日消息时间;3) `None`
- 自动识别 `anchor_date` 只允许命中明确消息时间结构(如 `小李(2026-08-09): ...`、`[2026-08-09 10:30] 小李：...`);若出现多个不同日期、普通正文交付日期,或 observation 里没有明确 `消息日期/日期/timestamp + 完整年月日`(可带时间),一律返回 `None`
- 截图/图片 observation 若直接可见完整年月日或完整日期+时间,允许提取为 `kind="timestamp"`;若只有 `19:21` 这类时分,不得补造年月日,也不得拿上传时间冒充聊天日期
- 无论用户填写还是正文可靠识别,都绝不能自动拿 `captured_at` / 上传时间冒充发生日期;当前 Evidence 会在 `meta` 中额外记录 anchor source(`user` / `content`)
- Transcript 与 Visual Observation 都属于 Evidence 的提取层,仍与后续 Fact / Event 解读层分离
- Observation 可以支持 Fact / Interpretation 生成,但不得越过“画面直接可观察事实”边界;若与 transcript 冲突,应显式保留 uncertainty / ambiguity,不得静默脑补
- Semantic Run 记录一次具体语义解析所使用的 provider / model / parser_version 以及精确 Evidence / Extraction 输入,供 Fact / Interpretation provenance 追溯;run 生命周期为 `running -> succeeded/failed`,新的 run 通过 `supersedes_run_id` 保留历史
- 文档提取与 OCR 产物当前仍会兼容写入 `raw_text`,后续搜索、导出、详情页、语义解析继续复用现有链路
- Facts / Interpretations 采用整批原子落库;provider、解析或持久化失败时不得留下半套新语义结果
- 这样可以把"原始材料"与"AI 解读"严格分离,继续满足 D2

### D6. 访客沙箱优先于账号体系
**决策**:当前阶段不引入登录系统。每个访客通过 `wc_sid` cookie 获得一个独立沙箱(SQLite + blobs 副本),24 小时后自动清理。

**理由**:
- 演示与真实试用可以共存,且互不影响
- 避免在早期阶段把精力耗在账号、权限、找回、风控上
- 让"先试一下"的成本最低,适合演示和评审

### D7. 语义业务层依赖 Provider abstraction
**决策**:业务语义层(`Semantic Parser` / `Event Matcher`)依赖受控的 Provider abstraction,不直接绑定单一模型供应商的 URL、API Key 和 HTTP 请求细节。当前 production text provider 为 DeepSeek V4 Flash(`DEEPSEEK_MODEL`,默认 `deepseek-v4-flash`),未来允许在 Provider 层做 A/B 替换。

**理由**:
- 让 Prompt / normalize / routing policy 和供应商接入细节解耦
- 为未来引入其他文本模型或视觉模型预留边界,同时不把供应商细节扩散到业务模块

---

## 3. 数据结构(冻结)

> 字段一旦冻结不得擅自增删。需变更请先更新本文档并记入第 7 节变更日志。

### 3.0 V2 目标语义模型

V2 的目标底层关系为:

`Submission → Evidence → Fact → Event → Derived State`

其中:
- **Submission**: 一次用户提交动作,可以包含多份 Evidence,并保留选择顺序;当前允许 `1 段文字`、`1 个文档` 或 `N 张图片`,其中多文件批次只允许全部为图片
- **Evidence**: 原始不可变材料,仍是哈希链唯一保护对象
- **Fact**: 从一份或多份 Evidence 中抽出的中立事实表达
- **Event**: 多个 Fact 归并后形成的事件容器
- **Derived State**: 从 Event 聚合出的当前状态、摘要、风险标签、视图投影
- **Interpretation**: 独立解释层,可挂在 Fact 或 Evidence 上,但不进入哈希链,也不得反向改写 Fact

> 当前 `threads / slot_* / slot_direction / plain_summary` 仍然保留并继续服务现有功能,但它们属于 **兼容层**，不再代表目标底层模型。

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
| `schema_version` | `1/2/3/4/5/6` | schema 版本 |
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

> `parse_status` 描述的是处理阶段,不是业务判断结果;若存在成功的 Semantic Run,详情页优先展示该 run 的 Facts / Interpretations。没有 Semantic Run 的旧记录,继续回退到 `kind + slot_* + plain_summary + 变更链路` 兼容层。

### 3.7 V2 语义层新增表

#### submissions

| 字段 | 类型 | 说明 |
|---|---|---|
| submission_id | TEXT PK | 一次提交动作 |
| created_at | INTEGER NOT NULL | 创建时间 |
| source_hint | TEXT | 来源提示 |

语义边界:
- 新产生的单文字、单图片、单文档提交,也必须创建一个只含 `1` 个 Evidence 的 Submission
- 多图提交时,每张图片都必须作为独立 Evidence 写入,不得把多图合并为一个 Evidence / blob / transcript / Semantic Run
- 同一次多图提交里的 `submission_evidence.position` 只表示用户选择顺序,不代表真实发生时间顺序

#### submission_evidence

| 字段 | 类型 | 说明 |
|---|---|---|
| submission_id | TEXT NOT NULL | 所属提交 |
| evidence_id | TEXT NOT NULL UNIQUE | 一份 evidence 只属于一个 submission |
| position | INTEGER NOT NULL | 在本次提交中的顺序 |

约束:
- PK `(submission_id, evidence_id)`
- UNIQUE `(submission_id, position)`

#### events

| 字段 | 类型 | 说明 |
|---|---|---|
| event_id | TEXT PK | 事件 ID |
| title | TEXT NOT NULL | 事件标题 |
| status | TEXT NOT NULL | active / resolved / archived |
| summary | TEXT | 事件摘要 |
| created_at | INTEGER NOT NULL | 创建时间 |
| updated_at | INTEGER NOT NULL | 更新时间 |

#### facts

| 字段 | 类型 | 说明 |
|---|---|---|
| fact_id | TEXT PK | 事实 ID |
| event_id | TEXT NULL | 所属事件 |
| fact_type | TEXT NOT NULL | request / commitment / confirmation / scope_change / responsibility_change / deadline_change / delivery / cancellation / denial / statement / reference |
| content | TEXT NOT NULL | 中立事实内容 |
| occurred_at | INTEGER | 发生时间 |
| due_at | INTEGER | 解析后的时限;没有可靠锚点时允许为空 |
| due_raw | TEXT | 时限原文,保留歧义,如"下下周五" |
| due_anchor_at | INTEGER NULL | 相对日期换算所依据的可靠时间锚点 |
| confidence | REAL NULL | Fact 内容抽取置信度,0..1 |
| event_assignment | TEXT NOT NULL DEFAULT `unassigned` | unassigned / auto / suggested / confirmed |
| event_assignment_confidence | REAL NULL | Event 归属置信度,0..1,与 `confidence` 分离 |
| origin | TEXT NOT NULL DEFAULT `ai` | ai / user,标记该 Fact 当前结果来源 |
| review_status | TEXT NOT NULL DEFAULT `unreviewed` | unreviewed / confirmed / corrected |
| semantic_run_id | TEXT NULL | 生成该 Fact 的 Semantic Run |
| created_at | INTEGER NOT NULL | 创建时间 |
| updated_at | INTEGER NOT NULL | 更新时间 |

约束:
- `event_assignment = 'unassigned'` 时,`event_id` 必须为 `NULL`
- 其余 assignment 时,`event_id` 必须非 `NULL`
- `due_raw` 可独立存在;没有可靠时间锚点时,不得为了凑结构强行生成 `due_at`
- 当 `due_raw` 被换算为具体 `due_at`,且换算依据来自可靠消息时间时,应同时保存 `due_anchor_at`
- `origin` / `review_status` 用于保护用户确认或修正后的结果,避免后续 AI 重跑无条件覆盖

#### fact_evidence

`Fact ↔ Evidence` 的多对多关系表。

#### fact_actors

`Fact ↔ Actor` 的关系表,带 `role` 字段。

> `role` 当前不做枚举 CHECK,避免过早锁死语义角色集合。

#### interpretations

| 字段 | 类型 | 说明 |
|---|---|---|
| interpretation_id | TEXT PK | 解释 ID |
| fact_id | TEXT NULL | 可挂到 Fact |
| evidence_id | TEXT NULL | 也可直接挂到 Evidence |
| kind | TEXT NOT NULL | explanation / term / action_hint / uncertainty |
| content | TEXT NOT NULL | 解释内容 |
| confidence | REAL NULL | 0..1 |
| semantic_run_id | TEXT NULL | 生成该 Interpretation 的 Semantic Run |
| created_at | INTEGER NOT NULL | 创建时间 |

约束:
- `fact_id` / `evidence_id` 至少一个非 `NULL`

语义边界:
- Interpretation 只表达解释、术语、行动提示或不确定性
- Interpretation 不得反向改写 Fact 的原始抽取结果
- Evidence 始终代表原始证据,解释层变化不改变 Evidence 的完整性语义

### 3.8 V6 语义解析履历新增表

#### semantic_runs

| 字段 | 类型 | 说明 |
|---|---|---|
| semantic_run_id | TEXT PK | 一次语义解析运行 |
| provider | TEXT NOT NULL | 文本模型供应方 |
| model | TEXT NOT NULL | 模型名 |
| parser_version | TEXT NOT NULL | 解析器版本,如 `2.2` |
| status | TEXT NOT NULL | running / succeeded / failed |
| anchor_date | TEXT NULL | 本次解析显式使用的日期锚点 |
| created_at | INTEGER NOT NULL | 创建时间 |
| completed_at | INTEGER NULL | 完成时间 |
| failure_type | TEXT NULL | 失败类型 |
| supersedes_run_id | TEXT NULL | 指向被 supersede 的旧 run |

#### semantic_run_inputs

| 字段 | 类型 | 说明 |
|---|---|---|
| semantic_run_id | TEXT NOT NULL | 所属 Semantic Run |
| evidence_id | TEXT NOT NULL | 本次 run 消费的 Evidence |
| extraction_id | TEXT NULL | 若非空,指向本次 run 实际消费的 Extraction 版本 |
| position | INTEGER NOT NULL | 多 Evidence 输入顺序 |

约束:
- PK `(semantic_run_id, evidence_id)`
- UNIQUE `(semantic_run_id, position)`
- 若 `extraction_id` 非空,必须属于同一个 `evidence_id`

语义边界:
- Fact / Interpretation 可以通过 `semantic_run_id` 追溯到具体 Semantic Run
- Semantic Run 必须记录精确输入 Evidence / Extraction,而不是只记录"模型跑过一次"
- Semantic 履历不进入 Evidence 原件哈希链,不改变 `content_hash / chain_hash`

### 3.9 V4 提取层新增表

#### evidence_extractions

| 字段 | 类型 | 说明 |
|---|---|---|
| extraction_id | TEXT PK | 提取版本 ID |
| evidence_id | TEXT NOT NULL | 所属 Evidence |
| origin | TEXT NOT NULL | machine / user |
| provider | TEXT NOT NULL | 提取供应方标识 |
| model | TEXT NULL | 提取模型名 |
| transcript | TEXT NULL | 提取到的文字 |
| observations | TEXT NOT NULL | JSON 数组,默认 `[]` |
| warnings | TEXT NOT NULL | JSON 数组,默认 `[]`,记录 fallback 或提取提示 |
| created_at | INTEGER NOT NULL | 创建时间 |
| supersedes_extraction_id | TEXT NULL | 指向被 supersede 的旧版本 |

语义边界:
- Extraction 是 Evidence 的可追溯提取层,不等于 Fact
- `transcript` 记录提取到的文字
- `observations` 只允许记录**界面可观察事实**,例如"小王账号对该消息显示👍反应"
- Observation 不得推断心理、态度或意图,例如不得写成"小王已认真阅读并同意全部内容"
- 如果画面里能看到 reaction 存在,但看不到反应者身份,只能记录 reaction 存在或身份未知,不得根据后续对话猜测是谁点的
- 旧版本 Extraction 不删除、不覆盖
- 图片类 Evidence 的 `raw_text` 仅保留 transcript 兼容层,不得把 observations 混入原文
- 当前 `raw_text` 仍保留为兼容层展示 / 搜索 / 解析输入,但机器或用户提取历史应落入 `evidence_extractions`
- `evidence_extractions` 不进入 Evidence 哈希链,不改变原件与 `content_hash`
- `semantic_runs / semantic_run_inputs / facts.semantic_run_id / interpretations.semantic_run_id` 同样不进入 Evidence 哈希链

### 3.11 V10 来源核实 provenance

#### source_reviews

| 字段 | 类型 | 说明 |
|---|---|---|
| review_id | TEXT PK | 一次来源核实记录 |
| evidence_id | TEXT NOT NULL | 对应 Evidence |
| extraction_id | TEXT NOT NULL | 被核实的 Extraction 版本 |
| original_source_hint | TEXT NOT NULL | 本次核实时生效的 declared source |
| observed_platform | TEXT NULL | 机器观察到的平台 |
| resolved_source_hint | TEXT NOT NULL | 用户确认/修正后的来源 |
| decision | TEXT NOT NULL | `confirmed_declared` / `corrected` |
| created_at | INTEGER NOT NULL | 创建时间 |

语义边界:
- `source_reviews` 不进入 Evidence 哈希链,不改变原始 `evidence.source_hint / content_hash / chain_hash`
- 无来源核实时,`effective source = evidence.source_hint`
- 有来源核实时,`effective source = 最新 source_review.resolved_source_hint`
- Vision 重跑、Semantic Parser 输入与当前用户可见来源,都必须读取 `effective source`
- 用户确认高于机器判断;机器发现 mismatch 只负责提示核实,不得自动覆盖来源

### 3.10 V8 事项归属确认履历

#### event_match_runs

| 字段 | 类型 | 说明 |
|---|---|---|
| event_match_run_id | TEXT PK | 一次 Event Matcher 运行 |
| semantic_run_id | TEXT NOT NULL | 对应的 Semantic Run |
| provider | TEXT NOT NULL | 供应方 |
| model | TEXT NOT NULL | 模型名 |
| matcher_version | TEXT NOT NULL | Matcher 版本 |
| status | TEXT NOT NULL | running / succeeded / failed |
| routing_mode | TEXT NULL | auto / confirm / needs_context |
| result_json | TEXT NULL | 仅保存 Fact ID 映射后的安全快照 |
| failure_type | TEXT NULL | 失败类型 |
| created_at | INTEGER NOT NULL | 创建时间 |
| completed_at | INTEGER NULL | matcher 完成时间 |
| supersedes_run_id | TEXT NULL | 指向旧 run |
| review_status | TEXT NOT NULL | pending / completed |
| reviewed_at | INTEGER NULL | 用户完成确认归属的时间 |

确认语义边界:
- `auto` 成功后直接写入 Fact 归属,并把 `review_status` 记为 `completed`
- `confirm / needs_context` 成功后只保存建议结果,`review_status=pending`,等待用户确认归属
- 用户提交时客户端只允许提交 `event_match_run_id + group_index + choice(existing/new/unassigned) + event_id/new_title`;不得提交 `fact_ids`
- 服务端必须从已保存的安全 `result_json` 重新取出真实 `fact_ids`,并校验该 run 仍然是当前 Evidence 最新的 succeeded match run
- `existing` 只能指向 active Event;`new_title` 必须 trim 后非空且长度受限;所有 groups 必须一次性给出决定
- 用户确认写入必须整批原子完成:existing/new 写成 `event_assignment=confirmed`,unassigned 保持 `event_id=NULL` 但视为用户已决定
- 同一次提交里若多个 new group 的规范化标题完全相同,只创建一个 Event 并共同归入
- 完成后统一更新相关 `events.updated_at`,并把 run 标记为 `review_status=completed`
- 用户确认后的 `event_assignment=confirmed` 后续不得被 AI 覆盖;该流程不得改写 Fact 内容,也不得影响 Evidence 哈希链

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
| OCR | DashScope `vanchin/deepseek-ocr` | 生产图片提取默认 provider,也作为 Ark Vision 失败时的 fallback |
| 文档提取 | `pypdf` / `python-docx` | PDF / docx / txt |
| LLM | DeepSeek OpenAI-compatible (`DEEPSEEK_MODEL`) | 默认 `deepseek-v4-flash`,用于 Semantic Parser 与 Event Matcher |
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
- Semantic Parser V2.2:Extraction transcript + visual observations → Fact / Interpretation
- Semantic Run provenance:Fact / Interpretation 可追溯到具体 parser run 与输入 Evidence / Extraction
- PDF / docx / txt 文档提取
- text / file / image 新记录都会先落 Extraction provenance,再由 production semantic route 读取 latest Extraction 进入 V2.2
- 图片链路:OCR 或 Ark Vision 提取 `transcript + observations` 后,先经过 Source Gate;只有 gate 通过才进入 production semantic route
- 身份设置(`self_names`)与私人词典(`glossary`)
- 人工槽位修正
- OCR 文字人工校正后会生成新的 user extraction,并触发 superseding semantic run

### 阶段三:归并与检测
**已完成首版**
- Actor 归一与词典辅助别名回写
- 事项线展示
- 风险标签与变更提示
- 搜索覆盖 `raw_text` / `plain_summary` / actor 关联信息

**正在收口**
- Event Matcher:Fact 分组 + 归属建议
- `auto / confirm / needs_context` 由确定性 routing policy 决定,不由模型决定
- V1 production apply:matcher result 以 run 履历保存;`auto` 自动归入事项,`confirm / needs_context` 通过用户确认归属一次性落库

### 阶段四:界面与导出
**已完成首版**
- FastAPI + Jinja 首页 / 详情页 / 帮助页 / 搜索页 / 事项线页
- 首页主体验已切到单列聚焦布局:Header 后直接进入主输入区,输入区下方保留独立的 `STEP 1 / STEP 2 / STEP 3` 轻量流程区,再往下才是 `我的事项`
- 首页提交区现为单输入模式:textarea 与 file upload 必须二选一;前端互斥仅做 UX 引导,服务端仍会拒绝 text + file 同时提交
- 首页图片上传现支持一次多选多张图片:前端保留选择顺序、展示编号/缩略图/文件名/大小,允许逐项移除与继续添加,拖拽同样支持多图;文档仍保持一次一个
- 首页图片粘贴入口现属于整个 evidence upload area:第一次 Ctrl/Cmd+V 贴图后,上传区会保持焦点以便继续粘贴第 2 / 3 张图片;粘贴、本地选择、拖拽三种入口最终都走同一个 `appendFiles / validation` 流程,且不会干扰 `source / source_detail / record_date` 等其它表单输入
- `/api/evidence` 现统一以 Submission 为入口:单文字/单图片/单文档会创建 `1 Submission + 1 Evidence`,多图会创建 `1 Submission + N Evidence`;后台对多图只启动一个顺序 wrapper,按 `submission_evidence.position` 逐张执行现有 `Extraction -> Semantic Parse -> Event Matcher`
- 多图批次禁止共用一个 `record_date`:前端会禁用并清空日期输入,服务端也会拒绝 `multi-image + non-empty record_date`,避免把同一日期广播到全部图片
- 首页补充信息现仅保留 `source / source_detail / record_date`,继续收纳进"补充信息(可选)"折叠区;前端已移除 `counterpart`,但服务端兼容字段仍保留
- 首页示例 chips 已移除,避免次要引导继续抢占首屏与主输入动作
- 首页主输入区下方不再保留额外产品说明或营销文案,只保留三步流程说明;Header 品牌副标题继续作为轻量品牌信息
- `我的词典` 已退出首页侧栏,改为 Header 入口打开的右侧抽屉;继续复用现有 `GET/POST /api/settings` 保存 glossary,关闭后保持当前首页状态
- 旧 demo threads 兼容路由仍保留,但普通用户首页不再公开展示;仅在 diagnostics/debug 场景下保留只读入口,避免与真实 Event 体系混淆
- Header 当前正式导航为 `首页 / 我的事项 / 记录 / 使用说明`,并保留 `搜索 / 我的词典` 工具入口;品牌区恢复为紧凑的 `WorkChain` 文字标题,不再显示 `WC` 方形 logo
- 新增 `/records` 页面作为正式入口:复用现有 Evidence 历史查询能力,按保存时间倒序列出当前 sandbox 的记录,展示来源、媒体类型、解析状态、最新 1~2 条 Fact 摘要、所属事项与详情入口,顶部提供 `导出完整记录(PDF)` 链接
- 首页不再直接展示 `历史事项 / 最近证据 / 参考信息 / 导出入口 / diagnostics`;相关路由和兼容能力保留,其中 `记录` 与导出入口已迁移到独立 `/records` 页面
- 新增只读 Event 详情页(`/event/{event_id}`):按时间展示 Facts,并可回跳到支撑它的 Evidence 详情核对原始证据
- Event 详情页现已升级为 Competition Event Control Center V1:可修改事项名称、纠正 Fact 的 `content / fact_type / due_at`、直接补/改相关 Evidence 的记录日期,并通过受控接口切换 `active <-> resolved`
- Event 详情页的原始 Evidence 改为事项级集中展示:在 header/action 区之后先渲染 `相关原始记录`,按当前 Event 全部 Facts 关联到的 `evidence_id` 去重后展示图片缩略图/文字摘要/文件信息,再进入弱化说明文字与 Facts 时间线;Fact 卡内不再重复渲染"支撑它的原始证据"
- Event Change Detection V1 是独立的 semantic derived layer:只比较同一 Event 内、来自不同 Evidence 的历史 Facts,每次最多取最近 20 条并保持时间顺序,识别 `requirement_change / deadline_change / responsibility_change / contradiction` 四类中立差异
- Event Change Detection 结果单独保存到 `event_change_runs / event_changes`,只保留较早 Fact、较新 Fact、change type、summary 与 confidence,不保存模型原始 response,也不把结果写入 Evidence hash chain
- Event Change Detection 的触发点仅限两处:新 Fact 自动归入既有 Event 后,以及用户确认归属后使某 Event 拥有来自至少 2 个 Evidence 的 Facts;若 Event 只有一个 Evidence,则不调用 detector;detector 失败只记录 failed run,不影响 Semantic Parse、Event assignment 与 Fact 主流程
- Event 详情页在 `相关原始记录` 之后、Facts 之前可追加 `这件事发生过这些变化` 区块,只展示 latest succeeded change run,每条 change 必须能回溯到较早/较新两个 Fact 及其原始 Evidence;没有可靠 changes 时不渲染空卡片
- Event 页 Fact 纠正必须走 `correct_fact_by_user(...)`;只允许修改语义字段,不得改 `fact_id / semantic_run_id / Evidence` 关联,写入后统一标记 `origin=user / review_status=corrected` 并更新 `events.updated_at`
- Evidence 详情页改为单列阅读流,顺序固定为:记录标题/来源/时间与轻量 badge -> 原始材料 -> 事项归属(如有) -> `AI 整理` -> 真 legacy 记录的兼容编辑(如有) -> `记录信息` 折叠区 -> diagnostics(仅 dev-only);不再使用"解析结果 / 保存与校验"并列双栏
- Evidence 详情页已支持 `confirm / needs_context` 的用户确认归属表单,位置固定在原始材料之后、`AI 整理` 之前,并与首页共用同一服务端确认写入逻辑;完成后展示最终事项归属结果
- Evidence 详情页若已存在 succeeded Semantic Run,主阅读区改为按 Fact 展示:每条 Fact 内部使用左事实/右解释的局部双栏,Interpretation 必须按稳定 `fact_id` 关联到对应 Fact;`fact_id = NULL` 的 Interpretation 单独进入`整体提醒`;只有无 Semantic Run 的旧记录继续保留兼容编辑入口
- Evidence 详情页在存在相对日期 Fact 时,会在 `AI 整理` 标题下方展示统一的"记录日期"操作区:缺少 anchor 时允许补充日期,已有 anchor 时展示来源(`你填写的` / `从记录时间中识别`)并允许修改,不把日期操作重复塞进每条 Fact
- Production Evidence 详情页不再向普通用户展示"看看系统读到了什么"、OCR transcript、machine extraction 摘要或 provider/model 等提取层信息;相关 extraction history、OCR/视觉能力与受控接口继续保留,但不作为正式 UI 入口
- 首页提交成功后,前端不再 `window.location.reload()` 回首页,而是根据 `/api/evidence` 返回的 `evidence_id` 立即跳转 `/evidence/{id}`;这样即使尚未形成 Event,用户也能立即看到"刚刚这条记录在哪里、当前处理到哪一步"
- Evidence 详情页对 `ocr_running / llm_running` 增加最小状态轮询:复用 `GET /api/evidence/{id}/status` 周期性刷新进度,在 `done / failed / unsupported` 稳定状态后停止轮询并重新渲染页面;不新增业务状态,不额外调用 AI
- `/api/evidence/{id}/record-date` 只做确定性补算:保存 `meta` 中的 anchor/source=`user`,对 latest succeeded semantic run 内 `due_raw` 为相对日期的 Facts 重新计算 `due_at / due_anchor_at`,并以原子事务更新 `origin=user / review_status=corrected`;不重跑 DeepSeek,不改 Evidence 原件,不改 Semantic Run 历史
- 首页 `我的事项` 现分为两类:已建立的 `进行中的事项` 与尚未确认归属的 `待确认事项`;前者仍只展示 `status=active` 的事项并安全截取最近 6 个,后者展示 latest succeeded Semantic Run 对应、且 `routing_mode in (confirm, needs_context) + review_status=pending` 的 Evidence 待确认卡,每条 Evidence 只出现一次
- `待确认事项` 卡只展示 `待确认` 标签、AI 建议标题(多组时显示 `可能涉及 N 个事项`)、最近 1 条 Fact 摘要和 `去确认` 入口,不会为了首页展示而提前创建假 Event
- 用户可见文案不得直接暴露 `anchor_date / due_date / fact_index / event_assignment` 等内部字段名;相对日期缺少可靠锚点时,统一改写为用户能理解的提示,并明确只有补充"记录发生日期"后才能换算
- 当 Evidence 已有可靠 anchor 后,当前视图要过滤掉"缺少记录发生日期 / 无法换算今天或周五"这类 stale date uncertainty;历史 Interpretation 保留,但可补充一条轻量提示: `已按 YYYY-MM-DD 换算相对日期。`
- Help 文案已同步当前真实能力:匿名沙箱、24 小时清理、AI 结果可继续纠正、原件独立保存与完整性校验;用户可见文案统一改成人话,不直接暴露 `AUTO / CONFIRM / NEEDS_CONTEXT` 等实现术语
- 访客沙箱(`wc_sid`)与 24 小时过期清理
- 图片/文档上传、预览、Lightbox
- PDF 导出
- 完整举证包 zip 导出(全链)
- 普通用户页面不直接展示 provider/model、parser version、semantic run id、HTTP status/latency/timeout、schema/provider/internal routing enum 等开发诊断信息;Evidence 普通区只展示用户可理解的整理结果、原始材料、事项归属、OCR 修正入口与`记录信息`折叠,其中`记录信息`只显示保存时间、记录时间(可选)、内容摘要 prefix 与完整性状态
- diagnostics 能力继续保留,但统一受 `WORKCHAIN_DIAGNOSTICS=1` 控制:Production 应设置 `WORKCHAIN_DIAGNOSTICS=0`,Dev 环境可设置为 `1`;关闭时 diagnostics UI、按钮与对应 JS 不渲染,`/api/diag/llm`、`/api/diag/ocr`、Evidence diagnostics / preflight / Ark 实验接口全部返回 404,且不会额外触发真实模型探测
- diagnostics-only Visual A/B Diagnostics:可对同一图片 Evidence 临时运行 Ark Vision,与当前 machine extraction 做只读对照,结果不落库不改状态
- diagnostics-only DeepSeek text preflight:可对同一 Evidence 临时运行轻量 JSON ping,不写 DB,用于区分 Key / 余额 / 模型 / API 问题与真实 Semantic Parser 问题
- Evidence diagnostics 可追踪 Semantic Parser provider failure stage,仅暴露安全元数据(如 stage / http status / timeout / thinking mode / safe message),不包含 prompt、transcript 或完整模型输出,且这些信息只出现在 diagnostics 区

### 下一阶段
- TSA/外部时间锚点,进一步补强 checkpoint 之后的链尾截断问题
- 更稳健的事项线自动归并策略
- 更细的权限/账号体系(若未来从访客模式转正)

---

## 7. 变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-08-09 | Competition UI Round 1 验收返修(问题 2～5):首页移除产品说明文案,恢复独立三步流程,导航补 `首页`,并把 pending confirm / needs_context 归属显示为 `待确认事项` | 修复 active Event 尚未形成时首页像“空的”这一体验断层,同时把首页信息架构收回到更明确的输入与确认主路径 |
| 2026-08-09 | Competition UI Round 1.1:移除 `WC` logo、首页输入区前置并缩小、删示例与 counterpart、补 `/records` 正式入口、提交后直达 Evidence 详情并轮询处理状态 | 根据桌面与手机验收反馈修正主次关系,避免"刚提交的记录消失在空首页"这一致命路径问题,同时把记录历史从首页迁到独立入口 |
| 2026-08-09 | Competition Event Control Center V1:Event 页支持改名 / Fact 纠正 / 记录日期补改 / 结档与重开,首页新增历史事项并将 `我刚存的` 改为 `最近证据` | 让 `我的事项` 成为用户纠正 AI 结果、补日期、结档和回看历史的主入口,同时保持旧记录兼容与哈希链不受影响 |
| 2026-08-09 | Competition Release Hardening:普通页清理 provider/model/parser/run id 等 dev 痕迹,diagnostics 关闭即整组 404,Help 与 demo surface 做正式版收口 | 发布版默认只保留用户能理解的界面与安全错误面,同时继续保留同一套代码下的 diagnostics 能力 |
| 2026-08-09 | Competition UI Round 1:重做全局导航与首页信息架构,词典改为 Header 抽屉,首页只保留 Hero / 主输入 / onboarding / active 事项 | 让第一次打开的普通用户先看懂如何放入记录,其它能力退到次级层级,同时不改变现有业务规则 |
| 2026-08-07 | v1.0 初始版本 | — |
| 2026-08-07 | 简化 slot_direction CHECK 写法;新增 §9 | 原写法正确,等价简化为 IN 单条件;NULL 由 SQLite 三值逻辑天然放行 |
| 2026-08-07 | verify_chain 增加 checkpoint 校验;append_evidence 每 100 条自动打点 | 链尾截断此前无法检测 |
| 2026-08-07 | 新增 export.py 与独立 verify.py | 阶段一收尾 |
| 2026-08-07 | Web 应用、访客沙箱、搜索、帮助页与导出入口落地 | 项目已进入可演示可试用状态 |
| 2026-08-07 | 新增 PDF / docx / txt 提取链路 | 文件类证据需进入统一解析与搜索链路 |
| 2026-08-08 | 图片 OCR 改为 DashScope `vanchin/deepseek-ocr`;新增 `/api/diag/ocr` | 当前工程已不再使用本地 PaddleOCR 方案 |
| 2026-08-08 | 新增 `ocr_running` / `llm_running` 状态、OCR 文本人工校正、举证包附带 PDF 与说明文件 | 让处理过程可见、结果可追溯,并与现有导出物保持一致 |
| 2026-08-08 | 核心定位从"谁欠谁"调整为"中立记录谁对谁表达了什么、发生了什么变化" | `"我的待办"`降级为可选身份视图,不再作为底层事实模型的核心差异点 |
| 2026-08-08 | 新增 V2 语义模型骨架: Submission → Evidence → Fact → Event → Derived State | 为后续语义归档与事件层演进预留安全迁移路径,同时保留现有兼容层 |
| 2026-08-08 | V3 为 facts 增加 `due_anchor_at / event_assignment_confidence / origin / review_status` | 加固相对日期换算语义,拆分事件归属置信度,并为用户确认/修正预留保护状态 |
| 2026-08-08 | V4 新增 `evidence_extractions`,将提取层显式化为 `Evidence -> Extraction -> Fact` | 为 OCR / 文档提取 / 人工校正提供可追溯版本历史,同时保持 `raw_text` 兼容层不变 |
| 2026-08-08 | 新增实验 Visual Provider 边界,production 默认仍为 OCR | 为本地评测和未来多模态能力预留接口,但不改变线上默认图片提取链路 |
| 2026-08-08 | diagnostics 入口加开关保护,并新增 Ark Vision 只读实验对照接口 | 避免真实模型调用裸露对外,同时支持同 Evidence 的安全 A/B 视觉提取诊断 |
| 2026-08-08 | Ark diagnostics Extraction 使用 disabled thinking,并拆分 text/vision 独立 timeout | 单独验证视觉 timeout 与推理开关对 Ark 实验提取成功率的影响,不改其它请求变量 |
| 2026-08-08 | 生产图片 Extraction 改为 provider-aware 路由:默认 OCR,可切 Ark Vision,失败后按 OCR 配置与配额 fallback | 保持默认行为稳定,同时让 transcript 与 observations 分离落库并保留真实 provenance |
| 2026-08-08 | Semantic Parser V2.2 支持同时消费 transcript + visual observations | 让语义解析可以基于文字与画面直接可观察事实共同生成 Fact / Interpretation,同时保持 Observation 边界与冲突显式化 |
| 2026-08-08 | V6 新增 Semantic Run / Fact Extraction Provenance | 让 Fact / Interpretation 能追溯到具体 parser run、模型版本与精确 Evidence / Extraction 输入,同时继续保持其在哈希链之外 |
| 2026-08-08 | Production Semantic Pipeline V2 接管旧 `extract_slots`,详情页最小展示最新 Semantic Run 结果 | 新 Evidence 的生产语义入口已统一切到 latest Extraction -> Semantic Parser V2.2,旧 slot 字段降级为兼容展示层 |
| 2026-08-09 | V7 新增 Event Match Run 履历,Production 改为 Semantic succeeded 后运行 Event Matcher | 让 Event 建议/自动归档同样具备 provenance,并明确只有 auto 会实际改写 Fact 归属 |
| 2026-08-09 | V8 新增 Event Assignment Confirmation:pending/completed review 状态、Evidence 详情确认归属 UI、整批原子提交 | 让 confirm / needs_context 不再停留在只读建议,而是能由用户一次性确认归入已有事项、新建事项或暂不归入 |
| 2026-08-09 | Competition UX V1:首页主体验切到真实 Event / Fact,新增只读 Event 详情页,Help 同步真实能力 | 让用户首先看到 `我的事项` 与可回看原始证据的事实链路,同时把 demo threads 降级为示例区并清理过时文案 |
| 2026-08-09 | Competition Core UX Polish:首页输入互斥、首页直确认事项、人话日期提示、事项截止展示 | 抹平首页与详情之间的确认断点,并明确"记录发生日期"只能来自用户显式填写,不能自动借用上传时间 |
| 2026-08-10 | V10 新增 Source Review / effective source / clarification_required Source Gate | 固定把来源核实放在 `Extraction -> Semantic Parser` 之间,保持 `evidence.source_hint` 不可变,并让用户确认优先于机器平台判断 |
| 2026-08-10 | 新增 Pre-Semantic Temporal Gate、system_events、结构证据群聊保护与 emoji exact-or-unknown | 固定图片链路为 `Extraction -> Source Gate -> Temporal Gate -> Semantic Parser`,避免缺少可靠时间锚点时误调 DeepSeek,并禁止 system event / 标题语义伪造 participant 或群聊结构 |
| 2026-08-11 | V11 新增 Image Semantic Preparation:Semantic Projection、Context Assembly、speaker context provenance 与 multi-evidence Semantic Run | 把聊天类图片的 Extraction 与 Semantic 输入彻底分层,让多图先做分组与阅读顺序决定,并将时间设为 required context、speaker real name 设为 optional context,同时保证上传顺序与 AI 分析顺序分离 |
| 2026-08-09 | Competition Date & Input UX Hardening:输入模式真互斥、可靠 anchor 识别、详情补日期闭环、确定性相对日期换算 | 保持"用户填写优先、正文可靠日期次之、上传时间禁用"的锚点边界,并让补日期后 due_at 与首页事项截止立即联动 |

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
