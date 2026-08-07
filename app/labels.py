STATUS = {
    "open": "进行中",
    "delivered": "已交付",
    "closed": "已关闭",
    "disputed": "有争议",
    "abandoned": "已放弃",
}

KIND = {
    "request": "提出需求",
    "confirm": "确认",
    "change": "变更",
    "deliver": "交付",
    "dispute": "争议",
    "reference": "参考信息",
}

RISK = {
    "due_missing": "未约定时限",
    "due_advanced": "时限被提前",
    "overdue": "已超期",
    "unconfirmed": "变更未确认",
    "changed_3x": "已变更3次",
}

SOURCE_PRESETS = [
    "飞书",
    "Lark",
    "企业微信",
    "微信",
    "钉钉",
    "QQ",
    "Slack",
    "Teams",
    "邮件",
    "Jira",
    "Confluence",
    "腾讯文档",
    "其他",
]

SOURCE_BADGE_CLASSES = {
    "飞书": "bg-sky-400/15 text-sky-200 border-sky-400/30",
    "Lark": "bg-sky-400/15 text-sky-200 border-sky-400/30",
    "企业微信": "bg-emerald-400/15 text-emerald-200 border-emerald-400/30",
    "微信": "bg-green-400/15 text-green-200 border-green-400/30",
    "钉钉": "bg-blue-400/15 text-blue-200 border-blue-400/30",
    "QQ": "bg-indigo-400/15 text-indigo-200 border-indigo-400/30",
    "Slack": "bg-fuchsia-400/15 text-fuchsia-200 border-fuchsia-400/30",
    "Teams": "bg-violet-400/15 text-violet-200 border-violet-400/30",
    "邮件": "bg-amber-400/15 text-amber-200 border-amber-400/30",
    "Jira": "bg-cyan-400/15 text-cyan-200 border-cyan-400/30",
    "Confluence": "bg-teal-400/15 text-teal-200 border-teal-400/30",
    "腾讯文档": "bg-lime-400/15 text-lime-200 border-lime-400/30",
    "其他": "bg-slate-400/15 text-slate-200 border-slate-400/30",
}


def thread_headline(thread: dict) -> str:
    risk_flags = set(thread.get("risk_flags", []))
    status = thread.get("status")
    if "changed_3x" in risk_flags and "unconfirmed" in risk_flags:
        return "需求改了 3 次,你一次都没等到确认"
    if "changed_3x" in risk_flags:
        return "需求已经改了 3 次"
    if "overdue" in risk_flags:
        return "已经超期,对方还在等"
    if "due_missing" in risk_flags:
        return "没人说清什么时候要"
    if status == "delivered":
        return "已交付,记录完整"
    return "进行中"


def source_label(source_hint: str | None) -> tuple[str, str]:
    if not source_hint:
        return ("其他", "")
    if "-" not in source_hint:
        return (source_hint, "")
    platform, scene = source_hint.split("-", 1)
    return (platform, scene)


def source_badge_class(platform: str) -> str:
    return SOURCE_BADGE_CLASSES.get(platform, SOURCE_BADGE_CLASSES["其他"])
