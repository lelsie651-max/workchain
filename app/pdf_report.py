from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.labels import KIND, STATUS, source_label


CJK_FONT_NAME = "STSong-Light"
PDF_FILENAME_PREFIX = "workchain-记录"
PDF_FOOTER_TEXT = "本文件证明所列记录自采集时起未被修改,不证明记录内容的真实性,亦不代表全部相关记录。"


def _ensure_cjk_font() -> None:
    try:
        pdfmetrics.getFont(CJK_FONT_NAME)
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont(CJK_FONT_NAME))


def _format_datetime(value: int | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if value is None:
        return "未记录"
    return datetime.fromtimestamp(value / 1000).strftime(fmt)


def _decode_json_array(value: str | None) -> list[str]:
    if not value:
        return []
    return json.loads(value)


def _extract_filename(raw_text: str | None) -> str | None:
    if not raw_text or not raw_text.startswith("[文件] "):
        return None
    return raw_text[5:]


def _resolve_rows(
    conn: sqlite3.Connection,
    *,
    thread_id: str | None,
    scope: str | None,
) -> list[sqlite3.Row]:
    if thread_id:
        rows = conn.execute(
            """
            SELECT
                e.*,
                sr.canonical_name AS requester_name,
                so.canonical_name AS owner_name
            FROM evidence AS e
            LEFT JOIN actors AS sr ON sr.actor_id = e.slot_requester
            LEFT JOIN actors AS so ON so.actor_id = e.slot_owner
            WHERE e.thread_id = ?
            ORDER BY e.occurred_at ASC, e.seq ASC
            """,
            (thread_id,),
        ).fetchall()
        return list(rows)

    where_clause = "WHERE e.evidence_id NOT LIKE 'ev_demo_%'" if scope == "mine" else ""

    rows = conn.execute(
        f"""
        SELECT
            e.*,
            sr.canonical_name AS requester_name,
            so.canonical_name AS owner_name
        FROM evidence AS e
        LEFT JOIN actors AS sr ON sr.actor_id = e.slot_requester
        LEFT JOIN actors AS so ON so.actor_id = e.slot_owner
        {where_clause}
        ORDER BY e.occurred_at ASC, e.seq ASC
        """
    ).fetchall()
    return list(rows)


def _load_thread_summary(conn: sqlite3.Connection, thread_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT thread_id, title, status, version
        FROM threads
        WHERE thread_id = ?
        """,
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def _build_styles() -> dict[str, ParagraphStyle]:
    _ensure_cjk_font()
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "WorkChainBody",
        parent=sample["BodyText"],
        fontName=CJK_FONT_NAME,
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        wordWrap="CJK",
    )
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=body,
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=10,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=body,
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8,
        ),
        "section_title": ParagraphStyle(
            "SectionTitle",
            parent=body,
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
        ),
        "body": body,
        "muted": ParagraphStyle(
            "Muted",
            parent=body,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#475569"),
        ),
        "small": ParagraphStyle(
            "Small",
            parent=body,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
        ),
        "record_title": ParagraphStyle(
            "RecordTitle",
            parent=body,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceAfter=4,
        ),
        "change": ParagraphStyle(
            "Change",
            parent=body,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#b91c1c"),
        ),
    }


class _NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(page_count)
            super().showPage()
        super().save()

    def _draw_footer(self, page_count: int) -> None:
        self.saveState()
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.line(18 * mm, 12 * mm, A4[0] - 18 * mm, 12 * mm)
        self.setFillColor(colors.HexColor("#475569"))
        self.setFont(CJK_FONT_NAME, 8.5)
        footer = f"第 {self._pageNumber} / {page_count} 页 · {PDF_FOOTER_TEXT}"
        self.drawCentredString(A4[0] / 2, 7.5 * mm, footer)
        self.restoreState()


def _append_paragraph(story: list[Any], text: str, style: ParagraphStyle) -> None:
    story.append(Paragraph(text.replace("\n", "<br/>"), style))


def _build_overview_lines(rows: list[sqlite3.Row], thread: dict[str, Any] | None) -> list[str]:
    occurred_values = [row["occurred_at"] for row in rows if row["occurred_at"] is not None]
    participants = sorted(
        {
            name
            for row in rows
            for name in (row["requester_name"], row["owner_name"])
            if name
        }
    )
    change_count = sum(1 for row in rows if row["kind"] == "change")
    lines = [
        f"时间范围：{_format_datetime(min(occurred_values)) if occurred_values else '未记录'} 至 {_format_datetime(max(occurred_values)) if occurred_values else '未记录'}",
        f"记录条数：{len(rows)}",
        f"变更次数：{change_count}",
    ]
    if thread is not None:
        lines = [
            f"事项名称：{thread['title']}",
            f"当前状态：{STATUS.get(thread['status'], thread['status'])}",
            f"版本次数：第 {thread['version']} 版",
            f"参与人：{'、'.join(participants) if participants else '未识别'}",
            *lines,
        ]
    return lines


def _fit_image(blob_path: Path, max_width: float, max_height: float) -> Image:
    reader = ImageReader(str(blob_path))
    width, height = reader.getSize()
    scale = min(max_width / width, max_height / height, 1)
    image = Image(str(blob_path))
    image.drawWidth = width * scale
    image.drawHeight = height * scale
    return image


def build_evidence_pdf(
    conn,
    *,
    blobs_root,
    thread_id=None,
    scope=None,
    out_path,
) -> Path:
    blobs_root = Path(blobs_root)
    out_path = Path(out_path)
    rows = _resolve_rows(conn, thread_id=thread_id, scope=scope)
    if not rows:
        raise ValueError("no evidence selected for export")

    thread = _load_thread_summary(conn, thread_id) if thread_id else None
    if thread_id and thread is None:
        raise ValueError("thread not found")

    styles = _build_styles()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="记录导出",
        author="WorkChain",
    )
    story: list[Any] = []

    occurred_values = [row["occurred_at"] for row in rows if row["occurred_at"] is not None]
    start_text = _format_datetime(min(occurred_values)) if occurred_values else "未记录"
    end_text = _format_datetime(max(occurred_values)) if occurred_values else "未记录"

    story.append(Spacer(1, 35 * mm))
    story.append(Paragraph("记录导出", styles["cover_title"]))
    story.append(Paragraph("该事项的完整经过", styles["cover_subtitle"]))
    _append_paragraph(story, f"导出时间：{_format_datetime(int(datetime.now().timestamp() * 1000))}", styles["body"])
    _append_paragraph(story, f"记录条数：{len(rows)}", styles["body"])
    _append_paragraph(story, f"时间范围：{start_text} 至 {end_text}", styles["body"])
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("本文件由 WorkChain 生成", styles["muted"]))
    story.append(PageBreak())

    story.append(Paragraph("概览", styles["section_title"]))
    for line in _build_overview_lines(rows, thread):
        _append_paragraph(story, line, styles["body"])
        story.append(Spacer(1, 2 * mm))
    story.append(PageBreak())

    story.append(Paragraph("正文", styles["section_title"]))
    for index, row in enumerate(rows, start=1):
        platform, scene = source_label(row["source_hint"])
        story.append(
            Paragraph(
                f"{index}. {_format_datetime(row['occurred_at'])} · {platform}{' · ' + scene if scene else ''} · {KIND.get(row['kind'], row['kind'])}",
                styles["record_title"],
            )
        )
        if row["kind"] == "change":
            story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#dc2626"), spaceAfter=4))
            story.append(Paragraph("此处发生变更", styles["change"]))
            story.append(Spacer(1, 2 * mm))

        _append_paragraph(story, row["raw_text"] or "", styles["body"])
        story.append(Spacer(1, 2.5 * mm))

        if row["plain_summary"]:
            _append_paragraph(story, f"一句话说明：{row['plain_summary']}", styles["small"])
        if row["slot_deliverable"]:
            _append_paragraph(story, f"交付物：{row['slot_deliverable']}", styles["small"])
        if row["slot_due_raw"] or row["slot_due"]:
            due_text = row["slot_due_raw"] or _format_datetime(row["slot_due"], "%Y-%m-%d")
            _append_paragraph(story, f"时限：{due_text}", styles["small"])
        caveats = _decode_json_array(row["caveats"])
        if caveats:
            _append_paragraph(story, f"提醒事项：{'；'.join(caveats)}", styles["small"])

        if row["media_type"] == "image" and row["blob_path"]:
            blob_path = blobs_root / row["blob_path"]
            if blob_path.exists():
                story.append(Spacer(1, 2 * mm))
                story.append(_fit_image(blob_path, doc.width, 90 * mm))
                filename = _extract_filename(row["raw_text"]) or "未命名图片"
                _append_paragraph(
                    story,
                    f"图片文件：{filename} · 内容摘要前 12 位：{(row['content_hash'] or '')[:12]}",
                    styles["muted"],
                )
        story.append(Spacer(1, 6 * mm))

    story.append(PageBreak())
    story.append(Paragraph("完整性信息", styles["section_title"]))
    table_data = [["序号", "发生时间", "内容摘要前 16 位"]]
    for index, row in enumerate(rows, start=1):
        table_data.append([str(index), _format_datetime(row["occurred_at"]), (row["content_hash"] or "")[:16]])
    table = Table(table_data, colWidths=[18 * mm, 45 * mm, 95 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), CJK_FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 6 * mm))
    _append_paragraph(
        story,
        "完整的可验证举证包请使用「导出完整举证包」功能。",
        styles["body"],
    )

    doc.build(story, canvasmaker=_NumberedCanvas)
    return out_path
