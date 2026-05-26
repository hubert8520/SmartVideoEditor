"""Human-readable editor review report helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from smart_video_editor.reporting.quality import normalize_repair_suggestion
from smart_video_editor.reporting.timeline import (
    map_final_range_to_raw_ranges,
    raw_range_tuples,
    timeline_from_edit_decisions,
)
from smart_video_editor.timecode import seconds_to_timestamp, timestamp_to_seconds


SEVERITY_LABELS = {
    "high": "WYSOKI",
    "medium": "SREDNI",
    "low": "NISKI",
}

CATEGORY_LABELS = {
    "cut_word": "uciete slowo",
    "repetition": "powtorzenie",
    "dangling_thought": "urwana mysl",
    "logic_gap": "problem logiczny",
    "off_topic": "off-topic",
    "noise_or_setup": "halas/setup",
    "other": "inne",
}

ACTION_LABELS = {
    "force_keep": "AUTO: przywroc fragment z raw",
    "force_drop": "AUTO: usun fragment z raw",
    "manual_review": "REVIEW: odsluchaj przed decyzja",
    "no_auto_repair": "INFO: bez automatycznej naprawy",
}


def format_range(start: float, end: float) -> str:
    return f"{seconds_to_timestamp(start)} - {seconds_to_timestamp(end)}"


def clip_range(start: float, end: float, margin: float) -> tuple[float, float]:
    return max(0.0, start - margin), max(start, end + margin)


def severity_label(issue: dict[str, Any]) -> str:
    severity = str(issue.get("severity", "")).lower()
    return SEVERITY_LABELS.get(severity, severity)


def category_label(issue: dict[str, Any]) -> str:
    category = str(issue.get("issue_category", ""))
    return CATEGORY_LABELS.get(category, category)


def action_label(action: str) -> str:
    return ACTION_LABELS.get(action, action)


def _window_seconds(window: dict[str, Any], key: str) -> float:
    value = window.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return timestamp_to_seconds(value)
    return 0.0


def _completeness_summary(evidence: dict[str, Any]) -> str:
    pieces: list[str] = []
    for label, source_key in (("wczesniejsza", "earlier"), ("pozniejsza", "later")):
        source = evidence.get(source_key)
        if not isinstance(source, dict):
            continue
        completeness = source.get("completeness")
        if not isinstance(completeness, dict):
            continue
        markers = completeness.get("markers", [])
        if isinstance(markers, list):
            marker_text = ", ".join(str(marker) for marker in markers) or "brak"
        else:
            marker_text = str(markers)
        pieces.append(
            f"{label}: score={completeness.get('score')}, "
            f"complete={completeness.get('is_complete')}, markers={marker_text}"
        )
    return "; ".join(pieces)


def planner_evidence_rows(edit_decisions: dict[str, Any]) -> list[dict[str, str]]:
    """Summarize local detector evidence carried into edit_decisions.json."""
    review = edit_decisions.get("cut_planner_review")
    if not isinstance(review, dict):
        return []

    rows: list[dict[str, str]] = []
    buckets = (
        ("applied_windows", "DROP"),
        ("review_windows", "REVIEW"),
        ("blocked_windows", "BLOCKED"),
    )
    for bucket, action in buckets:
        windows = review.get(bucket, [])
        if not isinstance(windows, list):
            continue
        for window in windows:
            if not isinstance(window, dict):
                continue
            evidence = window.get("evidence")
            if not isinstance(evidence, dict) or not evidence:
                continue
            start = _window_seconds(window, "candidate_start") or _window_seconds(window, "start")
            end = _window_seconds(window, "candidate_end") or _window_seconds(window, "end")
            rows.append(
                {
                    "action": action,
                    "category": str(window.get("category", "")),
                    "range": format_range(start, end),
                    "text": str(window.get("source_text", "")),
                    "reason": str(window.get("reason", "")),
                    "summary": _completeness_summary(evidence),
                }
            )
    return rows


def _issue_time(issue: dict[str, Any], key: str, fallback: str) -> float:
    final_range = issue.get("final_range")
    if isinstance(final_range, dict):
        seconds_key = f"{key}_seconds"
        if seconds_key in final_range:
            return float(final_range[seconds_key])
        if key in final_range:
            return timestamp_to_seconds(str(final_range[key]))
    return timestamp_to_seconds(str(issue.get(key, fallback)))


def canonical_raw_ranges(raw_ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for start, end, timeline_id in raw_range_tuples(raw_ranges):
        ranges.append(
            {
                "timeline_id": timeline_id,
                "raw_start_seconds": start,
                "raw_end_seconds": end,
                "raw_range": format_range(start, end),
            }
        )
    return ranges


def issue_raw_ranges(
    issue: dict[str, Any],
    final_start: float,
    final_end: float,
    edit_decisions: dict[str, Any],
) -> list[dict[str, Any]]:
    embedded = issue.get("raw_ranges")
    if isinstance(embedded, list) and embedded:
        canonical = canonical_raw_ranges([item for item in embedded if isinstance(item, dict)])
        if canonical:
            return canonical

    timeline = timeline_from_edit_decisions(edit_decisions)
    mapped = map_final_range_to_raw_ranges(final_start, final_end, timeline)
    return canonical_raw_ranges(mapped)


def raw_context_text(issue: dict[str, Any]) -> str:
    context = issue.get("raw_context")
    if not isinstance(context, list):
        return ""
    words: list[str] = []
    for item in context:
        if not isinstance(item, dict):
            continue
        text = str(item.get("word", "")).strip()
        if text:
            words.append(text)
    return " ".join(words)


def actionability_label(
    issue: dict[str, Any],
    suggestion: dict[str, Any],
    *,
    mapped_to_raw: bool,
) -> str:
    actionability = issue.get("actionability")
    requires_manual = bool(suggestion.get("requires_manual_review"))
    if isinstance(actionability, dict):
        requires_manual = bool(actionability.get("requires_manual_review", requires_manual))
        mapped_to_raw = bool(actionability.get("mapped_to_raw", mapped_to_raw))

    if requires_manual:
        return "manual_review"
    if not mapped_to_raw and suggestion.get("action") in {"force_keep", "force_drop"}:
        return "blocked_no_raw_mapping"
    return "auto_repair_candidate"


def editor_instruction(issue: dict[str, Any], suggestion: dict[str, Any]) -> str:
    action = str(suggestion.get("action", "manual_review"))
    if action == "force_keep":
        return "Sprawdz, czy raw zawiera brakujace slowo, lacznik albo krotki fragment sensu."
    if action == "force_drop":
        return "Sprawdz, czy wskazany fragment to rzeczywista powtorka, noise albo resztka setupu."
    if action == "manual_review":
        return "Odsluchaj edit i raw przed decyzja; automatyczna naprawa nie jest bezpieczna."
    if action == "no_auto_repair":
        return "Potraktuj jako informacje QA; nie ma prostej automatycznej naprawy keep/drop."
    return str(issue.get("suggested_action", "Sprawdz recznie."))


def build_editor_review_rows(
    quality_report: dict[str, Any],
    edit_decisions: dict[str, Any],
    *,
    clip_margin: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    issues = quality_report.get("issues", [])
    if not isinstance(issues, list):
        return rows

    for index, issue in enumerate(issues, start=1):
        if not isinstance(issue, dict):
            continue
        final_start = _issue_time(issue, "start", "00:00:00:000")
        final_end = _issue_time(issue, "end", seconds_to_timestamp(final_start))
        edited_clip_start, edited_clip_end = clip_range(final_start, final_end, clip_margin)
        raw_ranges = issue_raw_ranges(issue, final_start, final_end, edit_decisions)
        raw_compare_ranges = []
        for raw_range in raw_ranges:
            compare_start, compare_end = clip_range(
                float(raw_range["raw_start_seconds"]),
                float(raw_range["raw_end_seconds"]),
                clip_margin,
            )
            raw_compare_ranges.append(format_range(compare_start, compare_end))

        suggestion = normalize_repair_suggestion(issue)
        action = str(suggestion.get("action", "manual_review"))
        rows.append(
            {
                "index": index,
                "priority": severity_label(issue),
                "category": category_label(issue),
                "action": action,
                "action_label": action_label(action),
                "actionability": actionability_label(issue, suggestion, mapped_to_raw=bool(raw_ranges)),
                "repair_confidence": str(suggestion.get("confidence", "")),
                "repair_rationale": str(suggestion.get("rationale", "")),
                "manual_review_required": bool(suggestion.get("requires_manual_review")),
                "editor_instruction": editor_instruction(issue, suggestion),
                "edited_range": format_range(final_start, final_end),
                "edited_compare_range": format_range(edited_clip_start, edited_clip_end),
                "final_start_seconds": final_start,
                "final_end_seconds": final_end,
                "raw_ranges": raw_ranges,
                "raw_compare_ranges": raw_compare_ranges,
                "raw_context": raw_context_text(issue),
                "description": str(issue.get("description", "")),
                "affected_text": str(issue.get("affected_text", "")),
                "suggested_action": str(issue.get("suggested_action", "")),
                "edited_clip_path": None,
                "raw_clip_paths": [],
            }
        )
    return rows


def review_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "issues": len(rows),
        "manual_review": sum(
            1 for row in rows if row["actionability"] in {"manual_review", "blocked_no_raw_mapping"}
        ),
        "auto_repair_candidates": sum(1 for row in rows if row["actionability"] == "auto_repair_candidate"),
        "mapped_to_raw": sum(1 for row in rows if row["raw_ranges"]),
    }


def write_editor_review_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    quality_report: dict[str, Any],
    edited_video_label: str,
    raw_video_label: str,
    *,
    make_clips: bool,
    planner_evidence: list[dict[str, str]] | None = None,
) -> None:
    summary = review_summary(rows)
    lines = [
        "# Brief montazowy",
        "",
        f"Status QA: **{quality_report.get('status', '')}**",
        "",
        f"Film edytowany: `{edited_video_label}`",
        f"Film raw: `{raw_video_label}`",
        "",
        f"Liczba miejsc do sprawdzenia: **{summary['issues']}**",
        f"Kandydaci do automatycznej naprawy: **{summary['auto_repair_candidates']}**",
        f"Wymaga recznego review: **{summary['manual_review']}**",
        f"Zmapowane na raw: **{summary['mapped_to_raw']}**",
        "",
    ]
    notes = str(quality_report.get("overall_notes", "")).strip()
    if notes:
        lines.extend(["## Ogolna uwaga QA", "", notes, ""])

    planner_evidence = planner_evidence or []
    if planner_evidence:
        lines.extend(
            [
                "## Dowody plannera",
                "",
                "Te wpisy pokazuja, dlaczego lokalne kandydaty powtorek trafily do DROP/REVIEW/BLOCKED.",
                "",
            ]
        )
        for evidence in planner_evidence:
            lines.extend(
                [
                    f"- `{evidence['action']}` `{evidence['category']}` `{evidence['range']}`: "
                    f"{evidence['text']} ({evidence['reason']})",
                    f"  - {evidence['summary']}",
                ]
            )
        lines.append("")

    for row in rows:
        lines.extend(
            [
                f"## {row['index']}. Priorytet {row['priority']} - {row['category']}",
                "",
                f"Akcja QA: `{row['action']}` - {row['action_label']}",
                f"Status naprawy: `{row['actionability']}`",
                f"Pewnosc QA: `{row['repair_confidence']}`",
                "",
                f"Film edytowany: `{row['edited_range']}`",
                f"Do odsluchu w edicie z marginesem: `{row['edited_compare_range']}`",
                "",
                "Raw do porownania:",
            ]
        )
        if row["raw_compare_ranges"]:
            for raw_index, raw_range in enumerate(row["raw_compare_ranges"], start=1):
                timeline_id = row["raw_ranges"][raw_index - 1]["timeline_id"]
                lines.append(f"- `{raw_range}` (fragment raw/timeline #{timeline_id})")
        else:
            lines.append("- brak mapowania na raw; sprawdz recznie")

        raw_context = str(row.get("raw_context", "")).strip()
        if raw_context:
            lines.extend(["", "Kontekst raw:", "", f"> {raw_context}"])

        if make_clips:
            lines.extend(["", "Klipy porownawcze:"])
            if row["edited_clip_path"]:
                lines.append(f"- edit: `{row['edited_clip_path']}`")
            for raw_clip_path in row["raw_clip_paths"]:
                lines.append(f"- raw: `{raw_clip_path}`")

        lines.extend(
            [
                "",
                "Co brzmi podejrzanie:",
                "",
                row["description"],
                "",
                "Podejrzany tekst:",
                "",
                f"> {row['affected_text']}",
                "",
                "Uzasadnienie QA:",
                "",
                row["repair_rationale"] or row["suggested_action"],
                "",
                "Instrukcja dla montazysty:",
                "",
                row["editor_instruction"],
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_editor_review_csv(
    path: Path,
    rows: list[dict[str, Any]],
    edited_video_label: str,
    raw_video_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nr",
                "priorytet",
                "kategoria",
                "akcja_qa",
                "status_naprawy",
                "pewnosc_qa",
                "film_edytowany_czas",
                "film_edytowany_do_odsluchu",
                "raw_do_porownania",
                "kontekst_raw",
                "co_brzmi_podejrzanie",
                "podejrzany_tekst",
                "uzasadnienie_qa",
                "instrukcja_dla_montazysty",
                "plik_edytowany",
                "plik_raw",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "nr": row["index"],
                    "priorytet": row["priority"],
                    "kategoria": row["category"],
                    "akcja_qa": row["action"],
                    "status_naprawy": row["actionability"],
                    "pewnosc_qa": row["repair_confidence"],
                    "film_edytowany_czas": row["edited_range"],
                    "film_edytowany_do_odsluchu": row["edited_compare_range"],
                    "raw_do_porownania": " | ".join(row["raw_compare_ranges"]),
                    "kontekst_raw": row["raw_context"],
                    "co_brzmi_podejrzanie": row["description"],
                    "podejrzany_tekst": row["affected_text"],
                    "uzasadnienie_qa": row["repair_rationale"] or row["suggested_action"],
                    "instrukcja_dla_montazysty": row["editor_instruction"],
                    "plik_edytowany": edited_video_label,
                    "plik_raw": raw_video_label,
                }
            )
