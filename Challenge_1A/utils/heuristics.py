from collections import Counter
import re


def infer_thresholds(items):
    """More robust threshold detection with size clustering"""
    sizes = [it["fontsize"] for it in items]
    size_counts = Counter(sizes)

    distinct_sizes = sorted(
        {s for s in sizes if size_counts[s] / len(items) < 0.05}, reverse=True
    )

    if not distinct_sizes:
        distinct_sizes = sorted(set(sizes), reverse=True)[:3]

    thresholds = {
        "H1": distinct_sizes[0] if len(distinct_sizes) > 0 else 14,
        "H2": distinct_sizes[1] if len(distinct_sizes) > 1 else distinct_sizes[0],
        "H3": distinct_sizes[2] if len(distinct_sizes) > 2 else distinct_sizes[1] if len(distinct_sizes) > 1 else distinct_sizes[0]
    }

    if len(distinct_sizes) > 1 and abs(thresholds["H1"] - thresholds["H2"]) < 2:
        thresholds["H2"] = (thresholds["H1"] + thresholds["H2"]) / 2
    if len(distinct_sizes) > 2 and abs(thresholds["H2"] - thresholds["H3"]) < 2:
        thresholds["H3"] = (thresholds["H2"] + thresholds["H3"]) / 2

    return thresholds


def detect_heading(item, thresholds):
    """Enhanced heading detection with pattern matching"""
    txt = item["text"].strip()
    fs = item["fontsize"]
    page_width = item["page_width"]

    if (re.fullmatch(r"[-–—=_.•*~\s]+", txt) or
        re.match(r"^\d+$", txt) or
        "page" in txt.lower() or
        item["y"] > 0.9 * item["page_height"]):
        return None

    if item["y"] < 0.03 * item["page_height"] and not item["bold"]:
        return None

    midpoint = (item['x0'] + item['x1']) / 2
    is_centered = abs(midpoint - page_width/2) < 0.15 * page_width
    is_left_aligned = item['x0'] < 0.1 * page_width

    numbered = re.match(r"^(?:[\s]*[0-9]+[\.\)]|[\s]*[IVXLCDM]+\.)", txt)
    appendix = re.match(r"^\s*(appendix|attachment)\b", txt, re.IGNORECASE)

    if numbered or appendix:
        for level in ["H1", "H2", "H3"]:
            if abs(fs - thresholds[level]) < 1.0:
                return level
        return "H2" if appendix else None

    for level in ["H1", "H2", "H3"]:
        if abs(fs - thresholds[level]) < 1.0:
            if (item['bold'] or is_centered or is_left_aligned or len(txt.split()) < 8 or txt.isupper()):
                return level

    return None