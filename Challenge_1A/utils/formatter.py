from .heuristics import infer_thresholds, detect_heading
from itertools import groupby

def format_outline(raw_items):
    title = ""
    first_page_items = [it for it in raw_items if it["page"] == 1]
    if first_page_items:
        top_items = [it for it in first_page_items if it["y"] < 0.2 * it["page_height"]]
        if top_items:
            top_items.sort(key=lambda x: x["fontsize"], reverse=True)
            for candidate in top_items:
                midpoint = (candidate['x0'] + candidate['x1']) / 2
                if abs(midpoint - candidate['page_width']/2) < 0.2 * candidate['page_width']:
                    title = candidate["text"].strip()
                    break
    thresholds = infer_thresholds(raw_items)
    
    headings = []
    for it in raw_items:
        level = detect_heading(it, thresholds)
        if level:
            headings.append({
                "level": level,
                "text": it["text"].strip(),
                "page": it["page"],
                "y": it["y"]
            })
    
    cleaned = []
    for h in headings:
        if not cleaned or h["text"] != cleaned[-1]["text"] or h["page"] != cleaned[-1]["page"]:
            cleaned.append(h)
    
    outline = []
    for h in cleaned:
        outline.append({
            "level": h["level"],
            "text": h["text"],
            "page": h["page"]
        })
    
    return {
        "title": title,
        "outline": outline
    }