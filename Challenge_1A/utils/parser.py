import pdfplumber
from collections import defaultdict
import re

def extract_pdf_structure(path):
    """Enhanced PDF structure extraction with better line grouping"""
    items = []
    with pdfplumber.open(path) as pdf:
        for p_idx, page in enumerate(pdf.pages):
            page_width = page.width
            page_height = page.height
            
            lines = defaultdict(list)
            for ch in page.chars:
                y = round(ch["top"])
                lines[y].append(ch)
            
      
            merged_lines = {}
            sorted_ys = sorted(lines.keys())
            if sorted_ys:
                current_y = sorted_ys[0]
                merged_lines[current_y] = lines[current_y]
                for y in sorted_ys[1:]:
                    if abs(y - current_y) < 5: 
                        merged_lines[current_y].extend(lines[y])
                    else:
                        current_y = y
                        merged_lines[current_y] = lines[y]
            
            for y in sorted(merged_lines.keys(), reverse=True):
                chars = merged_lines[y]
                text = ''.join(c["text"] for c in chars).strip()
                if not text:
                    continue
                
                sizes = [c["size"] for c in chars]
                fonts = [c["fontname"].lower() for c in chars]
                is_bold = any('bold' in f or 'bd' in f or 'black' in f for f in fonts)
                
                item = {
                    "text": text,
                    "fontsize": sum(sizes)/len(sizes),
                    "bold": is_bold,
                    "x0": min(c["x0"] for c in chars),
                    "x1": max(c["x1"] for c in chars),
                    "page": p_idx + 1, 
                    "page_width": page_width,
                    "page_height": page_height,
                    "y": y
                }
                items.append(item)
    return items