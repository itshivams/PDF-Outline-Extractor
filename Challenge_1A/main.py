import os
import json
from utils.parser import extract_pdf_structure
from utils.formatter import format_outline

INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for file in os.listdir(INPUT_DIR):
    if file.lower().endswith('.pdf'):
        path = os.path.join(INPUT_DIR, file)
        raw = extract_pdf_structure(path)
        output = format_outline(raw)
        outpath = os.path.join(OUTPUT_DIR, file.replace('.pdf', '.json'))
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)



