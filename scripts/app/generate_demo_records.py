#!/usr/bin/env python3
from pathlib import Path
import sys
import fitz

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".oncotwin-demo-records")
OUT.mkdir(parents=True, exist_ok=True)

records = {
    "01_pathology.pdf": [
        "PATHOLOGY SUMMARY",
        "Date: 2026-01-12",
        "Diagnosis: metastatic breast cancer",
        "Estrogen receptor: positive",
        "Progesterone receptor: positive",
        "HER2: negative",
        "Biopsy confirms metastatic breast carcinoma.",
    ],
    "02_treatment_genomics.pdf": [
        "ONCOLOGY SUMMARY",
        "Date: 2026-02-02",
        "Current treatment: capecitabine",
        "Known metastatic disease in liver and bone.",
        "Genomic testing: PIK3CA mutation detected.",
    ],
    "03_radiology.pdf": [
        "CT CHEST ABDOMEN PELVIS - RADIOLOGY",
        "Date: 2026-03-18",
        "Known metastatic disease in liver and bone.",
        "Impression: stable disease.",
        "No new sites of metastatic disease are described in this synthetic report.",
    ],
}

for name, lines in records.items():
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    for i, line in enumerate(lines):
        size = 15 if i == 0 else 11
        page.insert_text((72, y), line, fontsize=size)
        y += 26 if i == 0 else 20
    path = OUT / name
    doc.save(path)
    doc.close()
    print(path)
