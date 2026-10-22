"""
Patch all generated medical reports to fix the overlapping image/text layout.
Replaces the old .visual-placeholder and .visual-card CSS with the corrected
responsive version in-place, without changing any clinical data.
"""
import re
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")

REPORTS_DIR = Path("bcmid_work/reports")

OLD_CSS = """.visual-card {
            background: #0f172a;
            color: white;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .visual-card h4 {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #94a3b8;
        }
        .visual-placeholder {
            width: 100%;
            height: 220px;
            background: #1e293b;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748b;
            font-size: 13px;
        }"""

NEW_CSS = """.visual-card {
            background: #0f172a;
            color: white;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
        }
        .visual-card h4 {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #94a3b8;
            font-weight: 600;
        }
        .visual-placeholder {
            width: 100%;
            min-height: 260px;
            height: auto;
            background: #1e293b;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #94a3b8;
            font-size: 13px;
            padding: 14px;
            box-sizing: border-box;
        }
        .visual-placeholder img {
            max-width: 100%;
            max-height: 240px;
            height: auto;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            border-radius: 6px;
        }"""

OLD_SECTION_CSS = """        .section-title {
            font-size: 16px;
            font-weight: 700;
            color: #0b2545;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
            margin-top: 28px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }"""

NEW_SECTION_CSS = """        .section-title {
            font-size: 16px;
            font-weight: 700;
            color: #0b2545;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
            margin-top: 32px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            clear: both;
        }"""

OLD_VG = """        .visual-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 16px;
        }"""

NEW_VG = """        .visual-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 16px;
            margin-bottom: 24px;
        }"""

# Fix img inline style in existing reports
OLD_IMG_MAMMO = 'style="max-width:320px; border-radius:8px; border:2px solid #134074;"'
NEW_IMG = 'style="max-width:100%; max-height:230px; object-fit:contain; border-radius:8px; border:2px solid #3b82f6; display:block; margin:0 auto;"'

OLD_CAPTION_COLOR = 'color:#64748b; margin-top:4px;'
NEW_CAPTION_COLOR = 'color:#cbd5e1; margin-top:8px; margin-bottom:0;'

patched = 0
for html_file in sorted(REPORTS_DIR.glob("medical_report_*.html")):
    content = html_file.read_text(encoding="utf-8")
    original = content

    content = content.replace(OLD_CSS, NEW_CSS)
    content = content.replace(OLD_SECTION_CSS, NEW_SECTION_CSS)
    content = content.replace(OLD_VG, NEW_VG)
    content = content.replace(OLD_IMG_MAMMO, NEW_IMG)
    content = content.replace(OLD_CAPTION_COLOR, NEW_CAPTION_COLOR)

    if content != original:
        html_file.write_text(content, encoding="utf-8")
        print(f"  [PATCHED] {html_file.name}")
        patched += 1
    else:
        print(f"  [skip]    {html_file.name} (already up to date)")

print(f"\nDone. {patched} / {len(list(REPORTS_DIR.glob('medical_report_*.html')))} reports patched.")
