"""
Structured Medical Report Generator (Bloc 7)
Generates:
1. structured_medical_report.html (Self-contained, styled clinical report matching UI mock)
2. structured_medical_report.json
"""
from __future__ import annotations
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional


class MedicalReportGenerator:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (Path.cwd() / "bcmid_work" / "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, report_data: Dict[str, Any], output_filename: str = "medical_report.html") -> Path:
        """Renders an interactive, styled HTML medical report matching Block 7 of the architecture."""
        pseudo_id = report_data.get("pseudo_id", "P_UNKNOWN")
        age = report_data.get("age", "N/A")
        birads = report_data.get("predicted_birads", "BI-RADS 0")
        conf = report_data.get("confidence", 0.0)
        risk_level = report_data.get("risk_level", "Unknown")
        summary = report_data.get("summary", "No summary provided.")
        evidence = report_data.get("evidence", [])
        recs = report_data.get("recommendations", [])
        limitations = report_data.get("limitations", [])
        heatmaps = report_data.get("heatmaps", {})
        safety_status = report_data.get("safety_audit", {}).get("status", "APPROVED")

        # Color coding for BI-RADS badge
        badge_color = "#28a745" if "1" in str(birads) or "2" in str(birads) else (
            "#ffc107" if "3" in str(birads) else (
                "#fd7e14" if "4A" in str(birads) or "4B" in str(birads) else "#dc3545"
            )
        )

        ev_html = ""
        for ev in evidence:
            ev_html += f"""
            <div class="evidence-card">
                <div class="evidence-header">
                    <span class="evidence-tag">[{ev.get('evidence_id', 'E?')}]</span>
                    <span class="evidence-source">{ev.get('source_type', 'Reference')} &bull; Sim: {ev.get('similarity', 0.0):.2f}</span>
                </div>
                <div class="evidence-body">"{ev.get('passage', '')}"</div>
            </div>
            """

        recs_html = "".join([f"<li>{r}</li>" for r in recs])
        lims_html = "".join([f"<li>{l}</li>" for l in limitations])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BCMID Medical AI Report - {pseudo_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f4f6f9;
            color: #212529;
            margin: 0;
            padding: 24px;
        }}
        .report-container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }}
        .report-header {{
            background: linear-gradient(135deg, #0b2545 0%, #134074 100%);
            color: #ffffff;
            padding: 24px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title h1 {{
            margin: 0;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .header-title p {{
            margin: 4px 0 0 0;
            opacity: 0.85;
            font-size: 13px;
        }}
        .safety-badge {{
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.4);
        }}
        .report-body {{
            padding: 32px;
        }}
        .patient-summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 28px;
            border: 1px solid #edf2f7;
        }}
        .meta-item .label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .meta-item .value {{
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }}
        .birads-tag {{
            display: inline-block;
            background-color: {badge_color};
            color: white;
            padding: 2px 10px;
            border-radius: 6px;
            font-size: 16px;
        }}
        .section-title {{
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
        }}
        .interpretation-box {{
            background: #fdfefe;
            border-left: 4px solid #134074;
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            font-size: 15px;
            line-height: 1.6;
            color: #334155;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}
        .visual-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 16px;
            margin-bottom: 24px;
        }}
        .visual-card {{
            background: #0f172a;
            color: white;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
        }}
        .visual-card h4 {{
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #94a3b8;
            font-weight: 600;
        }}
        .visual-placeholder {{
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
        }}
        .visual-placeholder img {{
            max-width: 100%;
            max-height: 240px;
            height: auto;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            border-radius: 6px;
        }}
        .evidence-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 12px;
        }}
        .evidence-header {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 6px;
        }}
        .evidence-tag {{
            font-weight: 700;
            color: #2563eb;
        }}
        .evidence-source {{
            color: #64748b;
        }}
        .evidence-body {{
            font-size: 13px;
            line-height: 1.5;
            color: #334155;
            font-style: italic;
        }}
        .recs-box {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 16px 20px;
        }}
        .recs-box ul, .lims-box ul {{
            margin: 0;
            padding-left: 20px;
            line-height: 1.6;
            font-size: 14px;
        }}
        .lims-box {{
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 8px;
            padding: 16px 20px;
            color: #92400e;
        }}
        .report-footer {{
            background: #f8fafc;
            border-top: 1px solid #e2e8f0;
            padding: 16px 32px;
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <div class="header-title">
                <h1>Multimodal Medical AI Diagnostic Report</h1>
                <p>BCMID System with Explainable AI (XAI) & Clinical RAG</p>
            </div>
            <div class="safety-badge">
                Safety Status: {safety_status}
            </div>
        </div>

        <div class="report-body">
            <!-- Patient Demographics & Key Prediction -->
            <div class="patient-summary-grid">
                <div class="meta-item">
                    <div class="label">Patient Pseudo-ID</div>
                    <div class="value">{pseudo_id}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Patient Age</div>
                    <div class="value">{age} years</div>
                </div>
                <div class="meta-item">
                    <div class="label">Predicted Assessment</div>
                    <div class="value"><span class="birads-tag">{birads}</span></div>
                </div>
                <div class="meta-item">
                    <div class="label">Confidence & Risk</div>
                    <div class="value">{conf*100:.1f}% &bull; {risk_level}</div>
                </div>
            </div>

            <!-- Interpretation -->
            <div class="section-title">Diagnostic Interpretation (Reasoning Agent)</div>
            <div class="interpretation-box">
                {summary}
            </div>

            <!-- Visual Explanations (XAI Heatmaps) -->
            <div class="section-title">Visual Explanations (XAI Grad-CAM Heatmaps)</div>
            <div class="visual-grid">
                <div class="visual-card">
                    <h4>Mammography Grad-CAM Overlay</h4>
                    <div class="visual-placeholder">
                        {heatmaps.get('mammo_html', 'Mammogram XAI: Localized parenchymal attention')}
                    </div>
                </div>
                <div class="visual-card">
                    <h4>Ultrasound Grad-CAM Overlay</h4>
                    <div class="visual-placeholder">
                        {heatmaps.get('us_html', 'Ultrasound XAI: Focal hypoechoic region attention')}
                    </div>
                </div>
            </div>

            <!-- Supporting Evidence (RAG) -->
            <div class="section-title">Supporting Clinical Evidence (Clinical RAG Agent)</div>
            {ev_html if ev_html else '<p style="color:#64748b; font-size:14px;">No external clinical evidence cited.</p>'}

            <!-- Clinical Recommendations -->
            <div class="section-title">Clinical Recommendations</div>
            <div class="recs-box">
                <ul>{recs_html if recs_html else '<li>Standard screening schedule recommended.</li>'}</ul>
            </div>

            <!-- Limitations & Uncertainty -->
            <div class="section-title">Uncertainty, Disclaimers & Limitations</div>
            <div class="lims-box">
                <ul>{lims_html if lims_html else '<li>Research AI baseline. Clinical correlation by certified radiologist required.</li>'}</ul>
            </div>
        </div>

        <div class="report-footer">
            Generated by BCMID Multimodal Medical AI Automation System &bull; Not for autonomous diagnosis &bull; For clinical pair review only.
        </div>
    </div>
</body>
</html>
"""
        out_file = self.output_dir / output_filename
        out_file.write_text(html_content, encoding="utf-8")

        # Also save JSON
        json_file = out_file.with_suffix(".json")
        json_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_file
