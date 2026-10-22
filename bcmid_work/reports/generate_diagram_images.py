"""
Generates high-resolution publication-quality PNG diagrams for BCMID report:
1. System Flowchart (7-bloc pipeline)
2. n8n 9-Node Clinical Workflow
3. Preprocessing & Quality Pipeline
4. Quality & Safety Feedback Loop Sequence
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

out_dir = Path("bcmid_work/reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ── 1. SYSTEM FLOWCHART ────────────────────────────────────────────────────────
def create_system_flowchart():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Background color
    fig.patch.set_facecolor('#ffffff')

    # Subgraphs (Phases)
    phases = [
        ("Phase 1: Ingestion & Quality Control", 2, 72, 96, 25, "#eff6ff", "#bfdbfe"),
        ("Phase 2: Vision & Knowledge Retrieval", 2, 48, 96, 21, "#f0fdf4", "#bbf7d0"),
        ("Phase 3: Cognitive Synthesis & Safety Verification", 2, 24, 96, 21, "#fefce8", "#fef08a"),
        ("Phase 4: Structured Clinical Delivery", 2, 2, 96, 19, "#faf5ff", "#e9d5ff"),
    ]

    for title, x, y, w, h, bg, border in phases:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1,rounding_size=2",
                                      linewidth=1.2, edgecolor=border, facecolor=bg)
        ax.add_patch(rect)
        ax.text(x + 2, y + h - 4, title, fontsize=9, fontweight='bold', color='#1e293b')

    # Helper for boxes
    def draw_box(x, y, w, h, text, bg='#ffffff', border='#3b82f6', text_color='#0f172a', fontsize=7.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5",
                                      linewidth=1, edgecolor=border, facecolor=bg)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize, ha='center', va='center',
                color=text_color, fontweight='bold', multialignment='center')

    # Helper for arrows
    def draw_arrow(x1, y1, x2, y2, color='#64748b', lw=1.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=3, shrinkB=3))

    # Phase 1 Nodes
    draw_box(4, 75, 24, 14, "Raw Dossier\n(Mammo PNG + US PNG + Text)", bg='#ffffff', border='#94a3b8')
    draw_arrow(28, 82, 34, 82)
    draw_box(34, 75, 26, 14, "Bloc 1: Ingestion Agent\n(SHA-256 & Decodability)", bg='#dbeafe', border='#3b82f6')
    draw_arrow(60, 82, 66, 82)
    draw_box(66, 75, 30, 14, "Bloc 2: Data Quality Agent\n(Otsu, Inversion, EasyOCR)", bg='#dbeafe', border='#3b82f6')

    # Down to Phase 2
    draw_arrow(81, 75, 81, 62)
    draw_box(68, 51, 26, 14, "GroupKFold (k=5)\n(0 Patient Leakage)", bg='#e0e7ff', border='#6366f1')

    # Phase 2 Nodes
    draw_arrow(68, 58, 56, 58)
    draw_box(28, 51, 28, 14, "Bloc 3: Vision Agent\n(ResNet-18 Dual-Branch)", bg='#dcfce7', border='#16a34a')
    draw_arrow(28, 58, 20, 58)
    draw_box(4, 51, 16, 14, "Grad-CAM\n(387 Maps)", bg='#f0fdf4', border='#16a34a')

    draw_arrow(81, 51, 81, 40)

    # Phase 3 Nodes
    draw_box(4, 27, 24, 14, "Multimodal Fusion\n& Shannon Entropy", bg='#fef9c3', border='#ca8a04')
    draw_arrow(16, 51, 16, 41)
    draw_arrow(28, 34, 34, 34)
    draw_box(34, 27, 28, 14, "Bloc 5: Reasoning Agent\n(Qwen2.5-3B-Instruct)", bg='#fef08a', border='#ca8a04')
    draw_arrow(62, 34, 68, 34)
    draw_box(68, 27, 28, 14, "Bloc 6: Safety Agent\n(Deterministic Audit Gates)", bg='#fee2e2', border='#dc2626')

    # Feedback arrow
    ax.annotate('', xy=(48, 41), xytext=(82, 41),
                arrowprops=dict(arrowstyle="->", color='#dc2626', lw=1.2, ls='--'))
    ax.text(65, 42.5, "Active Feedback Revision Loop (Max 2)", fontsize=7, color='#dc2626', ha='center', fontweight='bold')

    # Down to Phase 4
    draw_arrow(82, 27, 82, 16)
    draw_box(4, 5, 26, 11, "Bloc 7: Structured Gen.\n(Report Engine)", bg='#f3e8ff', border='#9333ea')
    draw_arrow(82, 10.5, 30, 10.5)

    draw_box(34, 5, 28, 11, "Doctor View (PDF)\n(ACR BI-RADS + Rx)", bg='#ffffff', border='#9333ea')
    draw_arrow(30, 10.5, 34, 10.5)

    draw_box(66, 5, 30, 11, "Technician View (HTML)\n(Grad-CAM + Provenance)", bg='#ffffff', border='#9333ea')
    draw_arrow(62, 10.5, 66, 10.5)

    plt.tight_layout()
    plt.savefig(out_dir / "diagram_system_flowchart.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved diagram_system_flowchart.png")

# ── 2. n8n CLINICAL WORKFLOW ───────────────────────────────────────────────────
def create_n8n_workflow():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    def draw_box(x, y, w, h, text, bg='#ffffff', border='#3b82f6', text_color='#0f172a', fontsize=7.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5",
                                      linewidth=1, edgecolor=border, facecolor=bg)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize, ha='center', va='center',
                color=text_color, fontweight='bold', multialignment='center')

    def draw_arrow(x1, y1, x2, y2, color='#64748b', lw=1.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=3, shrinkB=3))

    # Triggers
    draw_box(2, 70, 18, 18, "Node 1: Webhook\nPOST /triage\n(Hospital HIS)", bg='#eff6ff', border='#3b82f6')
    draw_box(2, 43, 18, 18, "Node 2: Manual\nClinical Sandbox\n(Single Patient)", bg='#eff6ff', border='#3b82f6')
    draw_box(2, 16, 18, 18, "Node 3: Web Form\n/form/bcmid\n(Physician Portal)", bg='#eff6ff', border='#3b82f6')

    # Aggregation
    draw_arrow(20, 79, 26, 52)
    draw_arrow(20, 52, 26, 52)
    draw_arrow(20, 25, 26, 52)

    draw_box(26, 43, 16, 18, "Node 4: Set\nExtract\npseudo_id", bg='#f8fafc', border='#64748b')
    draw_arrow(42, 52, 48, 52)

    # HTTP Request
    draw_box(48, 41, 18, 22, "Node 5: HTTP\nGET /kaggle/\nreports/{id}/triage\n(FastAPI)", bg='#f0fdf4', border='#16a34a')
    draw_arrow(66, 52, 71, 52)

    # Decision IF
    # Diamond for decision
    poly = patches.Polygon([[71, 52], [77, 65], [83, 52], [77, 39]], closed=True,
                           edgecolor='#d97706', facecolor='#fef3c7', linewidth=1.2)
    ax.add_patch(poly)
    ax.text(77, 52, "Node 6: IF\nescalation\n== true", fontsize=6.5, ha='center', va='center',
            color='#92400e', fontweight='bold')

    # Urgent vs Routine
    draw_arrow(77, 65, 87, 78, color='#dc2626')
    ax.text(80, 74, "TRUE", fontsize=7, color='#dc2626', fontweight='bold')
    draw_box(86, 70, 12, 18, "Node 7: URGENT\nAlert (BI-RADS 4-5)\nCore Biopsy", bg='#fee2e2', border='#dc2626', text_color='#991b1b', fontsize=6.5)

    draw_arrow(77, 39, 87, 26, color='#16a34a')
    ax.text(80, 30, "FALSE", fontsize=7, color='#16a34a', fontweight='bold')
    draw_box(86, 17, 12, 18, "Node 8: ROUTINE\nNotice (BI-RADS 1-3)\nSurveillance", bg='#dcfce7', border='#16a34a', text_color='#166534', fontsize=6.5)

    # Respond to Webhook
    draw_arrow(98, 79, 100, 79)
    draw_arrow(98, 26, 100, 26)

    plt.tight_layout()
    plt.savefig(out_dir / "diagram_n8n_workflow.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved diagram_n8n_workflow.png")

# ── 3. PREPROCESSING PIPELINE ──────────────────────────────────────────────────
def create_preprocessing_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    def draw_box(x, y, w, h, text, bg='#ffffff', border='#3b82f6', text_color='#0f172a', fontsize=7):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1",
                                      linewidth=1, edgecolor=border, facecolor=bg)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize, ha='center', va='center',
                color=text_color, fontweight='bold', multialignment='center')

    def draw_arrow(x1, y1, x2, y2, color='#64748b', lw=1.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=2, shrinkB=2))

    steps = [
        ("Raw Radiograph\n(Mammo / US PNG)", '#f8fafc', '#94a3b8'),
        ("1. Polarity Check\nCorner Avg > 127\nInversion if White", '#eff6ff', '#3b82f6'),
        ("2. Otsu Crop\nRemove Collimator\n& Air Background", '#eff6ff', '#3b82f6'),
        ("3. Letterbox Resize\nM:1280x1024\nUS:360x512", '#eff6ff', '#3b82f6'),
        ("4. EasyOCR Inpaint\nZero-fill Burned-in\nDevice Metadata", '#eff6ff', '#3b82f6'),
        ("5. CLAHE / Filter\nM: CLAHE clip=2.0\nUS: Median k=3", '#eff6ff', '#3b82f6'),
        ("Normalized Tensor\n(512x512, uint8)\nClean for Model", '#f0fdf4', '#16a34a')
    ]

    x = 1
    w = 12.5
    h = 24
    y = 3
    for i, (text, bg, border) in enumerate(steps):
        draw_box(x, y, w, h, text, bg=bg, border=border)
        if i < len(steps) - 1:
            draw_arrow(x + w, y + h/2, x + w + 1.8, y + h/2)
        x += w + 2.0

    plt.tight_layout()
    plt.savefig(out_dir / "diagram_preprocessing_pipeline.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved diagram_preprocessing_pipeline.png")

# ── 4. SAFETY FEEDBACK LOOP ────────────────────────────────────────────────────
def create_safety_sequence():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    def draw_actor(x, text, color='#0b2545'):
        rect = patches.FancyBboxPatch((x - 12, 85), 24, 10, boxstyle="round,pad=0.5,rounding_size=1.5",
                                      linewidth=1.2, edgecolor=color, facecolor='#f8fafc')
        ax.add_patch(rect)
        ax.text(x, 90, text, fontsize=7.5, ha='center', va='center', fontweight='bold', color=color)
        ax.plot([x, x], [10, 85], color='#cbd5e1', linestyle='--', lw=1)

    draw_actor(20, "Bloc 5: Reasoning LLM\n(Qwen2.5-3B)")
    draw_actor(55, "Bloc 6: Safety Agent\n(Deterministic Audit)")
    draw_actor(85, "Bloc 7: Structured Output\n(Doctor / PACS)")

    def draw_msg(x1, x2, y, text, color='#2563eb', is_dashed=False):
        ls = '--' if is_dashed else '-'
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, ls=ls, shrinkA=2, shrinkB=2))
        ax.text((x1 + x2)/2, y + 2, text, fontsize=7, ha='center', va='bottom', color=color, fontweight='bold')

    # Step 1
    draw_msg(20, 55, 75, "1. Submit Candidate Clinical JSON", color='#2563eb')

    # Step 2: Inspection box
    rect = patches.Rectangle((42, 53), 26, 17, linewidth=1, edgecolor='#94a3b8', facecolor='#f1f5f9')
    ax.add_patch(rect)
    ax.text(55, 61.5, "Audit Gates Check:\n- Lexicon Groundedness\n- Malignancy Prob ±0.6%\n- Entropy ≥ 0.50 Alert\n- ACR Citations Verified",
            fontsize=6.5, ha='center', va='center', color='#334155')

    # Step 3: Feedback revision
    draw_msg(55, 20, 45, "2. [Violation] Return Revision Feedback & Diagnostic Hints", color='#dc2626', is_dashed=True)
    draw_msg(20, 55, 33, "3. Resubmit Corrected Clinical JSON (Retry cycle <= 2)", color='#2563eb')

    # Step 4: Approval
    draw_msg(55, 85, 18, "4. Status: APPROVED_BY_SAFETY_AGENT -> Publish PDF / HTML", color='#16a34a')

    plt.tight_layout()
    plt.savefig(out_dir / "diagram_safety_feedback_loop.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved diagram_safety_feedback_loop.png")

if __name__ == "__main__":
    create_system_flowchart()
    create_n8n_workflow()
    create_preprocessing_pipeline()
    create_safety_sequence()
    print("All diagrams generated successfully!")
