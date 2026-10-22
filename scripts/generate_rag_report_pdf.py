"""
Générateur de rapport PDF professionnel pour le Lab RAG Agent (Blocs 4, 5, 6, 7)
Système Médical Multimodal BCMID avec XAI
"""
import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # En-tête (sauf page 1)
        if self._pageNumber > 1:
            self.drawString(54, 800, "BCMID Multimodal Medical AI — Rapport Technique : Agent RAG & Raisonnement Clinique")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 792, 541, 792)
        
        # Pied de page (toutes les pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 541, 45)
        
        self.drawString(54, 32, "ESPRIT — Deep Learning 5ème Année | Projet BCMID Multimodal Medical AI with XAI")
        page_str = f"Page {self._pageNumber} sur {page_count}"
        self.drawRightString(541, 32, page_str)
        self.restoreState()


def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # Couleurs du thème
    c_primary = colors.HexColor("#0B2545")    # Bleu marine profond
    c_secondary = colors.HexColor("#134074")  # Bleu nuit
    c_accent = colors.HexColor("#0284C7")     # Bleu cyan médical
    c_dark = colors.HexColor("#1E293B")       # Gris très foncé
    c_light = colors.HexColor("#F8FAFC")      # Fond blanc cassé
    c_card_border = colors.HexColor("#E2E8F0")
    c_green = colors.HexColor("#16A34A")
    c_amber = colors.HexColor("#D97706")
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=c_primary,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_accent,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=6
    )
    
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_dark
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # ── TITRE DU DOCUMENT ───────────────────────────────────────────────────
    story.append(Paragraph("BCMID Multimodal Medical AI Automation System with XAI", subtitle_style))
    story.append(Paragraph("Dossier Technique & Clinique : Agent RAG, Raisonnement LLM & Sécurité (Blocs 4 à 7)", title_style))
    story.append(Paragraph("<b>Auteur :</b> Travaux dirigés & Laboratoire RAG Agent | <b>Projet :</b> Diagnostic du Cancer du Sein Assisté par IA", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=4, spaceAfter=14))
    
    # ── BLOC RÉSUMÉ EXÉCUTIF ────────────────────────────────────────────────
    summary_html = """
    <b>Résumé Exécutif :</b> Le notebook <code>rag-agent.ipynb</code> implémente les Blocs 4, 5, 6 et 7 de l'architecture cible du projet.
    Il assure l'intégration synergique entre les représentations visuelles issues du modèle de Deep Learning (Vision Agent - ResNet-18 multimodal),
    l'indexation vectorielle dense de rapports cliniques et de guides officiels ACR BI-RADS via embeddings BGE et FAISS (Clinical RAG),
    la synthèse médicale par un modèle de langage (LLM Qwen2.5-3B-Instruct), et un agent d'audit de sécurité médical avec boucle de rétroaction active.
    Ce document détaille chaque étape technique, les choix d'ingénierie, les métriques obtenues et la structure des rapports finaux générés.
    """
    card_data = [[Paragraph(summary_html, callout_style)]]
    card_table = Table(card_data, colWidths=[487])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BFDBFE")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 10))
    
    # ── SECTION 1 : ARCHITECTURE DU LAB ─────────────────────────────────────
    story.append(Paragraph("1. Positionnement dans l'Architecture Globale du Projet", h1_style))
    story.append(Paragraph(
        "L'architecture de référence du projet ('Multimodal Medical AI Automation System with XAI') est structurée en 7 blocs séquentiels. "
        "Le présent laboratoire couvre l'intégralité de la chaîne décisionnelle avale :", body_style
    ))
    
    arch_data = [
        [Paragraph("Bloc", table_header), Paragraph("Dénomination", table_header), Paragraph("Rôle Technique & Médical", table_header), Paragraph("Statut Lab RAG", table_header)],
        [Paragraph("<b>Bloc 1 & 2</b>", table_cell), Paragraph("Input & Data Quality", table_cell), Paragraph("Nettoyage, anonymisation, 5 plis groupés par patiente (332 cas)", table_cell), Paragraph("Validé (Notebook 1)", table_cell)],
        [Paragraph("<b>Bloc 3</b>", table_cell), Paragraph("Multimodal Vision Agent + XAI", table_cell), Paragraph("ResNet-18 (Mammo + Écho), calibration BI-RADS, Grad-CAM (387 cartes)", table_cell), Paragraph("Validé (Notebook 2)", table_cell)],
        [Paragraph("<b>Bloc 4</b>", table_cell), Paragraph("Clinical RAG Agent", table_cell), Paragraph("Recherche vectorielle dense : guides ACR BI-RADS + rapports historiques", table_cell), Paragraph("<b>Au cœur du Lab RAG</b>", table_cell)],
        [Paragraph("<b>Bloc 5</b>", table_cell), Paragraph("Medical Reasoning Agent", table_cell), Paragraph("Synthèse multimodale par LLM (Qwen2.5-3B-Instruct) sous prompt contraint", table_cell), Paragraph("<b>Au cœur du Lab RAG</b>", table_cell)],
        [Paragraph("<b>Bloc 6</b>", table_cell), Paragraph("Quality & Safety Agent", table_cell), Paragraph("Détection hallucinations, vérification de l'ancrage et boucle de révision", table_cell), Paragraph("<b>Au cœur du Lab RAG</b>", table_cell)],
        [Paragraph("<b>Bloc 7</b>", table_cell), Paragraph("Final Structured Output", table_cell), Paragraph("Production de comptes-rendus standardisés (JSON structuré + HTML)", table_cell), Paragraph("<b>Au cœur du Lab RAG</b>", table_cell)]
    ]
    t_arch = Table(arch_data, colWidths=[60, 110, 227, 90])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))
    
    # ── SECTION 2 : PRÉTRAITEMENT DU TEXTE ET SÉCURITÉ CLINIQUE ─────────────
    story.append(Paragraph("2. Prétraitement des Rapports & Prévention des Fuites d'Information (R0)", h1_style))
    story.append(Paragraph(
        "L'évaluation rigoureuse d'un agent RAG médical impose qu'aucune information de vérité terrain (ground truth) "
        "ne soit accessible au modèle de langage lors de la génération. Trois niveaux de sécurisation ont été appliqués :", body_style
    ))
    
    p_leakage = """
    • <b>Création de <code>text_blind</code> :</b> Suppression systématique et vérifiée de toute mention explicite de score BI-RADS (ex. <code>BI-RADS 4</code>, <code>ACR 2</code>). Vérification par expression régulière : 0 chiffre BI-RADS subsistant dans les textes aveugles.<br/>
    • <b>Purge des antécédents oncologiques directs de l'en-tête (R0f) :</b> Les mentions de traitements adjuvants (<code>mastectomy</code>, <code>lumpectomy</code>, <code>chemotherapy</code>, <code>pathologically proved cancer</code>) ont été caviardées par la balise <code>[CLINICAL HISTORY REMOVED]</code> pour 15 patientes afin d'éviter tout raccourci diagnostique trivial.<br/>
    • <b>Dé-doublonnage sémantique :</b> Élimination des rapports dupliqués ou de moins de 30 mots, aboutissant à une cohorte exploitable de <b>317 rapports cliniques propres</b>.
    """
    story.append(Paragraph(p_leakage, body_style))
    story.append(Spacer(1, 8))
    
    # ── SECTION 3 : LE MOTEUR CLINICAL RAG DENSE ────────────────────────────
    story.append(Paragraph("3. Moteur Vectoriel Clinical RAG Dense (BGE + FAISS) (R1 & R2)", h1_style))
    story.append(Paragraph(
        "Le système repose sur un double index vectoriel dense assurant la contextualisation clinique de chaque dossier :", body_style
    ))
    
    rag_boxes = [
        [Paragraph("<b>Index 1 : Base Clinique ACR BI-RADS Atlas</b>", table_header), Paragraph("<b>Index 2 : Cas Patients Historiques BCMID</b>", table_header)],
        [
            Paragraph(
                "• <b>Source :</b> Recommandations officielles de l'American College of Radiology (ACR).<br/>"
                "• <b>Contenu :</b> 11 règles canoniques couvrant les catégories BI-RADS 0 à 6, les patrons de densité ACR A-D, et le lexique échographique (forme, contours, échogénicité).<br/>"
                "• <b>Index :</b> <code>acr_guidelines_index.faiss</code>.<br/>"
                "• <b>Objectif :</b> Fournir au LLM la justification normative et la conduite à tenir officielle (ex. biopsie au trocart vs surveillance 6 mois).",
                table_cell
            ),
            Paragraph(
                "• <b>Source :</b> Rapports textuels anonymisés du dataset BCMID.<br/>"
                "• <b>Découpage :</b> 317 chunks textuels nettoyés.<br/>"
                "• <b>Index :</b> <code>rag_index.faiss</code>.<br/>"
                "• <b>Objectif :</b> Raisonnement par analogie de cas (Case-Based Reasoning). Identification de patientes présentant des anomalies sémiologiques similaires.<br/>"
                "• <b>Isolation stricte :</b> Les cas du même pli ou de la patiente test sont exclus pour éviter toute contamination.",
                table_cell
            )
        ]
    ]
    t_rag = Table(rag_boxes, colWidths=[240, 247])
    t_rag.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BACKGROUND', (0,1), (-1,1), c_light),
    ]))
    story.append(t_rag)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Modèle d'embedding :</b> <code>BAAI/bge-small-en-v1.5</code> (dimension 384). Recherche par similarité cosinus "
        "via <code>IndexFlatIP</code> après normalisation L2 des vecteurs, garantissant une latence d'inférence < 5 ms par requête.", body_style
    ))
    
    story.append(PageBreak())
    
    # ── SECTION 4 : FORMULATION DU PROMPT MULTIMODAL ────────────────────────
    story.append(Paragraph("4. Conditionnement Multimodal & Ingestion des Scores Vision + XAI (R3)", h1_style))
    story.append(Paragraph(
        "Pour chaque patiente du pli d'évaluation (pli 0, 64 patientes), un dossier multimodal unifié est assemblé. "
        "L'agent de raisonnement est testé selon 3 conditions expérimentales contrôlées :", body_style
    ))
    
    cond_data = [
        [Paragraph("Condition", table_header), Paragraph("Données Fournies dans le Prompt", table_header), Paragraph("Nombre de Cas", table_header), Paragraph("Longueur Médiane", table_header)],
        [Paragraph("<b>image</b>", table_cell), Paragraph("Scores de probabilité du ResNet-18 (Mammo + US) + Résumé textuel des zones Grad-CAM", table_cell), Paragraph("64", table_cell), Paragraph("245 mots", table_cell)],
        [Paragraph("<b>texte</b>", table_cell), Paragraph("Rapport clinique nettoyé + Preuves RAG E1/E2 issues d'autres patientes (si rapport disponible)", table_cell), Paragraph("60", table_cell), Paragraph("442 mots", table_cell)],
        [Paragraph("<b>les_deux</b>", table_cell), Paragraph("Fusion intégrale : Scores vision + Zones chaudes Grad-CAM + Rapport patiente + Preuves RAG E1/E2", table_cell), Paragraph("64", table_cell), Paragraph("486 mots", table_cell)],
    ]
    t_cond = Table(cond_data, colWidths=[65, 272, 75, 75])
    t_cond.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_cond)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Intégration de l'Explicabilité XAI (Grad-CAM) dans le Prompt :", h2_style))
    story.append(Paragraph(
        "Plutôt que de fournir des pixels bruts, les 387 cartes de chaleur Grad-CAM générées par le Vision Agent "
        "sont traduites en descriptions spatiales sémantiques (ex. <i>'milieu de l'image, côté opposé à la paroi'</i>, "
        "<i>'haut de l'image, côté paroi thoracique'</i>) et injectées sous la clé <code>HEATMAP SUMMARY</code>. "
        "Le LLM sait ainsi précisément quelles régions ont motivé la prédiction du réseau convolutif.", body_style
    ))
    story.append(Spacer(1, 8))
    
    # ── SECTION 5 : AGENT DE RAISONNEMENT & LLM ─────────────────────────────
    story.append(Paragraph("5. Modèle de Raisonnement Médical (Qwen2.5-3B-Instruct) (R4a)", h1_style))
    story.append(Paragraph(
        "Le modèle de langage sélectionné est <b>Qwen2.5-3B-Instruct</b>, exécuté en précision <code>float16</code> sur GPU Tesla T4 (occupation VRAM : 3.1 Go). "
        "Ce modèle a été choisi pour son excellent respect des schémas JSON stricts et sa capacité de raisonnement médical concis.", body_style
    ))
    
    prompt_spec = """
    <b>Schéma JSON Imposé (Bloc 5 & 7) :</b>
    <font face="Courier" size="7.5">
    {<br/>
    &nbsp;&nbsp;"patient_summary": "Synthèse du profil patient, de la condition et du type de matériel (Standard/Non-standard)",<br/>
    &nbsp;&nbsp;"predicted_birads": "Catégorie BI-RADS attribuée (1, 2, 3, 4A, 4B, 5)",<br/>
    &nbsp;&nbsp;"confidence": "Score de confiance calibré (0.0 à 1.0)",<br/>
    &nbsp;&nbsp;"interpretation": "Raisonnement multimodal synthétique articulant vision et texte",<br/>
    &nbsp;&nbsp;"supporting_evidence": "Citations obligatoires des règles ACR [G1, G2] et cas similaires [E1, E2]",<br/>
    &nbsp;&nbsp;"visual_explanations": "Localisation des zones d'attention saillantes (Mammographie et Échographie)",<br/>
    &nbsp;&nbsp;"recommendations": "Conduite à tenir clinique concrète (Biopsie, surveillance 6 mois, dépistage annuel)",<br/>
    &nbsp;&nbsp;"limitations": "Limites de l'IA, fiabilité matériel, incertitude du modèle",<br/>
    &nbsp;&nbsp;"human_review_required": true<br/>
    }
    </font>
    """
    spec_table = Table([[Paragraph(prompt_spec, body_style)]], colWidths=[487])
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, c_card_border),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(spec_table)
    story.append(Spacer(1, 10))
    
    # ── SECTION 6 : SÉCURITÉ CLINIQUE & FEEDBACK LOOP ────────────────────────
    story.append(Paragraph("6. Agent de Sécurité Clinique & Boucle de Rétroaction (Bloc 6)", h1_style))
    story.append(Paragraph(
        "Dans un environnement médical automatisé, l'IA ne peut être déployée sans garde-fous stricts. "
        "Le Quality & Safety Agent applique 4 catégories de contrôles déterministes avant toute validation :", body_style
    ))
    
    safety_table_data = [
        [Paragraph("Règle de Sécurité", table_header), Paragraph("Mécanisme d'Audit Déterministe", table_header), Paragraph("Sanction en cas d'Échec", table_header)],
        [
            Paragraph("<b>Anti-Hallucination Médicale</b>", table_cell),
            Paragraph("Vérification que tout terme lésionnel (<i>mass, nodule, cyst, calcification, thickening</i>) cité par le LLM est présent mot pour mot dans le rapport d'origine ou les preuves.", table_cell),
            Paragraph("Erreur de groundedness ➡️ Révision demandée au LLM.", table_cell)
        ],
        [
            Paragraph("<b>Cohérence Numérique Stricte</b>", table_cell),
            Paragraph("Contrôle que toute valeur chiffrée citée (pourcentage de malignité, entropie) correspond exactement aux probabilités calculées par le ResNet-18 (tolérance ±0.6%).", table_cell),
            Paragraph("Rejet du résumé ➡️ Régénération sans invention de chiffre.", table_cell)
        ],
        [
            Paragraph("<b>Divulgation de l'Incertitude</b>", table_cell),
            Paragraph("Si l'entropie d'incertitude du modèle est ≥ 0.50, l'avertissement d'incertitude élevée doit obligatoirement figurer dans les <code>limitations</code>.", table_cell),
            Paragraph("Omission de sécurité ➡️ Injection de la réserve clinique requise.", table_cell)
        ],
        [
            Paragraph("<b>Avertissement Matériel</b>", table_cell),
            Paragraph("Pour les patientes acquises sur appareil non standard (faible résolution), une mention explicite de moindre fiabilité doit être signalée.", table_cell),
            Paragraph("Omission de sécurité ➡️ Blocage jusqu'à mention explicite.", table_cell)
        ]
    ]
    t_safety = Table(safety_table_data, colWidths=[110, 257, 120])
    t_safety.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#991B1B")), # Rouge bordeaux médical
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_safety)
    story.append(Spacer(1, 10))
    
    story.append(PageBreak())
    
    # ── SECTION 7 : EXEMPLE DE RAPPORT STRUCTURÉ GÉNÉRÉ ─────────────────────
    story.append(Paragraph("7. Résultat Concret : Exemple de Rapport Médical Structuré (Bloc 7)", h1_style))
    story.append(Paragraph(
        "Voici un compte-rendu réel extrait du fichier <code>final_medical_reports_sample.json</code> généré par la dernière cellule du notebook pour la patiente <b>P0159</b> :", body_style
    ))
    
    sample_json = """
    <b>Patient :</b> P0159 | <b>Condition :</b> Multimodale (les_deux) | <b>Matériel :</b> Non-standard (reduced resolution)<br/>
    <b>Prédiction Globale :</b> <font color="#D97706"><b>BI-RADS 4A</b></font> (Suspicion faible à modérée de malignité) | <b>Confiance :</b> 78.0%<br/>
    <hr color="#CBD5E1"/>
    <b>Synthèse & Interprétation Vision (ResNet-18) :</b><br/>
    <i>"Mammography model: normal 0.36, benign 0.51, malignant 0.14 (uncertainty entropy 0.90, high). Ultrasound model malignant probability 0.62. The two modality scores point in different directions. Non-standard acquisition device: scores are less reliable."</i><br/>
    <br/>
    <b>Explications Visuelles XAI (Grad-CAM) :</b><br/>
    • Mammographie : <i>Attention portée sur la densité parenchymateuse et l'asymétrie focale.</i><br/>
    • Échographie : <i>Attention portée sur la zone tissulaire hypoéchogène.</i><br/>
    <br/>
    <b>Preuves Cliniques Analogues Retrouvées par RAG :</b><br/>
    • <b>[E1] (Similarité 0.942) :</b> <i>"Both breasts are of scattered fibroglandular pattern (ACR b)... global asymmetry with markedly diminished size & upper central diffuse s"</i><br/>
    • <b>[E2] (Similarité 0.938) :</b> <i>"Both breasts are of scattered fibroglandular pattern (ACR b)... mild skin thickening with surgical clips"</i><br/>
    <br/>
    <b>Recommandations Cliniques Officielles ACR :</b><br/>
    1. Image-guided core needle biopsy (CNB) recommended.<br/>
    2. Multidisciplinary breast tumor board review.<br/>
    <br/>
    <b>Audit de Sécurité Qualité :</b><br/>
    • Groundedness vérifié : <b>OUI</b> | Hallucinations détectées : <b>NON</b> | Cohérence BI-RADS : <b>OUI</b> | <b>Validation humaine radiologue requise : OUI</b>
    """
    sample_table = Table([[Paragraph(sample_json, body_style)]], colWidths=[487])
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(sample_table)
    story.append(Spacer(1, 10))
    
    # ── SECTION 8 : INVENTAIRE DES OUTPUTS KAGGLE ────────────────────────────
    story.append(Paragraph("8. Inventaire des Fichiers Produits dans /kaggle/working", h1_style))
    story.append(Paragraph(
        "À l'issue de l'exécution, les artefacts suivants ont été créés et sont directement exploitables pour le déploiement n8n :", body_style
    ))
    
    files_data = [
        [Paragraph("Nom du Fichier", table_header), Paragraph("Format / Taille", table_header), Paragraph("Description & Destination n8n", table_header)],
        [Paragraph("<code>final_medical_reports_sample.json</code>", table_cell), Paragraph("JSON / ~25 KB", table_cell), Paragraph("Comptes-rendus médicaux finaux complets avec XAI et sécurité (Bloc 7)", table_cell)],
        [Paragraph("<code>acr_guidelines_index.faiss</code>", table_cell), Paragraph("Binaire FAISS", table_cell), Paragraph("Index vectoriel dense des recommandations cliniques ACR BI-RADS", table_cell)],
        [Paragraph("<code>acr_guidelines_chunks.csv</code>", table_cell), Paragraph("CSV", table_cell), Paragraph("Textes et métadonnées des règles cliniques ACR indexées", table_cell)],
        [Paragraph("<code>rag_index.faiss</code>", table_cell), Paragraph("Binaire FAISS", table_cell), Paragraph("Index vectoriel dense des 317 rapports cliniques patients BCMID", table_cell)],
        [Paragraph("<code>rag_chunks.csv</code>", table_cell), Paragraph("CSV", table_cell), Paragraph("Passages textuels des rapports utilisés pour le Case-Based Reasoning", table_cell)],
        [Paragraph("<code>dossiers_fold0.jsonl</code>", table_cell), Paragraph("JSONL / 64 cas", table_cell), Paragraph("Dossiers multimodaux combinant prédictions vision, fiches XAI et preuves", table_cell)],
        [Paragraph("<code>prompts_fold0.jsonl</code>", table_cell), Paragraph("JSONL / 188 prompts", table_cell), Paragraph("Prompts conditionnés multimodaux prêts pour l'inférence LLM", table_cell)],
        [Paragraph("<code>reports_blind_clean.csv</code>", table_cell), Paragraph("CSV / 317 cas", table_cell), Paragraph("Rapports d'origine anonymisés, caviardés et dé-doublonnés", table_cell)],
    ]
    t_files = Table(files_data, colWidths=[165, 80, 242])
    t_files.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 4.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_files)
    story.append(Spacer(1, 10))
    
    # ── SECTION 9 : CONCLUSION & VALEUR AJOUTÉE ─────────────────────────────
    story.append(Paragraph("9. Conclusion & Conformité au Projet", h1_style))
    story.append(Paragraph(
        "Le laboratoire de l'Agent RAG valide l'intégration réussie de l'intelligence artificielle générative dans un flux radiologique réaliste. "
        "En couplant la vision computationnelle (ResNet-18), l'ancrage normatif (guides ACR), l'explicabilité visuelle (Grad-CAM) "
        "et des garde-fous stricts contre les hallucinations, le système BCMID démontre sa capacité à générer des comptes-rendus "
        "médicaux décisionnels de haute fiabilité, prêts pour l'automatisation dans des workflows d'orchestration clinique tels que <b>n8n</b>.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF généré avec succès : {output_path}")

if __name__ == "__main__":
    out_dir = Path(r"c:\Users\moham\OneDrive - ESPRIT\Documents\deeplearning5eme")
    target = out_dir / "Rapport_Technique_Lab_RAG_Agent.pdf"
    build_pdf(str(target))
