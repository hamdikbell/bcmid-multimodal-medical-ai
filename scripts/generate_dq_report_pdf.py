"""
Générateur de rapport PDF professionnel pour le Lab Data & Quality Agent (Blocs 1 & 2)
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
            self.drawString(54, 800, "BCMID Multimodal Medical AI — Rapport Technique : Data & Quality Agent (Blocs 1 & 2)")
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
    
    # Couleurs du thème médical
    c_primary = colors.HexColor("#0B2545")    # Bleu marine profond
    c_secondary = colors.HexColor("#134074")  # Bleu nuit
    c_accent = colors.HexColor("#0D9488")     # Sarcelle / Teal médical
    c_dark = colors.HexColor("#1E293B")       # Gris ardoise foncé
    c_light = colors.HexColor("#F8FAFC")      # Fond clair
    c_card_border = colors.HexColor("#E2E8F0")
    c_green = colors.HexColor("#16A34A")
    c_amber = colors.HexColor("#D97706")
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=25,
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
        fontSize=13.5,
        leading=17,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=9,
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
    
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
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
    story.append(Paragraph("Dossier Technique & Clinique : Data & Quality Agent (Blocs 1 & 2)", title_style))
    story.append(Paragraph("<b>Auteur :</b> Laboratoire Data Quality & Ingestion | <b>Dataset :</b> BCMID (Breast Cancer Multimodal Integrated Dataset)", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=4, spaceAfter=14))
    
    # ── BLOC RÉSUMÉ EXÉCUTIF ────────────────────────────────────────────────
    summary_html = """
    <b>Résumé Exécutif :</b> Le notebook <code>data-quality-clean.ipynb</code> constitue le socle fondamental (Blocs 1 et 2) de tout le pipeline décisionnel.
    Face à un jeu de données multimodal réel hétérogène (2001 clichés d'imagerie, 321 rapports PDF non structurés), l'Agent Qualité opère une refonte
    complète : audit d'intégrité, normalisation des polarités inversées, suppression des bordures parasites, masquage OCR des métadonnées incrustées,
    anonymisation réglementaire des rapports textuels, neutralisation des raccourcis d'apprentissage (shortcut learning), et découpage étanche en 5 plis
    par groupe patient longitudinal (GroupKFold). Ce document synthétise les méthodes, les statistiques clés et les artefacts de sortie produits.
    """
    card_data = [[Paragraph(summary_html, callout_style)]]
    card_table = Table(card_data, colWidths=[487])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDFA")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#99F6E4")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 10))
    
    # ── SECTION 1 : VOLUMÉTRIE & INVENTAIRE DU DATASET BCMID ────────────────
    story.append(Paragraph("1. Profiling & Inventaire Initial du Dataset BCMID (Étape 0)", h1_style))
    story.append(Paragraph(
        "L'exploration exhaustive du stockage BCMID (12.11 GB) a permis d'établir l'inventaire strict des fichiers bruts "
        "avant toute opération de nettoyage :", body_style
    ))
    
    vol_data = [
        [Paragraph("Type d'Artefact", table_header), Paragraph("Extension", table_header), Paragraph("Nombre de Fichiers", table_header), Paragraph("Volume Disque", table_header), Paragraph("Usage dans le Pipeline", table_header)],
        [Paragraph("<b>Clichés d'Imagerie</b>", table_cell), Paragraph(".png", table_cell), Paragraph("<b>2 001</b>", table_cell), Paragraph("~12.0 GB", table_cell), Paragraph("Mammographies (CC, MLO) et Échographies (US)", table_cell)],
        [Paragraph("<b>Comptes-Rendus Médicaux</b>", table_cell), Paragraph(".pdf", table_cell), Paragraph("<b>321</b>", table_cell), Paragraph("~110 MB", table_cell), Paragraph("Rapports radiologiques d'examens bilingues", table_cell)],
        [Paragraph("<b>Table d'Annotations</b>", table_cell), Paragraph(".tsv", table_cell), Paragraph("<b>1</b>", table_cell), Paragraph("8.7 KB", table_cell), Paragraph("Vérité terrain (BI-RADS, Pathologie, Binaire)", table_cell)],
        [Paragraph("<b>TOTAL GLOBAL</b>", table_cell), Paragraph("Tous formats", table_cell), Paragraph("<b>2 323</b>", table_cell), Paragraph("<b>12.11 GB</b>", table_cell), Paragraph("332 patientes uniques", table_cell)],
    ]
    t_vol = Table(vol_data, colWidths=[110, 55, 95, 75, 152])
    t_vol.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, c_light]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_vol)
    story.append(Spacer(1, 10))
    
    # ── SECTION 2 : GESTION DES CLASSES & CAS PARTICULIERS BI-RADS 0 ────────
    story.append(Paragraph("2. Distribution Pathologique & Gestion Rigoureuse des 5 Cas BI-RADS 0 (Étape 1)", h1_style))
    story.append(Paragraph(
        "La table d'annotation <code>BCMID_annotations.tsv</code> indexe 332 patientes réparties en 3 classes pathologiques principales, "
        "avec un sous-groupe singulier exigeant un traitement clinique dédié :", body_style
    ))
    
    dist_data = [
        [Paragraph("Classe Pathologique", table_header), Paragraph("Label Binaire", table_header), Paragraph("Score BI-RADS Typique", table_header), Paragraph("Patientes", table_header), Paragraph("Traitement en Machine Learning", table_header)],
        [Paragraph("<b>Bénin (Benign)</b>", table_cell), Paragraph("Negative", table_cell), Paragraph("BI-RADS 2 ou 3", table_cell), Paragraph("<b>214 (64.5%)</b>", table_cell), Paragraph("Supervisé (Classe 1)", table_cell)],
        [Paragraph("<b>Malin (Malignant)</b>", table_cell), Paragraph("Positive", table_cell), Paragraph("BI-RADS 4 ou 5", table_cell), Paragraph("<b>63 (19.0%)</b>", table_cell), Paragraph("Supervisé (Classe 2 - Priorité oncologique)", table_cell)],
        [Paragraph("<b>Normal</b>", table_cell), Paragraph("Negative", table_cell), Paragraph("BI-RADS 1", table_cell), Paragraph("<b>50 (15.1%)</b>", table_cell), Paragraph("Supervisé (Classe 0)", table_cell)],
        [Paragraph("<b>BI-RADS 0 (Incomplet)</b>", table_cell), Paragraph("Non étiqueté (NaN)", table_cell), Paragraph("BI-RADS 0", table_cell), Paragraph("<b>5 (1.5%)</b>", table_cell), Paragraph("<b>Conservées !</b> Flag <code>eval_unlabeled=True</code>", table_cell)],
    ]
    t_dist = Table(dist_data, colWidths=[110, 80, 102, 65, 130])
    t_dist.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_dist)
    story.append(Spacer(1, 8))
    
    birads0_note = """
    <b>Traitement des 5 Patientes BI-RADS 0 (<code>3761y16</code>, <code>7281y18</code>, <code>1946y18</code>, <code>1265y07</code>, <code>00639y20</code>) :</b><br/>
    En milieu médical, BI-RADS 0 signifie <i>'Évaluation incomplète - Nécessite des clichés complémentaires ou une échographie ciblée'</i>.
    <b>Décision d'ingénierie :</b> Les patientes ne sont <b>PAS supprimées</b> (ce qui détruirait des données réelles précieuses). Elles sont intégrées
    avec l'attribut <code>eval_unlabeled=True</code>. Elles sont exclues de la fonction de perte supervisée (Cross-Entropy) lors de l'entraînement,
    mais conservées pour les démonstrations de l'agent RAG et les audits de sécurité clinique.
    """
    card_birads = Table([[Paragraph(birads0_note, callout_style)]], colWidths=[487])
    card_birads.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FEF3C7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#FCD34D")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(card_birads)
    story.append(Spacer(1, 10))
    
    # ── SECTION 3 : PRÉVENTION DES FUITES LONGITUDINALES ───────────────────
    story.append(Paragraph("3. Gestion du Biais Longitudinal & Découpage Étanche GroupKFold (Étapes 3 & 7)", h1_style))
    story.append(Paragraph(
        "Certaines patientes possèdent des examens répétés sur plusieurs années (ex. dépistage initial en 2011, puis suivi en 2013). "
        "Si l'examen de 2011 se trouvait dans le train et celui de 2013 dans le test, le réseau apprendrait à reconnaître la texture mammaire "
        "individuelle (Data Leakage majeur).", body_style
    ))
    
    p_leak = """
    • <b>Création de l'identifiant patient regroupé (<code>group_pseudo</code>) :</b> Regroupement systématique de tous les examens appartenant à la même patiente physique.<br/>
    • <b>Découpage par <code>GroupKFold(n_splits=5)</code> :</b> Garantie mathématique stricte qu'une patiente et tous ses clichés (quelles que soient les années) n'apparaissent que dans <b>un seul pli</b>.<br/>
    • <b>Équilibre inter-plis :</b> Répartition homogène des classes pathologiques (environ 40 Bénins, 13 Malins et 10 Normaux par pli).
    """
    story.append(Paragraph(p_leak, body_style))
    story.append(Spacer(1, 8))
    
    story.append(PageBreak())
    
    # ── SECTION 4 : DIAGNOSTIC & NORMALISATION DES IMAGES ───────────────────
    story.append(Paragraph("4. Diagnostic d'Image & Normalisation Visuelle Avancée (Étape 5)", h1_style))
    story.append(Paragraph(
        "Les 2001 images présentaient des variations extrêmes d'acquisition selon les centres et constructeurs. "
        "Quatre transformations correctives standardisées ont été déployées :", body_style
    ))
    
    diag_data = [
        [Paragraph("Anomalie Détectée", table_header), Paragraph("Description Technique", table_header), Paragraph("Correctif Appliqué dans le Lab", table_header)],
        [
            Paragraph("<b>Inversion de Polarité</b>", table_cell),
            Paragraph("Certaines mammographies sont acquises avec fond blanc (255) et parenchyme sombre, à l'inverse du standard radiologique.", table_cell),
            Paragraph("Calcul de la moyenne des 4 coins d'image. Si fond > 127 ➡️ Inversion <code>img = 255 - img</code>. 100% des images remises sur fond noir.", table_cell)
        ],
        [
            Paragraph("<b>Bordures & Artéfacts</b>", table_cell),
            Paragraph("Bandes noires, collimateurs optiques ou zones blanches parasites sur les bordures externes.", table_cell),
            Paragraph("Seuillage Otsu + détection de la plus grande composante connexe ➡️ Rognage automatique centré sur le sein (Bounding Box).", table_cell)
        ],
        [
            Paragraph("<b>Hétérogénéité de Taille</b>", table_cell),
            Paragraph("Mammographies allant de 800 à 4000 px ; Échographies de résolutions hétérogènes.", table_cell),
            Paragraph("Redimensionnement bilinéaire avec maintien du ratio d'aspect (Letterboxing) :<br/>• <b>Mammographie :</b> 1280 × 1024 px<br/>• <b>Échographie :</b> 360 × 512 px", table_cell)
        ],
        [
            Paragraph("<b>Texte Incrusté (US)</b>", table_cell),
            Paragraph("Les constructeurs d'échographes incrustent le nom de la clinique, l'âge et les paramètres sur l'image.", table_cell),
            Paragraph("Détection OCR par <b>EasyOCR</b> ➡️ Masquage par inpainting noir (valeur 0) pour empêcher le réseau de lire des métadonnées.", table_cell)
        ]
    ]
    t_diag = Table(diag_data, colWidths=[110, 187, 190])
    t_diag.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_diag)
    story.append(Spacer(1, 10))
    
    # ── SECTION 5 : SÉPARATION DES SOURCES & SHORTCUT LEARNING ──────────────
    story.append(Paragraph("5. Neutralisation des Biais d'Acquisition (Shortcut Learning) (Étape 5C & 5E)", h1_style))
    story.append(Paragraph(
        "Une découverte critique du diagnostic qualité a mis en évidence deux catégories d'appareils mammographiques :", body_style
    ))
    
    shortcut_html = """
    • <b>Source Standard (Haute Définition) :</b> Capteurs numériques récents, excellente résolution spatiale et contraste élevé.<br/>
    • <b>Source Non Standard (Basse Résolution / Numériseur secondaire) :</b> Images plus bruitées, contraste réduit.<br/>
    <br/>
    <b>Risque de Raccourci (Shortcut Learning) :</b> Si une source d'appareil était corrélée avec la gravité des cas (ex. machine d'oncologie spécialisée),
    le réseau de neurones prédirait le cancer en se basant sur le <i>bruit du capteur</i> plutôt que sur les <i>anomalies tissulaires</i>.<br/>
    <b>Solutions apportées :</b><br/>
    1. Création de la variable <code>standard_device</code> (True/False) pour tracer chaque cliché.<br/>
    2. Définition du sous-ensemble <b><code>eval_main</code> (154 patientes unilatérales sur source standard)</b> pour tester le modèle en conditions d'évaluation pures et non biaisées.<br/>
    3. Injection de sondes de signature (probe features) pour s'assurer que la décision finale repose sur la biologie et non l'appareil.
    """
    card_sc = Table([[Paragraph(shortcut_html, body_style)]], colWidths=[487])
    card_sc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, c_card_border),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(card_sc)
    story.append(Spacer(1, 10))
    
    # ── SECTION 6 : ANONYMISATION DES RAPPORTS MÉDICAUX ─────────────────────
    story.append(Paragraph("6. Anonymisation Stricte des Rapports PDF (Étape 6)", h1_style))
    story.append(Paragraph(
        "Les 321 rapports médicaux PDF ont fait l'objet d'une extraction et d'un caviardage complet conforme aux normes HIPAA/RGPD :", body_style
    ))
    
    anon_p = """
    • <b>Suppression des Identifiants Directs :</b> Noms de patientes, numéros de dossier médical, noms des médecins référents et des radiologues remplacés par <code>[ID]</code>, <code>[DOCTOR]</code>.<br/>
    • <b>Masquage Temporel :</b> Dates exactes remplacées par <code>[DATE]</code>.<br/>
    • <b>Génération de deux versions par rapport :</b><br/>
    &nbsp;&nbsp;&nbsp;&nbsp;1. <code>text_clean</code> : Rapport complet anonymisé sans données d'identification.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;2. <code>text_blind</code> : Rapport sans aucune conclusion chiffrée BI-RADS ni mention directe de cancer, servant de base saine pour le RAG.
    """
    story.append(Paragraph(anon_p, body_style))
    story.append(Spacer(1, 8))
    
    story.append(PageBreak())
    
    # ── SECTION 7 : INVENTAIRE DES SORTIES GÉNÉRÉES ──────────────────────────
    story.append(Paragraph("7. Inventaire des Fichiers Clés Produits dans /kaggle/working", h1_style))
    story.append(Paragraph(
        "L'exécution complète de <code>data-quality-clean.ipynb</code> a produit les artefacts de référence réutilisés par les notebooks suivants :", body_style
    ))
    
    out_table_data = [
        [Paragraph("Artefact Généré", table_header), Paragraph("Format & Taille", table_header), Paragraph("Rôle dans le Pipeline Aval", table_header)],
        [
            Paragraph("<b><code>dataset_final.csv</code></b>", table_cell),
            Paragraph("CSV / ~35 KB", table_cell),
            Paragraph("<b>Fichier maître indispensable</b> : Indexe les 332 patientes, leurs identifiants anonymes (<code>pseudo_id</code>), leur groupe longitudinal, leurs labels, leurs plis 0-4 et les flags <code>eval_main</code> et <code>eval_unlabeled</code>.", table_cell)
        ],
        [
            Paragraph("<b><code>images_anon.csv</code></b>", table_cell),
            Paragraph("CSV / ~240 KB", table_cell),
            Paragraph("Table de correspondance des 1 974 clichés d'imagerie nettoyés, avec vue radiologique (CC/MLO), modalité (Mammo/US) et chemin physique.", table_cell)
        ],
        [
            Paragraph("<b><code>reports_anon/</code></b>", table_cell),
            Paragraph("Dossier / 321 fichiers JSON", table_cell),
            Paragraph("Rapports médicaux anonymisés prêts pour l'ingestion par le modèle d'embedding du Clinical RAG (Bloc 4).", table_cell)
        ],
        [
            Paragraph("<b><code>processed/</code></b>", table_cell),
            Paragraph("Dossier PNG / ~4 GB", table_cell),
            Paragraph("Dossier contenant les sous-dossiers <code>mammo/</code> (951 images) et <code>us/</code> (1023 images) normalisées et prêtes pour l'entraînement PyTorch.", table_cell)
        ],
        [
            Paragraph("<b><code>quality_report.json</code></b>", table_cell),
            Paragraph("JSON / ~10 KB", table_cell),
            Paragraph("Rapport d'audit qualité exhaustif résumant toutes les métriques de conformité pour traçabilité réglementaire.", table_cell)
        ],
    ]
    t_out = Table(out_table_data, colWidths=[140, 95, 252])
    t_out.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_out)
    story.append(Spacer(1, 12))
    
    # ── SECTION 8 : SYNTHÈSE & CONFORMITÉ DU LAB ────────────────────────────
    story.append(Paragraph("8. Bilan Qualité & Conformité Médicale", h1_style))
    story.append(Paragraph(
        "Le laboratoire Data & Quality garantit la solidité scientifique et éthique de l'ensemble du projet :", body_style
    ))
    
    checklist_data = [
        [Paragraph("Exigence du Projet", table_header), Paragraph("Statut", table_header), Paragraph("Résultat Obtenu", table_header)],
        [Paragraph("Intégrité des Métadonnées TSV", table_cell), Paragraph("<font color='#16A34A'><b>100% CONFORME</b></font>", table_cell), Paragraph("332 patientes vérifiées, aucune ligne corrompue", table_cell)],
        [Paragraph("Préservation des cas BI-RADS 0", table_cell), Paragraph("<font color='#16A34A'><b>100% CONFORME</b></font>", table_cell), Paragraph("5 cas conservés avec flag <code>eval_unlabeled</code>", table_cell)],
        [Paragraph("Absence de fuite patient inter-plis", table_cell), Paragraph("<font color='#16A34A'><b>100% CONFORME</b></font>", table_cell), Paragraph("GroupKFold strict sur <code>group_pseudo</code>", table_cell)],
        [Paragraph("Normalisation des polarités images", table_cell), Paragraph("<font color='#16A34A'><b>100% CONFORME</b></font>", table_cell), Paragraph("100% des fonds blancs inversés en fond noir standard", table_cell)],
        [Paragraph("Suppression métadonnées incrustées", table_cell), Paragraph("<font color='#16A34A'><b>100% CONFORME</b></font>", table_cell), Paragraph("EasyOCR + inpainting noir sur échographies", table_cell)],
        [Paragraph("Anonymisation des rapports cliniques", table_cell), Paragraph("<font color='#16A34A'><b>100% CONFORME</b></font>", table_cell), Paragraph("321 rapports caviardés selon normes médicales", table_cell)],
    ]
    t_check = Table(checklist_data, colWidths=[170, 110, 207])
    t_check.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('PADDING', (0,0), (-1,-1), 4.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_check)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(
        "<b>Conclusion :</b> Grâce à ce travail minutieux de préparation, le Vision Agent (Bloc 3) et le RAG Agent (Blocs 4-7) "
        "ont pu s'entraîner et raisonner sur des données rigoureusement étanches, dé-bruitées et standardisées, "
        "garantissant des performances prédictives fiables et transposables en pratique clinique.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Data Quality généré avec succès : {output_path}")

if __name__ == "__main__":
    out_dir = Path(r"c:\Users\moham\OneDrive - ESPRIT\Documents\deeplearning5eme")
    target = out_dir / "Rapport_Technique_Lab_Data_Quality.pdf"
    build_pdf(str(target))
