# 🩺 BCMID — Diagnostic Multimodal du Cancer du Sein par IA

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![n8n](https://img.shields.io/badge/n8n-Orchestration-FF6D5A.svg?logo=n8n)](https://n8n.io)
[![Académique](https://img.shields.io/badge/Projet-Académique_ESPRIT-blue.svg)](https://esprit.tn)

> **Projet académique — Advanced Deep Learning, 5ᵉ année d'ingénierie, ESPRIT (2025–2026)**

> **Sujet :** Système d'automatisation d'IA médicale multimodale avec IA explicable (XAI), RAG clinique, raisonnement LLM et orchestration hospitalière pour le triage du cancer du sein.

---

## 👥 Équipe

Ce projet est un **travail d'équipe** réalisé par :

- **Ikbel Hamdi**
- **Mohamed Malek Hammami**
- **Samar Omrani**
- **Ahmed Fekih**

**Encadrante :** Oumayma Guesmi
**Établissement :** ESPRIT — École Supérieure Privée d'Ingénierie et de Technologies, Tunis, Tunisie
**Année universitaire :** 2025–2026

---

## 📑 Sommaire

1. [Contexte et problématique](#-contexte-et-problématique)
2. [Objectifs](#-objectifs)
3. [Architecture du système](#-architecture-du-système)
4. [Jeu de données et méthodologie](#-jeu-de-données-et-méthodologie)
5. [Résultats](#-résultats)
6. [Structure du dépôt](#-structure-du-dépôt)
7. [Installation et exécution](#-installation-et-exécution)
8. [Cas de test de référence](#-cas-de-test-de-référence)
9. [Tests](#-tests)
10. [Livrables et documentation](#-livrables-et-documentation)
11. [Éthique, confidentialité et avertissement](#-éthique-confidentialité-et-avertissement)

---

## 📌 Contexte et problématique

La mammographie de dépistage réduit la mortalité du cancer du sein, mais utilisée seule elle présente deux limites : une **sensibilité réduite sur les seins denses** et un **taux élevé de faux positifs** conduisant à des biopsies inutiles. Associer la **mammographie** (architecture tissulaire, microcalcifications) à l'**échographie** (morphologie, atténuation acoustique) permet une caractérisation plus fiable des lésions.

Les modèles d'IA « boîte noire » sont toutefois peu adoptés en clinique : absence de preuve visuelle, risque d'hallucination des modèles de langage et déconnexion du flux hospitalier. Notre projet répond à ces trois verrous par une **architecture multi-agents** traçable.

## 🎯 Objectifs

- Concevoir un pipeline **multimodal** (mammographie + échographie) de classification ACR BI-RADS.
- Rendre les prédictions **explicables** (cartes d'activation Grad-CAM).
- Ancrer les recommandations dans les **lignes directrices ACR BI-RADS** via un moteur **RAG**.
- Générer un rapport clinique par un **agent LLM** contrôlé par un **agent de sécurité déterministe**.
- **Orchestrer** le triage hospitalier (urgent / surveillance / routine) avec **n8n**.
- Garantir la **confidentialité** (pseudonymisation) et l'**absence de fuite de données** entre jeux d'entraînement et de test.

## 🏗️ Architecture du système

```
   [ Ingestion patient ] ──────► [ Bloc 1 : Agent Données & Qualité ]
 (Mammographie + Échographie)       ├── Pseudonymisation SHA-256 salée
                                    ├── CLAHE (mammo) & filtre médian (écho), 512×512
                                    └── Contrôle d'intégrité et de résolution
                                                │
                                                ▼
                                 [ Blocs 2-3 : Agent Vision multimodal ]
                                    ├── Double branche ResNet-18
                                    ├── Couche de fusion → catégorie BI-RADS
                                    └── Cartes Grad-CAM (XAI)
                                                │
                                                ▼
                                 [ Bloc 4 : Moteur RAG clinique ]
                                    ├── Lignes directrices ACR BI-RADS (5ᵉ éd.)
                                    └── Recherche vectorielle FAISS
                                                │
                                                ▼
                                 [ Blocs 5-6 : Raisonnement & Sécurité ]
                                    ├── Synthèse clinique LLM (Qwen2.5-3B)
                                    ├── Vérifications anti-hallucination déterministes
                                    └── Validation humaine obligatoire
                                                │
                                                ▼
                                 [ Bloc 7 : Rapport médical ]
                                    ├── Rapport HTML interactif
                                    └── Dossier JSON structuré
                                                │
                                                ▼
                                 [ Bloc 8 : Orchestration n8n ]
                                    ├── URGENT        ➔ alerte biopsie + comité (RCP)
                                    ├── SURVEILLANCE  ➔ suivi à 6 mois
                                    └── ROUTINE       ➔ dépistage annuel
                                                │
                                                ▼
                                 [ Notification e-mail au radiologue ]
```

**Justification multi-agents :** la séparation perception (Vision) / mémoire (RAG) / raisonnement (LLM) / contrôle (Safety) assure la traçabilité de chaque décision et limite le risque d'hallucination.

**Choix techniques :** Python · FastAPI · PyTorch (ResNet-18, transfert d'apprentissage) · Grad-CAM · FAISS + sentence-transformers · Qwen2.5-3B · ReportLab · n8n.

## 📊 Jeu de données et méthodologie

- **Cohorte :** 332 patientes (948 mammographies, 1 049 échographies), annotations dans `BCMID_annotations.tsv`.
- **Séparation au niveau patient (zéro fuite) :**

| Jeu | Part | Patientes |
| :--- | :---: | :---: |
| Entraînement | 70 % | 228 |
| Validation | 15 % | 47 |
| Test (hold-out) | 15 % | 57 |

- **Garantie :** `leakage = 0` — aucune patiente n'apparaît dans plusieurs jeux.
- **Prétraitement :** CLAHE + filtrage (mammographie), filtre médian (échographie), redimensionnement 512×512 (voir `config.yaml`).

## 📈 Résultats

Évaluation de la vision sur la cohorte de fusion (323 patientes) :

| Modalité | AUC | Sensibilité |
| :--- | :---: | :---: |
| Mammographie (ResNet-18) | 0,7917 | 76,2 % |
| Échographie (ResNet-18) | 0,6743 | 68,4 % |
| **Fusion multimodale** | **0,8198** | **83,1 %** |

La fusion surpasse chaque modalité isolée (ΔAUC = +0,0281). L'audit Grad-CAM indique que 87,4 % de l'énergie d'activation reste dans le parenchyme mammaire. Détails complets dans les rapports techniques (section [Livrables](#-livrables-et-documentation)).

## 🗂️ Structure du dépôt

```
├── api_server.py                  # API FastAPI (endpoints des agents + triage)
├── n8n_bcmid_workflow.json        # Workflow d'orchestration n8n
├── config.yaml                    # Configuration globale du pipeline
├── requirements.txt               # Dépendances
├── BCMID_annotations.tsv          # Métadonnées cliniques et étiquettes
│
├── dq_agent/                      # Bloc 1 : agent Données & Qualité
│   ├── anonymize.py               #   pseudonymisation SHA-256 salée
│   ├── integrity.py               #   détection d'images corrompues / doublons
│   ├── preprocess.py              #   CLAHE & filtre médian
│   └── splits.py                  #   découpage stratifié au niveau patient
│
├── src/                           # Agents modulaires
│   ├── ingestion/                 #   agent d'ingestion
│   ├── vision/                    #   agent de vision (ResNet-18 + Grad-CAM)
│   └── reasoning/                 #   agent de raisonnement LLM
│
├── ikbelwork/                     # Agents RAG, sécurité et rapport
│   ├── clinical_rag_enhancer.py   #   Bloc 4 : RAG ACR BI-RADS
│   ├── safety_agent.py            #   Bloc 6 : agent de sécurité
│   ├── medical_report_generator.py#   Bloc 7 : génération du rapport
│   ├── run_orchestrator.py        #   exécution de bout en bout
│   └── acr_birads_guidelines.json #   base de connaissances
│
├── bcmid_work/                    # Données de travail, index FAISS, rapports générés
├── sample_data/                   # 3 cas de référence (jeu de test)
├── scripts/                       # Génération des rapports PDF, réconciliation
├── docs/                          # Rapports techniques de laboratoire (PDF)
├── tests/                         # Tests unitaires et d'intégration
├── *.ipynb                        # Notebooks d'expérimentation
└── BCMID_*.md                     # Rapports techniques (FR / EN)
```

## ⚡ Installation et exécution

### 1. Installation

```bash
git clone https://github.com/hamdikbell/bcmid-multimodal-medical-ai.git
cd bcmid-multimodal-medical-ai

python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Copier `.env.example` vers `.env` et renseigner `DQ_SALT` ainsi que les paramètres SMTP (envoi d'e-mails). **Ne jamais versionner le fichier `.env`.**

> Les chemins dans `config.yaml` sont propres à une machine : à adapter avant d'exécuter l'agent Données & Qualité.

### 2. Lancer l'API

```bash
python api_server.py
```

- Documentation Swagger : <http://localhost:8000/docs>
- Console clinique web : <http://localhost:8000/form>

### 3. Importer le workflow n8n

1. Lancer n8n : `npx n8n`, puis ouvrir <http://localhost:5678>.
2. Menu `...` → **Import from File** → `n8n_bcmid_workflow.json`.
3. Cliquer sur **Publish** pour activer le webhook.

## 🧪 Cas de test de référence

| Cas | Patiente | Diagnostic réel | BI-RADS prédit | Priorité n8n | Action recommandée |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Maligne | `P1eb742860c` | Maligne | **5** (94 %) | `URGENT` | Biopsie sous guidage |
| Surveillance | `Pf9bea655cf` | Bénigne | **3** (78 %) | `SURVEILLANCE` | Suivi à 6 mois |
| Normale | `Paacb1ec57b` | Normale | **1** (94 %) | `ROUTINE` | Dépistage annuel |

## ✅ Tests

```bash
pytest tests/ -v
```

Les tests couvrent l'anonymisation, l'intégrité des images, le prétraitement, les découpages sans fuite, la complétude bimodale, l'agent de vision, l'agent d'ingestion et le pipeline de sécurité.

## 📚 Livrables et documentation

| Document | Contenu |
| :--- | :--- |
| `BCMID_Rapport_Technique_FR.md` | Rapport technique principal (français) |
| `BCMID_Final_Technical_Report_EN.md` | Rapport technique (anglais) |
| `BCMID_Rapport_Fine_Tuning.md` | Choix de fine-tuning |
| `BCMID_Rapport_Agents_FineTuning.md` | Agents et stratégie de fine-tuning |
| `docs/Rapport_Technique_Lab_Data_Quality.pdf` | Rapport de laboratoire — Données & Qualité |
| `docs/Rapport_Technique_Lab_RAG_Agent.pdf` | Rapport de laboratoire — Agent RAG |
| `bcmid_work/reports/` | Rapports PDF et rapports patients générés |

## 🛡️ Éthique, confidentialité et avertissement

- **Pseudonymisation** : identifiants remplacés par un hachage SHA-256 salé ; filtrage textuel des données nominatives.
- **Validation humaine obligatoire** : chaque rapport exige la signature d'un radiologue.
- **Avertissement médical** : ce logiciel est développé **exclusivement à des fins académiques et de recherche**. Il n'est pas certifié pour un usage diagnostique autonome ; toute sortie doit être revue par un radiologue qualifié.

## 🙏 Remerciements

Nous remercions notre encadrante, Mme Oumayma Guesmi, ainsi que l'équipe pédagogique d'ESPRIT pour leur accompagnement, et les auteurs du jeu de données BCMID et des lignes directrices ACR BI-RADS.

---

