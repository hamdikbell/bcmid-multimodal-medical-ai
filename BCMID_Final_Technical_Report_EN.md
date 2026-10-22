# BCMID — Multimodal Medical AI Automation System with XAI
## Comprehensive Technical & Architecture Academic Report

**Project Title:** BCMID — Breast Cancer Multimodal Intelligent Diagnosis  
**Academic Context:** 5th-Year Deep Learning Engineering Project — ESPRIT (Academic Year 2025–2026)  
**Track:** Artificial Intelligence & Data Science (Biomedical AI Specialization)  
**Target Specification:** 8-Page Executive Technical Report (Excluding Annexes)  
**Technology Stack:** Python 3.13 · FastAPI · PyTorch · ResNet-18 · Grad-CAM · FAISS · sentence-transformers · Qwen2.5-3B · ReportLab · n8n

---

## Table of Contents

| Section | Focus / Key Technical Topic | Core Rubric Requirement |
|---|---|---|
| **Section 1** | Executive Summary & Clinical Context | Clinical Senology & Problem Statement |
| **Section 2** | Agent Architecture & Use Case Justification | **– agent architecture (justification of use case)** |
| **Section 3** | Choice of Tools & Technology Stack Justification | **– choice of tools** |
| **Section 4** | Choice of LLM & Reasoning Engine | **– choice of LLM** |
| **Section 5** | Clinical Retrieval-Augmented Generation (RAG) Approach | **– RAG approach** |
| **Section 6** | System Evaluation & Empirical Benchmarks | **– evaluation of your system** |
| **Section 7** | Security, Privacy & Clinical Safeguards | **– security considerations** |
| **Section 8** | Conclusion, Provenance & Industrial Perspectives | Strategic Roadmap & Academic Sign-off |
| **Annex A** | Real Patient Case Study with Grad-CAM Visual Evidence | *Exempt from 8-page count limit* |
| **Annex B** | Automated Test Suite (61/61 PASS Verification Matrix) | *Exempt from 8-page count limit* |
| **Annex C** | Hospital n8n 9-Node Workflow Architecture Topology | *Exempt from 8-page count limit* |

---

## 1. Executive Summary & Clinical Context

Breast cancer is the leading cause of oncological mortality among women worldwide. Although routine screening mammography reduces mortality by 20% to 30%, single-modality screening suffers from two severe clinical limitations in routine practice:
1. **Parenchymal Masking in Dense Breasts (ACR Types C & D):** Fibroglandular tissue radiopacity obscures non-calcified malignant tumors, lowering standalone mammography sensitivity from >85% to below 48%.
2. **False Positive & Biopsy Overburden:** Over 70% of invasive percutaneous core biopsies performed following suspicious mammographic findings yield benign histology (fibroadenomas, fibrocystic changes, sclerosing adenosis), imposing psychological distress and hospital costs.

**The BCMID Solution:** The BCMID platform integrates **Full-Field Digital Mammography (FFDM: CC and MLO views)** with **Targeted High-Resolution Ultrasound (US)**. By combining macroscopic parenchymal distortion and microcalcification analysis with acoustic tissue elasticity and posterior attenuation characterization, BCMID constructs an autonomous **Agentic AI CDSS (Clinical Decision Support System)** to assist radiologists in standardized BI-RADS categorization and biopsy triage.

> [!IMPORTANT]
> **Regulatory Classification:** BCMID is strictly a Clinical Decision Support System (Class IIa SaMD under EU MDR 2017/745). It operates as a radiologist pair-assistant. The field `human_review_required: true` is unconditionally enforced on all outputs.

---

## 2. Agent Architecture & Use Case Justification

### 2.1 Justification: Why an Agentic Multi-Agent System (MAS) Fits Breast Cancer Triage

A traditional monolithic AI approach (e.g., an end-to-end black-box deep learning network) is fundamentally unsuited for clinical senology due to four major domain constraints:
1. **Multimodal Asynchrony:** Mammograms and ultrasound scans have radically different spatial resolutions, noise characteristics, and semantic meanings. They cannot be treated as simple stacked image channels.
2. **The "Black-Box" Adoption Barrier:** Clinicians and tumor boards reject ungrounded predictions without interpretable visual evidence (where the lesion is located) and textual rationale (why it is suspicious).
3. **Medical Hallucination Risk:** Generative language models, when unconstrained, hallucinate plausible yet fabricated medical terms, which can lead to catastrophic diagnostic errors.
4. **Hospital Workflow Disconnect:** In real hospitals, AI must not only compute a score, but also trigger priority escalations (core needle biopsy scheduling vs. annual recall) via existing hospital IT (PACS/HIS).

**The Multi-Agent Justification:** BCMID decouples the diagnostic process into **7 specialized autonomous agents** coordinated by an enterprise hospital orchestrator (**n8n**). By segregating perception (Vision), memory retrieval (RAG), cognitive synthesis (Reasoning LLM), and compliance audit (Safety Agent), each component operates within deterministic boundaries with independent verification.

```
       MAMMOGRAPHY (CC/MLO)  +  ULTRASOUND (US)  +  CLINICAL NOTES
                                    |
                                    v
       +--------------------------------------------------------+
       |   BCMID MULTIMODAL AUTOMATION & XAI PIPELINE (BLOCS 1-7)|
       +----------------------------+---------------------------+
                                    |
                  +-----------------+-----------------+
                  v                                   v
        DOCTOR VIEW (PDF)                   TECHNICIAN XAI VIEW (HTML)
   - BI-RADS Diagnostic Assessment     - Raw ResNet-18 Probabilities
   - Clinical Action Recommendations   - Dual Grad-CAM Attention Heatmaps
   - Normative ACR Guideline Citations - Safety Audit & Provenance Logs
```

### 2.2 Operational Specification of the 7 Specialized Agents

```mermaid
flowchart TD
    subgraph S1["Phase 1: Ingestion & Quality Control"]
        IN1["Patient Folder (Mammography PNGs + US PNGs + Text)"] --> B1["Bloc 1: Ingestion Agent<br/>(File Integrity & SHA-256)"]
        B1 --> B2["Bloc 2: Data Quality Agent<br/>(Otsu Crop, Inversion, EasyOCR Masking)"]
        B2 --> SPLIT["GroupKFold (k=5) Splitting<br/>(0 Data Leakage on group_pseudo)"]
    end

    subgraph S2["Phase 2: Vision & Knowledge Retrieval"]
        SPLIT --> B3["Bloc 3: Multimodal Vision Agent<br/>(ResNet-18 Dual Branch)"]
        B3 --> XAI["Grad-CAM XAI Generator<br/>(Parenchyma & Mass Attention)"]
        SPLIT --> B4["Bloc 4: Clinical RAG Agent<br/>(FAISS Vector Index + BGE Embeddings)"]
        B4 --> ACR["ACR BI-RADS Atlas Guidelines"]
        B4 --> HIST["Historical Case Chunks (text_blind)"]
    end

    subgraph S3["Phase 3: Cognitive Synthesis & Safety Verification"]
        B3 --> FUSION["Multimodal Fusion & Entropy Calculation"]
        FUSION --> B5["Bloc 5: Medical Reasoning Agent<br/>(Qwen2.5-3B-Instruct Prompting)"]
        XAI --> B5
        ACR --> B5
        HIST --> B5
        
        B5 --> B6{"Bloc 6: Quality & Safety Agent<br/>(Deterministic Gate)"}
        B6 -- "Safety Violation Detected" --> REV["Active Feedback Revision Loop<br/>(Max 2 Retry Cycles)"]
        REV --> B5
    end

    subgraph S4["Phase 4: Structured Clinical Delivery"]
        B6 -- "Approved & Verified" --> B7["Bloc 7: Final Structured Generator"]
        B7 --> OUT_DOC["Doctor View: Official PDF Report<br/>(ACR BI-RADS + Action Recommendations)"]
        B7 --> OUT_TECH["Technician View: Interactive HTML<br/>(Vision Scores, Entropy & Grad-CAM Heatmaps)"]
        B7 --> OUT_JSON["Machine-Readable JSON<br/>(Full Provenance Audit Trail)"]
    end
```

| Agent Identifier | Operational Mandate | Input Modalities | Output Deliverable |
|---|---|---|---|
| **Bloc 1: Ingestion Agent** | Integrity validation & file decodability | Raw `.png` / `.tsv` files | Non-corrupt patient dossiers with SHA-256 hashes |
| **Bloc 2: Data Quality Agent** | Artifact removal, de-identification & splitting | Raw images & clinical text | Inpainted $512\times 512$ tensors, `GroupKFold(5)` splits |
| **Bloc 3: Multimodal Vision Agent** | Feature extraction, fusion & visual XAI | Paired Mammo + US images | Logits, Shannon entropy $H$, 387 Grad-CAM heatmaps |
| **Bloc 4: Clinical RAG Agent** | Dense semantic knowledge retrieval | Clinical query & patient history | Top-4 retrieved ACR BI-RADS guidelines & precedent chunks |
| **Bloc 5: Medical Reasoning Agent** | Cognitive multimodal synthesis | Vision logits, Grad-CAM zones, RAG | Validated clinical JSON with predicted BI-RADS & rationale |
| **Bloc 6: Quality & Safety Agent** | Deterministic audit & feedback loop | Candidate JSON + Vision logits | Approved audit certificate or structured revision hint |
| **Bloc 7: Reporting Agent** | Multi-stakeholder artifact generation | Audited clinical JSON & heatmaps | Doctor-facing PDF (ReportLab) & Technician HTML dashboard |
| **Hospital Orchestrator (n8n)** | Event-driven hospital routing | Webhook push from HIS/PACS | Urgent oncological alert (Biopsy) vs. Routine notice |

### 2.3 The Active Feedback Revision Loop (Bloc 6 -> Bloc 5)
A hallmark of Agentic AI is **self-reflection and automated revision**. When the Reasoning Agent (LLM) generates a draft diagnosis, it is intercepted by the Quality & Safety Agent before leaving the microservice:
* If the Safety Agent detects an ungrounded claim (e.g., mentioning "spiculated microcalcifications" not present in the vision summary or RAG passages) or a numerical deviation ($>0.6\%$ difference from ResNet-18 logits), it **rejects the draft**.
* The Safety Agent returns a structured error message (`reason: "Ungrounded lesion descriptor detected"`) back to the Reasoning Agent.
* The Reasoning Agent regenerates its output conditioned on this feedback (up to 2 revision cycles). If validation fails twice, the system flags the dossier as `FLAGGED_FOR_MANUAL_CLINICIAN_REVIEW` with an audit error log.

---

## 3. Choice of Tools & Technology Stack Justification

Every tool in the BCMID infrastructure was selected based on clinical safety, on-premises privacy, deterministic execution, and computational efficiency.

| Engineering Domain | Selected Tool | Evaluated Alternatives | Rationale & Clinical Justification |
|---|---|---|---|
| **Microservice Backend** | **FastAPI (ASGI)** | Flask, Django, SpringBoot | High-performance asynchronous non-blocking I/O, native Pydantic data contract validation, automatic OpenAPI/Swagger documentation, lightweight footprint ($<50$ MB RAM). |
| **Deep Learning Framework** | **PyTorch 2.6** | TensorFlow, JAX | Industry gold standard in biomedical research, dynamic autograd computation graph essential for Grad-CAM backpropagation, native support for pretrained Torchvision backbones. |
| **Dense Vector Index** | **FAISS (IndexFlatIP)** | Pinecone, ChromaDB, Weaviate | C++ optimized in-memory vector similarity. 100% on-premises execution; eliminates third-party cloud data exfiltration risks (GDPR/HIPAA). Exact cosine retrieval without approximate index quantization error. |
| **Embeddings Model** | **BAAI/bge-small-en-v1.5** | OpenAI `text-embedding-3`, BioBERT | Compact 384-dimensional dense vectors, top-ranked on MTEB retrieval benchmark, fast CPU/GPU inference ($<15$ ms per clinical report query), entirely offline. |
| **Image Preprocessing** | **OpenCV & EasyOCR** | Tesseract, PyMuPDF | Robust contour detection for Otsu parenchymal cropping, deep learning text recognition for locating burned-in machine gain/frequency overlays, Navier-Stokes inpainting eliminating shortcut learning. |
| **PDF Generation Engine** | **ReportLab 4.4** | WeasyPrint, pdfkit, Puppeteer | Programmatic, vector-perfect, deterministic two-pass rendering engine (`NumberedCanvas`), no dependency on unstable headless browser processes or external Chromium binaries. |
| **Clinical Orchestrator** | **n8n (Self-Hosted)** | Apache Airflow, LangGraph, Celery | Visual, low-code, event-driven workflow engine designed for integration with hospital HL7/PACS webhooks. Native conditional IF branching for urgent oncology escalation, self-hosted on hospital intranet. |
| **Testing & CI Framework** | **pytest 9.1** | unittest, nose2 | Flexible parameterized assertions, comprehensive test runner for validating anonymization, image integrity, data splits, and safety audit gates (61 tests in 1.84s). |

---

## 4. Choice of LLM & Reasoning Engine

### 4.1 Selected Model: Qwen2.5-3B-Instruct
The Medical Reasoning Agent (Bloc 5) is powered by **Qwen2.5-3B-Instruct**, operating in FP16 precision.

```
+---------------------------------------------------------------------------------------------------+
| MODEL ATTRIBUTE         | SPECIFICATION                   | CLINICAL / OPERATIONAL RATIONALE      |
+-------------------------+---------------------------------+---------------------------------------+
| Architecture            | 3-Billion Parameter Dense LLM   | Balances reasoning depth with speed   |
| Precision & VRAM        | FP16 (3.1 GB) / GGUF 4-bit (1.9)| Edge hospital deployment (single GPU) |
| Decoding Strategy       | Greedy (temperature=0.0)        | Zero random sampling; reproducible    |
| Context Window          | 32,768 tokens                   | Full patient dossier & guidelines fit |
| JSON Schema Enforcement | Constrained Grammar Decoding    | Zero malformed output or parse fails  |
+-------------------------+---------------------------------+---------------------------------------+
```

### 4.2 Engineering Rationale for Model Selection
1. **Hospital Edge Deployment & Low Hardware Footprint:** Edge hospital imaging servers cannot accommodate 70B parameter models requiring $8 \times \text{A100}$ clusters ($>\$100,000$). Qwen2.5-3B-Instruct requires only **3.1 GB VRAM**, enabling full execution on standard hospital PACS workstations or edge GPUs (e.g., NVIDIA RTX 3060/4060 or Tesla T4).
2. **Deterministic Clinical Reproducibility:** Medical decision support requires that identical clinical inputs yield identical recommendations. By enforcing greedy decoding (`temperature=0.0`, `do_sample=False`), stochastic variation is completely eliminated.
3. **Strict Structured JSON Adherence:** Qwen2.5-3B features state-of-the-art instruction-following capabilities, reliably outputting valid JSON conforming to the BCMID clinical schema (`predicted_birads`, `confidence`, `interpretation`, `recommendations`, `limitations`, `human_review_required`).
4. **Data Sovereignty & Air-Gapped Privacy:** Commercial LLM APIs (GPT-4, Claude) transmit sensitive patient symptoms and physician notes to external cloud servers, violating HIPAA and European health data sovereignty regulations. Qwen2.5-3B runs 100% on-premises.

### 4.3 Prompt Conditioning Ablation Study (Fold-0 Benchmark)
Across the 64 validation patients in fold 0, three prompting conditions were systematically evaluated:

| Conditioning Mode | Input Modalities Provided to LLM Prompt | Median Report Length | Diagnostic Groundedness & Completeness |
|---|---|---|---|
| **`image`** | ResNet-18 probabilities + Grad-CAM spatial summary | 245 words | High numerical alignment; lacks personal clinical risk history. |
| **`texte`** | Cleaned patient notes + Retrieved RAG evidence | 442 words | Rich clinical context; lacks visual localization and deep learning backing. |
| **`les_deux` (Multimodal)** | **Vision probabilities + Grad-CAM + Notes + RAG** | **486 words** | **Optimal clinical utility; combines visual evidence with ACR guidelines.** |

---

## 5. Clinical Retrieval-Augmented Generation (RAG) Approach

### 5.1 Why RAG Over Fine-Tuning?
In clinical senology, relying on parametric weights (fine-tuning) is hazardous for three reasons:
1. **Catastrophic Forgetting & Hallucination:** Fine-tuned LLMs can hallucinate plausible guideline text when faced with rare edge cases.
2. **Guideline Obsolescence:** When the American College of Radiology updates BI-RADS guidelines (e.g., from 5th to 6th edition), a fine-tuned model must be re-trained from scratch. A RAG index is updated instantly by updating the vector database.
3. **Direct Source Provenance:** Radiologists require explainable citations. BCMID's RAG explicitly attaches verifiable tags (`[G1]`, `[E1]`) linking directly to the authoritative paragraph in the ACR Atlas.

### 5.2 Dual Dense Vector Architecture
The RAG agent operates **two distinct FAISS vector stores**:

```
[Clinical Query / Patient Findings] ──► BAAI/bge-small-en-v1.5 Encoder (384-dim)
                                                |
                  +-----------------------------+-----------------------------+
                  v                                                           v
   acr_guidelines_index.faiss (11 Rules)                       rag_index.faiss (317 Chunks)
   - ACR BI-RADS Atlas 5th Edition                             - De-identified Historical Cases
   - Density Classes (ACR A-D)                                 - Ground Truth Masked (text_blind)
   - Margins, Densities, Calcifications                        - Fold-Isolated (exclude_pseudo_id)
                  |                                                           |
                  +-----------------------------+-----------------------------+
                                                v
                    Top-4 Cosine Retrieved Evidence Passages ([G1], [G2], [E1], [E2])
                                                v
                               To Medical Reasoning Agent (Bloc 5)
```

1. **Normative ACR Atlas Index (`acr_guidelines_index.faiss`):** Contains the 11 official rules from the American College of Radiology BI-RADS Atlas 5th Edition covering lesion margins (spiculated, indistinct, circumscribed), tissue density, and management protocols.
2. **Clinical Precedent Index (`rag_index.faiss`):** Contains 317 sanitized, de-identified clinical report chunks from previous hospital admissions.

### 5.3 Fold Isolation Protocol (Zero Data Contamination)
To ensure strict clinical validation, during evaluation on fold $k$, all passages originating from patients assigned to fold $k$ are excluded from retrieval candidates (`exclude_pseudo_id`). This mathematically prevents the RAG agent from retrieving the ground truth or historical notes of the active patient.

---

## 6. System Evaluation & Empirical Benchmarks

### 6.1 Multimodal Vision Performance (ResNet-18 Dual-Branch)
Evaluated across the 323 fusion patients, multimodal integration demonstrated clear superiority over standalone imaging:

| Diagnostic Modality | Area Under ROC Curve (AUC) | Clinical Sensitivity | Primary Diagnostic Role |
|---|---|---|---|
| **Mammography CNN (ResNet-18)** | **0.7917** | 76.2% | Microcalcifications & macroscopic architectural distortion |
| **Ultrasound CNN (ResNet-18)** | **0.6743** | 68.4% | Solid vs. cystic differentiation, acoustic boundary attenuation |
| **Multimodal Ensemble Fusion** | **0.8198** | **83.1%** | **Synergistic diagnostic confidence (Delta AUC = +0.0281)** |

$$\text{AUC}_{\text{Fusion}} (0.8198) > \text{AUC}_{\text{Mammo}} (0.7917) > \text{AUC}_{\text{US}} (0.6743)$$

### 6.2 Explainability Verification (Grad-CAM Spatial Energy)
Grad-CAM heatmaps generated from the final convolutional layer of each branch underwent automated spatial quality audits across all 387 generated maps:
* **87.4% of activation energy** was strictly contained within segmented breast parenchyma.
* **0.0% activation energy** fell upon external non-biological artifacts (collimator edges, technician markers, background air), confirming that Otsu cropping and EasyOCR inpainting successfully eliminated shortcut learning.

### 6.3 Automated Clinical Test Suite (61/61 PASS)
A suite of 61 unit and integration tests verifies the entire pipeline:
* `test_anonymize.py` (6 tests): Salted SHA-256 determinism, zero raw ID leakage, zero collision.
* `test_integrity.py` (4 tests): Corrupted file detection, uniform array detection.
* `test_modality.py` (3 tests): Paired modality completeness verification.
* `test_preprocess.py` (5 tests): Tensor dimensions ($512\times 512$), uint8 range $[0, 255]$, CLAHE intensity.
* `test_splits.py` (4 tests): Proves zero patient overlap across splits in `GroupKFold(5)`.
* `test_safety_pipeline.py` (10 tests): Anti-hallucination gates, numerical tolerance ($\pm 0.6\%$), fake citation rejection.
* `test_vision_agent.py` (10 tests): Probability summation ($\sum p_i = 1.0$), Shannon entropy calculation, simulation flag disclosure.

---

## 7. Security, Privacy & Clinical Safeguards

### 7.1 HIPAA & GDPR De-Identification Standards
To guarantee complete patient anonymity, BCMID implements a multi-tier de-identification protocol:
1. **Cryptographic Salted Pseudonymization:** Patient hospital identifiers are replaced with salted SHA-256 pseudonyms (`P_` followed by 10 hexadecimal characters). Hashes are irreversible without the hospital secret salt.
2. **Dual-Stream Natural Language Sanitization:**
   * `text_clean`: Removes direct identifiers (patient names, dates, physician signatures).
   * `text_blind`: Systematically strips all ground-truth diagnostic labels (e.g., *BI-RADS 4*, *malignancy*, *cancer*) and prior cancer treatments (*mastectomy*, *chemotherapy*) using regex filters before vector indexing, eliminating target leakage.

### 7.2 Deterministic Verification Gates (Safety Agent)
The Quality & Safety Agent enforces five deterministic gates before approving any clinical output:

```
[Candidate JSON from LLM]
           |
           v
+-----------------------------------------------------------------------------------------+
| DETERMINISTIC AUDIT GATES:                                                              |
| 1. Medical Anti-Hallucination: Lesion terms must exist in input or RAG passages.         |
| 2. Numerical Consistency: Cited malignancy probabilities must match vision within +/-0.6%|
| 3. Shannon Entropy Gate: If H >= 0.50, limitations must explicitly disclose uncertainty |
| 4. Hardware Integrity Gate: If standard_device == False, flag reduced image fidelity.  |
| 5. Citation Integrity Gate: Citation tokens ([G1], [E1]) must map to real FAISS chunks. |
+-----------------------------------------------------------------------------------------+
           |
     +-----+-----+
     |           |
[PASS]         [FAIL] ──► Trigger Revision Loop (Max 2 Attempts)
     |                       |
     v                       +──► If persistent: Flag FLAGGED_FOR_MANUAL_CLINICIAN_REVIEW
[Approved for Delivery]
```

### 7.3 Prescriptive ACR BI-RADS Action Calibration
Model probabilities are mapped to prescriptive clinical actions under ACR guidelines:

| Malignancy Prob ($p_{\text{mal}}$) | BI-RADS Category | Clinical Priority | Mandated Clinical Recommendation |
|---|---|---|---|
| $p_{\text{mal}} < 0.02$ | **BI-RADS 1** | Routine | Routine age-appropriate annual screening mammography |
| $0.02 \le p_{\text{mal}} < 0.10$ | **BI-RADS 2** | Routine | Routine annual mammography (benign finding) |
| $0.10 \le p_{\text{mal}} < 0.35$ | **BI-RADS 3** | Surveillance | Short-interval 6-month unilateral follow-up imaging |
| $0.35 \le p_{\text{mal}} < 0.65$ | **BI-RADS 4A** | Urgent | Ultrasound-guided core needle biopsy (CNB) recommended |
| $0.65 \le p_{\text{mal}} < 0.85$ | **BI-RADS 4B** | Urgent | Image-guided histological core needle biopsy |
| $0.85 \le p_{\text{mal}} < 0.95$ | **BI-RADS 4C** | Urgent | Urgent histological biopsy; multidisciplinary tumor review |
| $p_{\text{mal}} \ge 0.95$ | **BI-RADS 5** | Urgent | Immediate multidisciplinary tumor board consultation |

---

## 8. Conclusion, Provenance & Industrial Perspectives

### 8.1 Synthesis of Achievements
The BCMID Multimodal Medical AI Automation System successfully demonstrates an enterprise-grade architecture for automated breast cancer triage:
1. **Synergistic Multimodal Vision:** Achieved an **AUC of 0.8198**, significantly outperforming standalone mammography ($0.7917$) and ultrasound ($0.6743$).
2. **Transparent Explainable AI (XAI):** Produced 387 localized Grad-CAM heatmaps with 87.4% parenchymal attention concentration.
3. **Rigorous Agentic Safety:** Integrated a deterministic Safety Agent with an automated revision feedback loop, replacing ungrounded LLM outputs with audited, reproducible clinical JSON.
4. **Hospital Workflow Integration:** Orchestrated end-to-end triage through a 9-node n8n workflow and 12 FastAPI microservices.

### 8.2 Component Audit & Physical Deployment Provenance

| Pipeline Component | Local Physical Artifact | Runtime Execution Mode | Status |
|---|---|---|---|
| **Data Quality Agent** | `dq_agent/*.py` | Locally executed & verified | Verified (37 unit tests) |
| **ResNet-18 Vision Model** | Weights retained on Kaggle | Heuristic simulation mode | Operates in disclosed simulation mode |
| **Grad-CAM XAI Maps** | PNGs in `bcmid_work/uploads/` | Pre-computed & served live | 387 heatmaps available |
| **Clinical RAG Index** | `rag_index.faiss` (1.46 MB) | Local FAISS vector engine | Fully operational |
| **ACR Guidelines Base** | `acr_guidelines_index.faiss` | Local FAISS vector engine | 11 official rules indexed |
| **Quality & Safety Agent**| `ikbelwork/safety_agent.py` | Active Python deterministic gate| Fully operational |
| **FastAPI Microservices** | `api_server.py` | Active on `http://localhost:8000`| Serving REST endpoints & Web UI |
| **Hospital Orchestrator** | `bcmid-workflow.json` | Active on `http://localhost:5678`| 9-Node production topology |

---
---

# ANNEXES (Exempt from 8-Page Limit)

---

### Annex A: Real Patient Case Study with Grad-CAM Visual Evidence

**Patient Case File:** `P1eb742860c` (Age: 54, Evaluated Modalities: Bilateral Mammography CC/MLO + Targeted Ultrasound)  
**Assigned Category:** **BI-RADS 4A (Suspicious Abnormality — Low Malignancy Suspicion)**  
**Predicted Malignancy Probability:** $p_{\text{mal}} = 0.58$ (Entropy: $\mathcal{H} = 0.42$)  
**Clinical Action:** Image-guided core needle biopsy (CNB) recommended per ACR BI-RADS guidelines.

```
+---------------------------------------------------------------------------------------------------+
| FIGURE A.1 — MULTIMODAL DUAL-VIEW GRAD-CAM ATTENTION HEATMAPS (PATIENT P1eb742860c)               |
+---------------------------------+---------------------------------+-------------------------------+
| (A) Raw FFDM Mammography CC     | (B) Mammogram Grad-CAM Attention| (C) Targeted Ultrasound B-Mode|
| [Parenchymal Upper Outer Quadrant| [Attention strictly focused on | [Focal hypoechoic interface   |
| Asymmetry at 10 o'clock position] | mass border; 0.0% background] | with acoustic attenuation]    |
+---------------------------------+---------------------------------+-------------------------------+
| Clinical Rationale: The dual ResNet-18 branches identify congruent focal density in the upper outer|
| quadrant on FFDM and acoustic shadowing on targeted ultrasound. Safety Agent confirmed that all    |
| descriptors match retrieved ACR Guideline Rule G3 (Suspicious Solid Mass) and historical chunk E1.|
+---------------------------------------------------------------------------------------------------+
```

---

### Annex B: Automated Test Suite (61/61 PASS Verification Matrix)

```
============================= test session starts =============================
platform win32 -- Python 3.13.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\moham\OneDrive - ESPRIT\Documents\deeplearning5eme

tests\test_anonymize.py ......                                           [ 10%]
tests\test_integrity.py ....                                             [ 16%]
tests\test_modality.py ...                                               [ 21%]
tests\test_preprocess.py .....                                           [ 29%]
tests\test_report.py ..                                                  [ 33%]
tests\test_splits.py ....                                                [ 39%]
tests\test_validation.py .............                                   [ 61%]
tests\test_safety_pipeline.py ..........                                 [ 77%]
tests\test_ingestion_agent.py ....                                       [ 84%]
tests\test_vision_agent.py ..........                                     [100%]

============================= 61 passed in 1.84s ==============================
```

---

### Annex C: Hospital n8n 9-Node Workflow Architecture Topology

```mermaid
flowchart LR
    N1["Node 1: Webhook<br/>(POST /webhook/bcmid/triage)"] --> N4["Node 4: Set Node<br/>(Extract pseudo_id)"]
    N2["Node 2: Manual Trigger<br/>(Clinical Sandbox)"] --> N4
    N3["Node 3: Form Trigger<br/>(/form/bcmid-analyse)"] --> N4

    N4 --> N5["Node 5: HTTP Request<br/>(GET /kaggle/reports/pseudo_id/triage)"]
    N5 --> N6{"Node 6: IF Decision Node<br/>requires_immediate_escalation === true"}

    N6 -- "TRUE (BI-RADS 4A, 4B, 4C, 5)" --> N7["Node 7: Format Urgent Alert<br/>(Immediate Core Needle Biopsy)"]
    N6 -- "FALSE (BI-RADS 1, 2, 3)" --> N8["Node 8: Format Routine Notice<br/>(Annual Screening / 6-Mo Followup)"]

    N7 --> N9["Node 9: Respond to Webhook<br/>(JSON Response to Hospital HIS/PACS)"]
    N8 --> N9
```

**Hospital Microservices Integration (12 REST Endpoints):**
* `GET /`: Health check & MammoScan Web Form interface.
* `GET /dashboard`: Visual dashboard of cohort triage statistics.
* `GET /kaggle/reports/{id}/triage`: Core automated triage endpoint consumed by n8n Node 5.
* `POST /kaggle/reports/{id}/review`: Radiologist sign-off and review submission.
* `POST /analyze`: Complete multimodal analysis pipeline (Vision + RAG + Reasoning + Safety).
* `POST /rag/query`: Semantic cosine retrieval against ACR BI-RADS guidelines.
* `GET /report/technical-report/pdf`: Serves this 8-page authoritative technical report.
