
# Rapport de Qualité — Data & Quality Agent (BCMID)

**Généré le :** 2026-10-05T08:03:06.802460+00:00


## 1. Résumé Dataset

| Métrique | Valeur |
| --- | --- |
| Dossiers patients (ZIP) | 332 |
| Patients dans les annotations (TSV) | 332 |
| **Écart détecté** | **0** |
| Patients prêts (multimodal complet) | 332 |
| Patients sans rapport PDF | 11 |

## 2. Distribution BI-RADS

| Classe BI-RADS | Nombre de patients |
| --- | --- |
| 0 | 5 |
| 1 | 50 |
| 2 | 167 |
| 3 | 47 |
| 4 | 24 |
| 5 | 15 |
| 6 | 24 |

## 3. Statuts de Modalité

| Statut | Nombre |
| --- | --- |
| complete | 332 |

## 4. Intégrité des Images

| Indicateur | Valeur |
| --- | --- |
| Images corrompues / invalides | 1 |
| Groupes de doublons exacts | 6 |

### Exemples d'images invalides (max 20)

| Fichier | Raison |
| --- | --- |
| P0c0449d36e/Ultrasound/P0c0449d36e_US_1.png | cannot decode image |

## 5. Annotations BI-RADS

| Indicateur | Valeur |
| --- | --- |
| Colonnes TSV détectées | Patient ID, BI-RADS Label, Binary Label, Pathology Label |
| Colonne ID détectée | Patient ID |
| Colonne label détectée | BI-RADS Label |
| Patients sans label | 0 |
| Labels sans dossier | 0 |
| IDs dupliqués dans TSV | 0 |
| Labels mal formés | 0 |

## 6. Statistiques Images


**Échographie (US)** — min=1, max=16, moy=3.16, écart-type=2.47


**Mammographie** — min=1, max=4, moy=2.86, écart-type=1.0


## 7. Paramètres de Prétraitement

| Paramètre | Valeur |
| --- | --- |
| target_size | 512 |
| mammo_clahe_clip_limit | 2.0 |
| mammo_clahe_tile_grid | 8 |
| us_median_ksize | 3 |
| nlm_h | 5 |
| bit_depth | 8 |

## 8. Versions des Librairies

| Librairie | Version |
| --- | --- |
| Pillow | 12.3.0 |
| numpy | 2.4.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| tqdm | 4.67.1 |
| pyyaml | 6.0.1 |
| pydicom | N/A |