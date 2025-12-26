# Artifacts

Ce document décrit les entrées/sorties produites par chaque stage du pipeline de traitement de PDF pour l'extraction d'annuaires historiques.

## Terminologie

- **run root**: `RunContext.output_dir` (par défaut `./<pdf_stem>/artifacts/`).
- **stage dir**: un sous-dossier sous le run root, calculé via `BasePipelineStage.stage_name()`.
  - Dans le code actuel, les stages définissent `STAGE_NAME` pour des noms stables: `ocr/`, `chunking/`, `extraction/`, `assembly/`.
- **Artifact final**: Fichier JSON contenant l'artifact complet produit par le stage (e.g., `ocr.json`).
- **Artifact streaming**: Fichier JSONL contenant les éléments individuels émis pendant l'exécution (e.g., `ocr.items.jsonl`).
- **Cache**: Fichier `.cache` associé au PDF pour éviter de recalculer les stages inchangés.

## Pipeline Overview

Le pipeline suit cette séquence de stages :

1. **OCR** : Reconnaissance optique de caractères et analyse de layout.
2. **Chunking** : Découpage du texte OCR en chunks de taille limitée.
3. **Extraction** : Extraction structurée des entités (entrées d'annuaire) à partir des chunks.
4. **Assembly** : Assemblage final des entités extraites avec enrichissement (texte brut, alignement, bounding boxes).

## 1/ OCR

- **Entrée** : Fichier PDF.
- **Cache** : `<pdf_path>.ocr_cache` (contient l'artifact OCRDocument sérialisé).
- **Artifact final** : `<run_root>/ocr.json` (contenu : `OCRDocument`).
- **Artifact streaming** : `<run_root>/ocr.items.jsonl` (non utilisé pour OCR, car l'artifact est atomique).
- **Contenu de OCRDocument** :
  - `layout` : Liste de `LayoutResult` (résultats d'analyse de layout par page, avec bounding boxes des éléments).
  - `ocr` : Liste de `OCRResult` (résultats OCR par page, avec `text_lines` contenant le texte reconnu et métadonnées).
- **Debug** (si `RunContext.debug=True`) :
  - Répertoire : `<run_root>/ocr/layout/`.
  - Fichiers : `page_<n>_layout.png` (visualisations du layout).

## 2/ Chunking

- **Entrée** : Artifact OCR (`OCRDocument`).
- **Artifact final** : `<run_root>/chunking.json` (contenu : `Chunks`).
- **Artifact streaming** : `<run_root>/chunking.items.jsonl` (chaque ligne : un `Chunk`).
- **Contenu de Chunks** :
  - Liste de `Chunk`, chaque `Chunk` étant une liste de `ChunkLine`.
- **Contenu de ChunkLine** :
  - `page` : Numéro de page (0-based).
  - `line` : Index de ligne dans la page.
  - `layout_ix` : Index dans le layout de la page.
  - `layout_label` : Label du layout (e.g., "Title", "Text").
  - `confidence` : Confiance de l'OCR.
  - `spuriousness` : Score de spuriousness (lignes parasites).
  - `text` : Texte de la ligne.
  - `is_margin` : Booléen indiquant si la ligne est en marge.

## 3/ Extraction

- **Entrée** : Artifact Chunking (`Chunks`).
- **Artifact final** : `<run_root>/extraction.json` (contenu : `StructuredSeq`).
- **Artifact streaming** : `<run_root>/extraction.items.jsonl` (chaque ligne : un `Structured` ou `StructuredCompact`).
- **Contenu de StructuredSeq** :
  - Séquence de `Structured` ou `StructuredCompact` (modèles compacts pour optimisation).
- **Contenu de Structured** :
  - `items` : Liste d'éléments extraits (`Entry`, `Title`, `Text`).
- **Types d'éléments** :
  - `Entry` : Entrée d'annuaire (nom, activité, adresses).
  - `Title` : Titre ou en-tête.
  - `Text` : Texte libre.
- **Debug** (si `RunContext.debug=True` et `ExtractionStageParams.debug_dir` défini) :
  - Répertoire : `<run_root>/extraction/debug/`.
  - Fichiers : `chunk_<n>.json` (résultats intermédiaires par chunk).

## 4/ Assembly

- **Entrées** : Artifacts OCR, Chunking, Extraction.
- **Artifact final** : `<run_root>/assembly.json` (contenu : `RichStructured`).
- **Artifact streaming** : `<run_root>/assembly.items.jsonl` (chaque ligne : un `RichEntry`, `RichTitle`, ou `RichText`).
- **Contenu de RichStructured** :
  - `items` : Liste d'éléments enrichis (`RichEntry`, `RichTitle`, `RichText`).
  - `images_bboxes` : Bounding boxes des images par page.
- **Enrichissement** :
  - `raw_text` : Texte brut assemblé.
  - `alignment` : Score d'alignement (0.0 à 1.0).
  - `lines_resolved` : Liste des `ChunkLine` résolues.
  - `bboxes` : Bounding boxes associées.
- **Sortie CSV** : Générée via `scripts/assembled_to_csv.sh` à partir de `assembly.json`.
  - Format : Colonnes pour ID, catégorie, alignement, texte, nom, activité, adresse, etc.

## 5/ Cache et Invalidation

- Le cache utilise des fichiers `.cache` (shelve) associés au PDF.
- Invalidation basée sur empreinte (fingerprint) des paramètres et métadonnées du stage.
