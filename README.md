
# SoduCo v4 Pipeline

Pipeline OCR → chunking → extraction → assembly pour l'extraction d'annuaires historiques, piloté par un CLI `soduco`.

## Prérequis

- Python >= 3.12
- OCR: `surya-ocr` (installé via `pyproject.toml`)

## Installation

Le repo est configuré comme projet Python (PEP 621). Avec `uv`:

```sh
uv sync
```

## Utilisation

Aide:

```sh
python3 -m cli.soduco --help
```

Stages séparés (pipeline complet = ocr + chunking + extraction + assembly):

```sh
python3 -m cli.soduco ocr path/to/file.pdf
python3 -m cli.soduco chunking path/to/file.pdf
python3 -m cli.soduco extraction path/to/file.pdf --system-prompt-file prompts/system_prompt-ollama-latynna.txt
python3 -m cli.soduco assembly path/to/file.pdf
```

Flags utiles:

- `--debug`: écrit des artefacts de debug
- `--verbose`: logs plus verbeux

## Sorties

Par défaut, les sorties sont écrites sous:

`./<pdf_stem>/artifacts/`
