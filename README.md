
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

### Exemples de résultats

**1797 - Almanach du commerce de la ville de Paris[...], Duverneuil & La Tynna**

- [Liste des négocians, marchands et courtiers de la ville de Paris](https://hueynemud.github.io/directory-explorer/?entitiesUrl=https://files.catbox.moe/ev9iv0.json&manifestUrl=https://gallica.bnf.fr/iiif/ark%3A%2F12148%2Fbd6t58275556/manifest.json&pageOffset=21)
- [Autre listes particulières : banquiers, médecins, etc.](https://hueynemud.github.io/directory-explorer/?entitiesUrl=https://files.catbox.moe/b10mot.json&manifestUrl=https://gallica.bnf.fr/iiif/ark%3A%2F12148%2Fbd6t58275556/manifest.json&pageOffset=295)

**1807 - Almanach du commerce de Paris[...], Duverneuil & La Tynna**

- [Liste particulière par ordre alphabétique des fabricans, négocians, banquiers, agens de change, commerçans, marchands de tous états, de Paris](
https://hueynemud.github.io/directory-explorer/?entitiesUrl=https://files.catbox.moe/a8cjvs.json&manifestUrl=https://gallica.bnf.fr/iiif/ark%3A%2F12148%2Fbpt6k124557p/manifest.json&pageOffset=119)

**1845 - Annuaire général du commerce[...],  Firmin-Didot frères**

- [Liste générale des adresses de Paris et des principaux établissements de cette capitale](https://hueynemud.github.io/directory-explorer/?entitiesUrl=https://files.catbox.moe/1vimup.json&manifestUrl=https://gallica.bnf.fr/iiif/ark%3A%2F12148%2Fbpt6k6292987t/manifest.json&pageOffset=352)
