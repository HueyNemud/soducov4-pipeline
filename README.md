# SoDUCo — v.4 Directory Processing Pipeline

This repository contains **version 4 of the structured extraction pipeline** for **directory entries** found in Parisian city directories and commercial almanacs from the **19th and early 20th centuries**.

This pipeline extracts structured directory entries while preserving their spatial position in the original document.

It adapts the processing pipeline developed during the research project
[SoDUCo (2019–2023)](https://soduco.geohistoricaldata.org/) to integrate two major technological advances:

1. **OCR, layout parsing, and reading order detection** using [Surya OCR](https://github.com/datalab-to/surya)
2. **Structured extraction using Large Language Models**, either via Mistral (cloud) or locally with Ollama

---

## Installation

**Requirements:** Python 3.12+

Using [uv](https://github.com/astral-sh/uv):

```sh
uv sync
```

Using pip (≥ 21.3):

```sh
pip install .
```

---

## Usage

The pipeline is composed of **four sequential steps**:

1. **OCR** — document reading, layout detection, and reading order inference
2. **Chunking** — segmentation of text into blocks suitable for structured extraction
3. **Extraction** — structured extraction inside text blocks using an LLM
4. **Assembly** — alignment of extracted results with the original document

---

## Running the complete pipeline

The pipeline is designed to process **a single PDF document end-to-end**, step by step.

> 💡 Each step is **cached** and will only be recomputed if the `--force` flag is provided.

Example input file: `examples/didot_1903.pdf`

* Step outputs are written to: `examples/didot_1903/artifacts/`
* Cache directory: `examples/didot_1903.cache/`

### CLI help

```sh
uv run python -m cli.soduco --help
```

---

### Step 1 — OCR

Analyzes the PDF and extracts the document layout and text in reading order.

```sh
uv run python -m cli.soduco ocr examples/didot_1903.pdf
```

**Output**
`examples/didot_1903/artifacts/ocr.json`

* text lines in reading order
* detected layout blocks

---

### Step 2 — Chunking

Groups OCR lines into fixed-length text blocks to fit LLM context windows.

```sh
python -m cli.soduco chunking examples/didot_1903.pdf
```

**Output**
`examples/didot_1903/artifacts/chunking.json`

* sequences of lines formatted as: `line_number @ text`

---

### Step 3 — Structured extraction

Uses a language model to extract structured information from text blocks.

```sh
python -m cli.soduco \
    extraction \
    --system-prompt-file prompts/system_prompt-mistral-didottype1900.txt \
    --mistral-api-key your_api_key_here \
    examples/didot_1903.pdf
```

**Output**
`examples/didot_1903/artifacts/extraction.json`

Extracted entities include:

* Directory entry (`cat = "ent"`)
* Title (`cat = "title"`)
* Isolated text (`cat = "txt"`)

---

### Step 4 — Assembly

Aligns the LLM extraction results with the original OCR data to produce spatially aligned structured data.

```sh
python -m cli.soduco \
    assembly \
    examples/didot_1903.pdf
```

**Output**
`examples/didot_1903/artifacts/assembly.json`

Final file linking structured extraction with the original OCR and layout information.

---

### Common options (all steps)

* `--force` — recompute the step even if cached
* `--debug` — display internal logs
* `--verbose` — enable detailed output

---

## Visualizing results

The web application [https://hueynemud.github.io/directory-explorer](https://hueynemud.github.io/directory-explorer) allows you to visualize extracted results overlaid on the original document, provided it is available via IIIF.

The application accepts three URL parameters:

1. `entitiesUrl` — URL of the `assembly.json` file (for example, hosted on [https://litterbox.catbox.moe/](https://litterbox.catbox.moe/))
2. `manifestUrl` — IIIF manifest of the original document
3. `pageOffset` — optional; offset between the first PDF page and the first IIIF view

---

## Example results

### **1797 — Almanach du commerce de la ville de Paris**, Duverneuil & La Tynna

* [List of merchants, traders, and brokers of Paris](https://hueynemud.github.io/directory-explorer/?entitiesUrl=https://files.catbox.moe/ev9iv0.json&manifestUrl=https://gallica.bnf.fr/iiif/ark%3A%2F12148%2Fbd6t58275556/manifest.json&pageOffset=21)
* [Other specialized lists: bankers, physicians, etc.](https://hueynemud.github.io/directory-explorer/?entitiesUrl=https://files.catbox.moe/b10mot.json&manifestUrl=https://gallica.bnf.fr/iiif/ark%3A%2F12148%2Fbd6t58275556/manifest.json&pageOffset=295)

---

### **1807 — Almanach du commerce de Paris**, Duverneuil & La Tynna

* [Alphabetical list of manufacturers, merchants, bankers, brokers, and tradespeople of Paris](https://hueynemud.github.io/directory-explorer/?entitiesUrl=https://files.catbox.moe/a8cjvs.json&manifestUrl=https://gallica.bnf.fr/iiif/ark%3A%2F12148%2Fbpt6k124557p/manifest.json&pageOffset=119)

---

### **1845 — Annuaire général du commerce**, Firmin-Didot frères

* [General list of addresses in Paris and major establishments](https://hueynemud.github.io/directory-explorer/?entitiesUrl=https://files.catbox.moe/1vimup.json&manifestUrl=https://gallica.bnf.fr/iiif/ark%3A%2F12148%2Fbpt6k6292987t/manifest.json&pageOffset=352)
