# Dirzas

This repository contains a minimal example of a retrieval-augmented generation (RAG) script that groups text by speaker and queries a local Ollama language model (e.g. `phi4`).

## Usage

1. Place `.txt` or `.docx` files in your own directory (e.g. `docs/`). Each line should begin with the speaker name followed by a colon or a period, for example:
```
Alice: Sveikas, tai yra tekstinis dokumentas.
Bob: Sveikas, aš Bobas.
PIRMININKAS (S. SKVERNELIS). Labas rytas.
```
2. Run the script:
```
python src/rag_speaker.py docs --speaker Alice --query "Kaip sekasi?"
```
3. Add `--dry-run` to show retrieved context without contacting the Ollama server.

Ensure that an [Ollama](https://ollama.com/) server is running locally and the `phi4` model is available if you want to generate answers.

> **Note:** Sample documents are not included in this repository.
