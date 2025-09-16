# Dirzas

This repository contains a minimal example of a retrieval-augmented generation (RAG) script that groups text by speaker and queries a local Ollama language model (e.g. `phi4`).

## Web UI

A simple Gradio-based interface is available via `python src/web_ui.py`. Upload `.txt` or `.docx` files in the *Documents* tab or paste a YouTube link into the *YouTube* tab. The app retrieves the transcript for the supplied video and stores it under a synthetic "YouTube ..." speaker name. If the video title contains a date (e.g. `2024-05-16` or `2024 m. gegužės 16 d.`), that date is assigned automatically so the conversation also becomes searchable from the *Date Chat* tab.

YouTube transcriptions are produced locally by downloading the audio with [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) and running an [OpenAI Whisper](https://github.com/openai/whisper) model. Install [FFmpeg](https://ffmpeg.org/) so that yt-dlp and Whisper can decode the audio stream. The default model is `large`, but you can override it by setting the `WHISPER_MODEL` environment variable before launching the app, for example `WHISPER_MODEL=base python src/web_ui.py`.

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
