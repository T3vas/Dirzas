from typing import List, Tuple

import gradio as gr
import json
from pathlib import Path

from rag_speaker import (
    SpeakerRAG,
    call_ollama,
    extract_document_date,
    load_text,
    parse_speakers,
)

rag = SpeakerRAG()
date_rag = SpeakerRAG()

DEFAULT_URL = 'http://localhost:11434/api/generate'
DEFAULT_MODEL = 'phi4'
CONFIG_PATH = Path(__file__).resolve().parent.parent / 'settings.json'

UNKNOWN_DATE_LABEL = 'Nežinoma data'

ollama_url = DEFAULT_URL
model_name = DEFAULT_MODEL

if CONFIG_PATH.exists():
    with CONFIG_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)
        ollama_url = data.get('ollama_url', DEFAULT_URL)
        model_name = data.get('model_name', DEFAULT_MODEL)

def chat_fn(history: List[Tuple[str, str]], question: str, speaker: str):
    if not speaker:
        history.append((question, 'Please select a speaker'))
        return history, ''
    if speaker not in rag.corpora:
        history.append((question, f"Speaker '{speaker}' not found"))
        return history, ''
    segments = rag.retrieve(speaker, question)
    context = '\n\n'.join(segments)
    prompt = (
        f"Use the following context from {speaker} to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = call_ollama(model=model_name, prompt=prompt, url=ollama_url)
    history.append((question, answer))
    return history, ''

def date_chat_fn(history: List[Tuple[str, str]], question: str, date_label: str):
    if not date_label:
        history.append((question, 'Please select a document date'))
        return history, ''
    if date_label not in date_rag.corpora:
        history.append((question, f"Date '{date_label}' not found"))
        return history, ''
    segments = date_rag.retrieve(date_label, question)
    context = '\n\n'.join(segments)
    prompt = (
        f"Use the following context from documents dated {date_label} to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = call_ollama(model=model_name, prompt=prompt, url=ollama_url)
    history.append((question, answer))
    return history, ''


def upload_fn(files: List[gr.File]) -> Tuple[gr.Dropdown, str, gr.Dropdown, str]:
    if not files:
        speakers = sorted(rag.corpora.keys())
        dates = sorted(date_rag.corpora.keys())
        return (
            gr.Dropdown(choices=speakers),
            ', '.join(speakers),
            gr.Dropdown(choices=dates),
            ', '.join(dates),
        )
    for file in files:
        try:
            text = load_text(file.name)
        except ValueError:
            # Skip files with unsupported extensions
            continue
        parsed = parse_speakers(text)
        date_segments: List[str] = []
        for sp, segs in parsed.items():
            existing = rag.corpora.get(sp, [])
            rag.add_speaker(sp, existing + segs)
            date_segments.extend(segs)
        if not date_segments:
            date_segments = [
                paragraph.strip()
                for paragraph in text.split('\n\n')
                if paragraph.strip()
            ]
        date_label = extract_document_date(text) or UNKNOWN_DATE_LABEL
        existing_date = date_rag.corpora.get(date_label, [])
        if date_segments:
            date_rag.add_speaker(date_label, existing_date + date_segments)
    speakers = sorted(rag.corpora.keys())
    dates = sorted(date_rag.corpora.keys())
    speakers_text = ', '.join(speakers)
    dates_text = ', '.join(dates)
    return (
        gr.Dropdown(choices=speakers),
        speakers_text,
        gr.Dropdown(choices=dates),
        dates_text,
    )


def clear_db() -> Tuple[gr.Dropdown, str, gr.Dropdown, str, str]:
    rag.corpora.clear()
    rag.tokens.clear()
    date_rag.corpora.clear()
    date_rag.tokens.clear()
    return (
        gr.Dropdown(choices=[]),
        'Database cleared',
        gr.Dropdown(choices=[]),
        '',
        '',
    )

def save_settings(url: str, model: str) -> str:
    global ollama_url, model_name
    ollama_url = url
    model_name = model
    with CONFIG_PATH.open('w', encoding='utf-8') as f:
        json.dump({'ollama_url': ollama_url, 'model_name': model_name}, f)
    return 'Settings saved'

with gr.Blocks() as demo:
    gr.Markdown('# RAG Web UI')
    with gr.Tab('Chat'):
        speaker_dd = gr.Dropdown(label='Speaker', choices=[])
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label='Your question')
        msg.submit(chat_fn, [chatbot, msg, speaker_dd], [chatbot, msg])
    with gr.Tab('Date Chat'):
        date_dd = gr.Dropdown(label='Document date', choices=[])
        date_chatbot = gr.Chatbot(height=400)
        date_msg = gr.Textbox(label='Your question')
        date_msg.submit(date_chat_fn, [date_chatbot, date_msg, date_dd], [date_chatbot, date_msg])
    with gr.Tab('Documents'):
        # Gradio 4 does not recognise ``docx`` in ``file_types``. Allow any
        # file to be uploaded and rely on ``load_text`` to filter by
        # extension instead.
        file_input = gr.File(file_types=['file'], file_count='multiple', label='Upload documents')
        upload_btn = gr.Button('Add to database')
        speakers_box = gr.Textbox(label='Known speakers', interactive=False)
        dates_box = gr.Textbox(label='Known dates', interactive=False)
        upload_btn.click(upload_fn, [file_input], [speaker_dd, speakers_box, date_dd, dates_box])
    with gr.Tab('Settings'):
        url_box = gr.Textbox(label='Ollama URL', value=ollama_url)
        model_box = gr.Textbox(label='LLM Model', value=model_name)
        clear_btn = gr.Button('Clear database')
        save_btn = gr.Button('Save settings')
        status = gr.Textbox(label='Status', interactive=False)
        clear_btn.click(
            clear_db,
            [],
            [speaker_dd, status, date_dd, speakers_box, dates_box],
        )
        save_btn.click(save_settings, [url_box, model_box], [status])

if __name__ == '__main__':
    demo.launch()
