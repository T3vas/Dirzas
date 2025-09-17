from typing import List, Optional, Tuple

import json
import re
import unicodedata
from datetime import date as dt_date
from pathlib import Path

import gradio as gr

from rag_speaker import (
    SpeakerRAG,
    call_ollama,
    extract_document_date,
    fetch_youtube_metadata,
    fetch_youtube_transcript,
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


_LT_MONTHS = {
    'sausio': 1,
    'sausis': 1,
    'vasario': 2,
    'vasaris': 2,
    'kovo': 3,
    'kovas': 3,
    'balandzio': 4,
    'balandis': 4,
    'geguzes': 5,
    'geguze': 5,
    'birzelio': 6,
    'birzelis': 6,
    'liepos': 7,
    'liepa': 7,
    'rugpjucio': 8,
    'rugpjutis': 8,
    'rugsėjo': 9,
    'rugsejo': 9,
    'rugsejis': 9,
    'spalio': 10,
    'spalis': 10,
    'lapkricio': 11,
    'lapkritis': 11,
    'gruodzio': 12,
    'gruodis': 12,
}


def _strip_accents(text: str) -> str:
    return ''.join(
        ch for ch in unicodedata.normalize('NFD', text)
        if unicodedata.category(ch) != 'Mn'
    )


def _parse_date_label(label: str) -> Optional[dt_date]:
    cleaned = (label or '').strip()
    if not cleaned or cleaned == UNKNOWN_DATE_LABEL:
        return None
    try:
        return dt_date.fromisoformat(cleaned)
    except ValueError:
        pass

    match = re.search(
        r'(\d{4})\s*m\.\s*([A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž\s\.]*)\s+(\d{1,2})\s*d\.?',
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        return None

    year = int(match.group(1))
    month_raw = match.group(2)
    day = int(match.group(3))

    month = None
    for token in re.sub(r'[^A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž\s]', ' ', month_raw).split():
        base = _strip_accents(token.lower())
        if base == 'men':
            continue
        month = _LT_MONTHS.get(base)
        if month is not None:
            break

    if month is None:
        return None

    try:
        return dt_date(year, month, day)
    except ValueError:
        return None


def _date_sort_key(label: str):
    parsed = _parse_date_label(label)
    return (parsed or dt_date.max, label.lower())


def _sorted_date_labels() -> List[str]:
    return sorted(date_rag.corpora.keys(), key=_date_sort_key)


def _default_period_values(dates: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not dates:
        return None, None
    parseable = [d for d in dates if _parse_date_label(d)]
    if parseable:
        return parseable[0], parseable[-1]
    if UNKNOWN_DATE_LABEL in dates:
        return UNKNOWN_DATE_LABEL, UNKNOWN_DATE_LABEL
    first = dates[0]
    return first, first


def _retrieve_period_segments(question: str, labels: List[str], top_k: int = 3) -> List[str]:
    combined: List[str] = []
    for label in labels:
        for segment in date_rag.corpora.get(label, []):
            combined.append(f'[{label}] {segment}')
    if not combined:
        return []
    temp_rag = SpeakerRAG()
    temp_rag.add_speaker('_period_', combined)
    return temp_rag.retrieve('_period_', question, top_k=top_k)

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

def date_chat_fn(
    history: List[Tuple[str, str]],
    question: str,
    start_label: str,
    end_label: str,
):
    if not start_label or not end_label:
        history.append((question, 'Please select both start and end dates'))
        return history, ''
    if start_label not in date_rag.corpora:
        history.append((question, f"Date '{start_label}' not found"))
        return history, ''
    if end_label not in date_rag.corpora:
        history.append((question, f"Date '{end_label}' not found"))
        return history, ''

    if start_label == end_label:
        segments = date_rag.retrieve(start_label, question)
        if not segments:
            history.append((question, f'No context available for {start_label}'))
            return history, ''
        if start_label == UNKNOWN_DATE_LABEL:
            descriptor = 'documents with an unknown date'
        else:
            descriptor = f'documents dated {start_label}'
        context = '\n\n'.join(segments)
        prompt = (
            f"Use the following context from {descriptor} to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        answer = call_ollama(model=model_name, prompt=prompt, url=ollama_url)
        history.append((question, answer))
        return history, ''

    if UNKNOWN_DATE_LABEL in {start_label, end_label}:
        history.append(
            (
                question,
                'The "Unknown date" entry can only be queried on its own.',
            )
        )
        return history, ''

    start_date = _parse_date_label(start_label)
    end_date = _parse_date_label(end_label)
    if start_date is None or end_date is None:
        history.append((question, 'Unable to interpret the selected period'))
        return history, ''
    if start_date > end_date:
        history.append((question, 'Period start must not be later than period end'))
        return history, ''

    labels_in_period = [
        label
        for label in date_rag.corpora
        if (parsed := _parse_date_label(label)) and start_date <= parsed <= end_date
    ]
    labels_in_period.sort(key=_date_sort_key)
    if not labels_in_period:
        history.append((question, 'No documents found in the selected period'))
        return history, ''

    segments = _retrieve_period_segments(question, labels_in_period)
    if not segments:
        history.append((question, 'No relevant context found for the selected period'))
        return history, ''

    period_desc = f"documents dated between {start_label} and {end_label}"
    context = '\n\n'.join(segments)
    prompt = (
        f"Use the following context from {period_desc} to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = call_ollama(model=model_name, prompt=prompt, url=ollama_url)
    history.append((question, answer))
    return history, ''


def upload_fn(
    files: List[gr.File],
) -> Tuple[gr.Dropdown, str, gr.Dropdown, gr.Dropdown, str]:
    if not files:
        speaker_dd, speakers_text, start_dd, end_dd, dates_text = _current_choices()
        return (
            speaker_dd,
            speakers_text,
            start_dd,
            end_dd,
            dates_text,
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
    speaker_dd, speakers_text, start_dd, end_dd, dates_text = _current_choices()
    return (
        speaker_dd,
        speakers_text,
        start_dd,
        end_dd,
        dates_text,
    )


def clear_db() -> Tuple[gr.Dropdown, str, gr.Dropdown, gr.Dropdown, str, str]:
    rag.corpora.clear()
    rag.tokens.clear()
    date_rag.corpora.clear()
    date_rag.tokens.clear()
    speaker_dd, speakers_text, start_dd, end_dd, dates_text = _current_choices()
    return (
        speaker_dd,
        'Database cleared',
        start_dd,
        end_dd,
        speakers_text,
        dates_text,
    )

def save_settings(url: str, model: str) -> str:
    global ollama_url, model_name
    ollama_url = url
    model_name = model
    with CONFIG_PATH.open('w', encoding='utf-8') as f:
        json.dump({'ollama_url': ollama_url, 'model_name': model_name}, f)
    return 'Settings saved'


def _current_choices(
    selected_speaker: Optional[str] = None,
    selected_start: Optional[str] = None,
    selected_end: Optional[str] = None,
):
    speakers = sorted(rag.corpora.keys())
    dates = _sorted_date_labels()
    speaker_value = selected_speaker if selected_speaker in speakers else None
    default_start, default_end = _default_period_values(dates)
    start_value = selected_start if selected_start in dates else default_start
    end_value = selected_end if selected_end in dates else default_end
    return (
        gr.Dropdown(choices=speakers, value=speaker_value),
        ', '.join(speakers),
        gr.Dropdown(choices=dates, value=start_value),
        gr.Dropdown(choices=dates, value=end_value),
        ', '.join(dates),
    )


def ingest_youtube(url: str):
    url = (url or '').strip()
    if not url:
        dropdowns = _current_choices()
        return (*dropdowns, 'Įveskite YouTube nuorodą')

    try:
        video_id, title, date_label, warning = fetch_youtube_metadata(url)
    except ValueError as exc:
        dropdowns = _current_choices()
        return (*dropdowns, str(exc))

    try:
        segments = fetch_youtube_transcript(video_id)
    except Exception as exc:
        dropdowns = _current_choices()
        return (*dropdowns, str(exc))

    if not segments:
        dropdowns = _current_choices()
        return (*dropdowns, 'Transkripcija tuščia')

    speaker_label = f'YouTube {video_id}: {title}'
    if speaker_label in rag.corpora:
        selected_date = date_label or UNKNOWN_DATE_LABEL
        dropdowns = _current_choices(speaker_label, selected_date, selected_date)
        message = f"Vaizdo įrašas '{title}' jau pridėtas"
        if warning:
            message += f'. {warning}'
        return (*dropdowns, message)

    rag.add_speaker(speaker_label, segments)
    date_key = date_label or UNKNOWN_DATE_LABEL
    existing_date = date_rag.corpora.get(date_key, [])
    date_rag.add_speaker(date_key, existing_date + segments)

    dropdowns = _current_choices(speaker_label, date_key, date_key)
    message = (
        f"Pridėtas YouTube vaizdo įrašas '{title}' ({len(segments)} segmentų)."
    )
    if date_label:
        message += f" Data iš pavadinimo: {date_label}."
    else:
        message += ' Data nenustatyta, priskirta prie "Nežinoma data".'
    if warning:
        message += f' Įspėjimas: {warning}.'
    return (*dropdowns, message)

with gr.Blocks() as demo:
    gr.Markdown('# RAG Web UI')
    with gr.Tab('Chat'):
        speaker_dd = gr.Dropdown(label='Speaker', choices=[])
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label='Your question')
        msg.submit(chat_fn, [chatbot, msg, speaker_dd], [chatbot, msg])
    with gr.Tab('Date Chat'):
        date_start_dd = gr.Dropdown(label='Start date', choices=[])
        date_end_dd = gr.Dropdown(label='End date', choices=[])
        date_chatbot = gr.Chatbot(height=400)
        date_msg = gr.Textbox(label='Your question')
        date_msg.submit(
            date_chat_fn,
            [date_chatbot, date_msg, date_start_dd, date_end_dd],
            [date_chatbot, date_msg],
        )
    with gr.Tab('Documents'):
        # Gradio 4 does not recognise ``docx`` in ``file_types``. Allow any
        # file to be uploaded and rely on ``load_text`` to filter by
        # extension instead.
        file_input = gr.File(file_types=['file'], file_count='multiple', label='Upload documents')
        upload_btn = gr.Button('Add to database')
        speakers_box = gr.Textbox(label='Known speakers', interactive=False)
        dates_box = gr.Textbox(label='Known dates', interactive=False)
        upload_btn.click(
            upload_fn,
            [file_input],
            [speaker_dd, speakers_box, date_start_dd, date_end_dd, dates_box],
        )
    with gr.Tab('YouTube'):
        youtube_url = gr.Textbox(label='YouTube URL')
        youtube_btn = gr.Button('Transcribe video')
        youtube_status = gr.Textbox(label='Status', interactive=False)
        youtube_btn.click(
            ingest_youtube,
            [youtube_url],
            [
                speaker_dd,
                speakers_box,
                date_start_dd,
                date_end_dd,
                dates_box,
                youtube_status,
            ],
        )
    with gr.Tab('Settings'):
        url_box = gr.Textbox(label='Ollama URL', value=ollama_url)
        model_box = gr.Textbox(label='LLM Model', value=model_name)
        clear_btn = gr.Button('Clear database')
        save_btn = gr.Button('Save settings')
        status = gr.Textbox(label='Status', interactive=False)
        clear_btn.click(
            clear_db,
            [],
            [speaker_dd, status, date_start_dd, date_end_dd, speakers_box, dates_box],
        )
        save_btn.click(save_settings, [url_box, model_box], [status])

if __name__ == '__main__':
    demo.launch()
