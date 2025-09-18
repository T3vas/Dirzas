from typing import List, Optional, Tuple

import inspect
import json
import re
import unicodedata
from datetime import date as dt_date
from pathlib import Path
from uuid import uuid4

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


_HTML_COMPONENT_KWARGS = {}
try:
    if 'sanitize_html' in inspect.signature(gr.HTML.__init__).parameters:
        _HTML_COMPONENT_KWARGS['sanitize_html'] = False
except (TypeError, ValueError):
    pass


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
        r'(\d{4})\s*m\.\s*([A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž\s\.]*)\s+(\d{1,2})(?:\s*d\.?)?',
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


def _available_iso_dates() -> List[str]:
    iso_dates: List[str] = []
    seen: set[str] = set()
    for label in _sorted_date_labels():
        parsed = _parse_date_label(label)
        if not parsed:
            continue
        iso = parsed.isoformat()
        if iso not in seen:
            seen.add(iso)
            iso_dates.append(iso)
    return iso_dates


def _default_period_values(dates: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not dates:
        return None, None
    return dates[0], dates[-1]


def _label_to_iso(label: str) -> Optional[str]:
    parsed = _parse_date_label(label)
    if parsed is None:
        return None
    return parsed.isoformat()


def _render_calendar_html(
    available_iso: List[str],
    start_value: Optional[str],
    end_value: Optional[str],
) -> str:
    uid = uuid4().hex
    payload = json.dumps(
        {
            'available': available_iso,
            'start': start_value or '',
            'end': end_value or '',
        }
    )
    if not available_iso:
        return (
            "<div class=\"date-calendar-empty\">"
            "<p>Nėra dokumentų su priskirtomis datomis.</p>"
            "<script>(function(){"
            "const findInput=(id)=>{const el=document.getElementById(id);return el?el.querySelector('textarea, input'):null;};"
            "const apply=(id,val)=>{const field=findInput(id);if(!field)return;field.value=val;field.dispatchEvent(new Event('input',{bubbles:true}));field.dispatchEvent(new Event('change',{bubbles:true}));};"
            "apply('date-start-value','');apply('date-end-value','');"
            "})();</script>"
            "</div>"
        )

    month_names = [
        'Sausis',
        'Vasaris',
        'Kovas',
        'Balandis',
        'Gegužė',
        'Birželis',
        'Liepa',
        'Rugpjūtis',
        'Rugsėjis',
        'Spalis',
        'Lapkritis',
        'Gruodis',
    ]
    weekday_headers = ['Pr', 'An', 'Tr', 'Kt', 'Pn', 'Št', 'Sk']

    return f"""
<div class=\"date-calendar-wrapper\" id=\"date-calendar-{uid}\">
  <script type=\"application/json\" id=\"date-calendar-data-{uid}\">{payload}</script>
  <div class=\"date-calendar-legend\">
    <span class=\"legend-swatch legend-available\"></span> – yra dokumentų
  </div>
  <div class=\"date-calendar-panels\">
    <div class=\"date-calendar-panel\" data-role=\"start\">
      <div class=\"date-calendar-title\">Pradžios data</div>
      <div class=\"date-calendar-body\"></div>
    </div>
    <div class=\"date-calendar-panel\" data-role=\"end\">
      <div class=\"date-calendar-title\">Pabaigos data</div>
      <div class=\"date-calendar-body\"></div>
    </div>
  </div>
</div>
<script>
(function() {{
  const styleId = 'date-calendar-style';
  if (!document.getElementById(styleId)) {{
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      .date-calendar-wrapper {{
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 12px;
        background: var(--block-background-fill);
      }}
      .date-calendar-panels {{
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
      }}
      .date-calendar-panel {{
        flex: 1 1 220px;
        min-width: 220px;
      }}
      .date-calendar-title {{
        font-weight: 600;
        margin-bottom: 4px;
      }}
      .date-calendar-legend {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9em;
        margin-bottom: 8px;
      }}
      .legend-swatch {{
        width: 16px;
        height: 16px;
        display: inline-block;
        border-radius: 4px;
        border: 1px solid var(--border-color-primary);
      }}
      .legend-available {{
        background: #d9d9d9;
      }}
      .date-calendar-body {{
        border: 1px solid var(--border-color-primary);
        border-radius: 6px;
        padding: 8px;
      }}
      .calendar-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
      }}
      .calendar-nav {{
        background: var(--button-secondary-background-fill);
        border: 1px solid var(--border-color-primary);
        border-radius: 4px;
        padding: 4px 8px;
        cursor: pointer;
      }}
      .calendar-month {{
        font-weight: 600;
      }}
      .calendar-weekdays, .calendar-grid {{
        display: grid;
        grid-template-columns: repeat(7, 1fr);
      }}
      .calendar-weekdays div {{
        text-align: center;
        font-size: 0.85em;
        padding-bottom: 4px;
        color: var(--body-text-color-subdued);
      }}
      .calendar-day {{
        border: none;
        background: transparent;
        padding: 6px 4px;
        text-align: center;
        cursor: pointer;
        border-radius: 4px;
        transition: background 0.15s ease;
      }}
      .calendar-day:hover:not(.unavailable) {{
        background: var(--border-color-primary);
        color: var(--body-text-color);
      }}
      .calendar-day.unavailable {{
        cursor: not-allowed;
        color: var(--body-text-color-subdued);
        opacity: 0.4;
      }}
      .calendar-day.available {{
        background: #d9d9d9;
        color: var(--body-text-color);
      }}
      .calendar-day.selected {{
        background: var(--button-primary-background-fill);
        color: var(--button-primary-text-color);
      }}
      .calendar-day.adjacent {{
        color: var(--body-text-color-subdued);
        opacity: 0.6;
      }}
    `;
    document.head.appendChild(style);
  }}

  const wrapper = document.getElementById('date-calendar-{uid}');
  if (!wrapper) return;
  const dataElem = document.getElementById('date-calendar-data-{uid}');
  if (!dataElem) return;
  const payload = JSON.parse(dataElem.textContent || '{{}}');
  const available = Array.isArray(payload.available) ? payload.available : [];
  const availableSet = new Set(available);

  const monthNames = {json.dumps(month_names)};
  const weekdayHeaders = {json.dumps(weekday_headers)};

  const findInput = (id) => {{
    const container = document.getElementById(id);
    if (!container) return null;
    return container.querySelector('textarea, input');
  }};

  const applyValue = (id, value) => {{
    const field = findInput(id);
    if (!field) return;
    const next = value || '';
    if (field.value === next) return;
    field.value = next;
    field.dispatchEvent(new Event('input', {{ bubbles: true }}));
    field.dispatchEvent(new Event('change', {{ bubbles: true }}));
  }};

  const parseISO = (value) => {{
    const parts = value.split('-').map(Number);
    if (parts.length !== 3 || parts.some((p) => Number.isNaN(p))) return null;
    return new Date(parts[0], parts[1] - 1, parts[2]);
  }};

  const formatISO = (dateObj) => {{
    const y = dateObj.getFullYear();
    const m = String(dateObj.getMonth() + 1).padStart(2, '0');
    const d = String(dateObj.getDate()).padStart(2, '0');
    return `${{y}}-${{m}}-${{d}}`;
  }};

  const createCalendar = (panel, hiddenId, initialValue) => {{
    let selected = initialValue && availableSet.has(initialValue) ? initialValue : '';
    let viewDate = selected ? parseISO(selected) : null;
    if (!viewDate) {{
      viewDate = available.length ? parseISO(available[0]) : new Date();
    }}

    const render = () => {{
      const body = panel.querySelector('.date-calendar-body');
      if (!body) return;
      body.innerHTML = '';

      const header = document.createElement('div');
      header.className = 'calendar-header';

      const prev = document.createElement('button');
      prev.type = 'button';
      prev.className = 'calendar-nav';
      prev.textContent = '‹';
      prev.addEventListener('click', () => {{
        const month = viewDate.getMonth();
        viewDate.setMonth(month - 1);
        render();
      }});

      const next = document.createElement('button');
      next.type = 'button';
      next.className = 'calendar-nav';
      next.textContent = '›';
      next.addEventListener('click', () => {{
        const month = viewDate.getMonth();
        viewDate.setMonth(month + 1);
        render();
      }});

      const title = document.createElement('div');
      title.className = 'calendar-month';
      title.textContent = `${{monthNames[viewDate.getMonth()]}} ${{viewDate.getFullYear()}}`;

      header.appendChild(prev);
      header.appendChild(title);
      header.appendChild(next);
      body.appendChild(header);

      const weekdayRow = document.createElement('div');
      weekdayRow.className = 'calendar-weekdays';
      weekdayHeaders.forEach((label) => {{
        const cell = document.createElement('div');
        cell.textContent = label;
        weekdayRow.appendChild(cell);
      }});
      body.appendChild(weekdayRow);

      const grid = document.createElement('div');
      grid.className = 'calendar-grid';
      const firstDay = new Date(viewDate.getFullYear(), viewDate.getMonth(), 1);
      const offset = (firstDay.getDay() + 6) % 7;
      const startDate = new Date(viewDate.getFullYear(), viewDate.getMonth(), 1 - offset);

      for (let i = 0; i < 42; i += 1) {{
        const cellDate = new Date(startDate.getFullYear(), startDate.getMonth(), startDate.getDate() + i);
        const iso = formatISO(cellDate);
        const inMonth = cellDate.getMonth() === viewDate.getMonth();
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'calendar-day';
        button.textContent = String(cellDate.getDate());
        if (!inMonth) {{
          button.classList.add('adjacent');
        }}
        if (availableSet.has(iso)) {{
          button.classList.add('available');
        }} else {{
          button.classList.add('unavailable');
          button.disabled = true;
        }}
        if (selected && iso === selected) {{
          button.classList.add('selected');
        }}
        button.addEventListener('click', () => {{
          if (!availableSet.has(iso)) return;
          if (selected === iso) {{
            selected = '';
          }} else {{
            selected = iso;
          }}
          applyValue(hiddenId, selected);
          render();
        }});
        grid.appendChild(button);
      }}

      body.appendChild(grid);
    }};

    applyValue(hiddenId, selected);
    render();
  }};

  const panels = wrapper.querySelectorAll('.date-calendar-panel');
  panels.forEach((panel) => {{
    const role = panel.getAttribute('data-role');
    if (role === 'start') {{
      createCalendar(panel, 'date-start-value', payload.start);
    }} else if (role === 'end') {{
      createCalendar(panel, 'date-end-value', payload.end);
    }}
  }});
}})();
</script>
"""


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
    start_value: str,
    end_value: str,
    include_unknown: bool,
):
    start_value = (start_value or '').strip()
    end_value = (end_value or '').strip()
    available_iso = set(_available_iso_dates())

    if include_unknown:
        if start_value or end_value:
            history.append(
                (
                    question,
                    'Please clear the calendar selection when querying "Unknown date" documents.',
                )
            )
            return history, ''
        if UNKNOWN_DATE_LABEL not in date_rag.corpora:
            history.append((question, 'No documents stored under "Unknown date"'))
            return history, ''
        segments = date_rag.retrieve(UNKNOWN_DATE_LABEL, question)
        if not segments:
            history.append((question, 'No context available for documents with an unknown date'))
            return history, ''
        context = '\n\n'.join(segments)
        prompt = (
            'Use the following context from documents with an unknown date to answer the question.\n\n'
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        answer = call_ollama(model=model_name, prompt=prompt, url=ollama_url)
        history.append((question, answer))
        return history, ''

    if not start_value or not end_value:
        history.append((question, 'Please select both start and end dates'))
        return history, ''

    try:
        start_date = dt_date.fromisoformat(start_value)
        end_date = dt_date.fromisoformat(end_value)
    except ValueError:
        history.append((question, 'Unable to interpret the selected period'))
        return history, ''

    if start_value not in available_iso or end_value not in available_iso:
        history.append((question, 'Please choose dates highlighted in the calendar'))
        return history, ''

    if start_date > end_date:
        history.append((question, 'Period start must not be later than period end'))
        return history, ''

    if start_value == end_value:
        matching_labels = [
            label
            for label in date_rag.corpora
            if (parsed := _parse_date_label(label)) and parsed.isoformat() == start_value
        ]
        matching_labels.sort(key=_date_sort_key)
        if not matching_labels:
            history.append((question, f'No context available for {start_value}'))
            return history, ''
        if len(matching_labels) == 1:
            segments = date_rag.retrieve(matching_labels[0], question)
        else:
            segments = _retrieve_period_segments(question, matching_labels)
        if not segments:
            history.append((question, f'No context available for {start_value}'))
            return history, ''
        descriptor = f'documents dated {start_value}'
        context = '\n\n'.join(segments)
        prompt = (
            f"Use the following context from {descriptor} to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        answer = call_ollama(model=model_name, prompt=prompt, url=ollama_url)
        history.append((question, answer))
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

    period_desc = f"documents dated between {start_value} and {end_value}"
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
    current_start: Optional[str],
    current_end: Optional[str],
) -> Tuple[gr.Dropdown, str, str, str, str, str]:
    if not files:
        speaker_dd, speakers_text, start_value, end_value, calendar_html, dates_text = _current_choices(
            selected_start=current_start,
            selected_end=current_end,
        )
        return (
            speaker_dd,
            speakers_text,
            start_value,
            end_value,
            calendar_html,
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
    speaker_dd, speakers_text, start_value, end_value, calendar_html, dates_text = _current_choices(
        selected_start=current_start,
        selected_end=current_end,
    )
    return (
        speaker_dd,
        speakers_text,
        start_value,
        end_value,
        calendar_html,
        dates_text,
    )


def clear_db() -> Tuple[gr.Dropdown, str, str, str, str, str, str]:
    rag.corpora.clear()
    rag.tokens.clear()
    date_rag.corpora.clear()
    date_rag.tokens.clear()
    speaker_dd, speakers_text, start_value, end_value, calendar_html, dates_text = _current_choices()
    return (
        speaker_dd,
        'Database cleared',
        start_value,
        end_value,
        calendar_html,
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
    iso_dates = _available_iso_dates()
    default_start, default_end = _default_period_values(iso_dates)

    def _select(candidate: Optional[str], default: Optional[str]) -> str:
        if candidate is None:
            return default or ''
        cleaned = candidate.strip()
        if cleaned == '':
            return ''
        if cleaned in iso_dates:
            return cleaned
        return default or ''

    start_value = _select(selected_start, default_start)
    end_value = _select(selected_end, default_end)
    calendar_html = _render_calendar_html(iso_dates, start_value or None, end_value or None)
    return (
        gr.Dropdown(choices=speakers, value=speaker_value),
        ', '.join(speakers),
        start_value,
        end_value,
        calendar_html,
        ', '.join(dates),
    )


def ingest_youtube(url: str, current_start: Optional[str], current_end: Optional[str]):
    url = (url or '').strip()
    if not url:
        dropdowns = _current_choices(
            selected_start=current_start,
            selected_end=current_end,
        )
        return (*dropdowns, 'Įveskite YouTube nuorodą')

    try:
        video_id, title, date_label, warning = fetch_youtube_metadata(url)
    except ValueError as exc:
        dropdowns = _current_choices(
            selected_start=current_start,
            selected_end=current_end,
        )
        return (*dropdowns, str(exc))

    try:
        segments = fetch_youtube_transcript(video_id)
    except Exception as exc:
        dropdowns = _current_choices(
            selected_start=current_start,
            selected_end=current_end,
        )
        return (*dropdowns, str(exc))

    if not segments:
        dropdowns = _current_choices(
            selected_start=current_start,
            selected_end=current_end,
        )
        return (*dropdowns, 'Transkripcija tuščia')

    speaker_label = f'YouTube {video_id}: {title}'
    if speaker_label in rag.corpora:
        selected_date = date_label or UNKNOWN_DATE_LABEL
        iso_value = _label_to_iso(selected_date) if selected_date else None
        dropdowns = _current_choices(
            speaker_label,
            iso_value if iso_value is not None else current_start,
            iso_value if iso_value is not None else current_end,
        )
        message = f"Vaizdo įrašas '{title}' jau pridėtas"
        if warning:
            message += f'. {warning}'
        return (*dropdowns, message)

    rag.add_speaker(speaker_label, segments)
    date_key = date_label or UNKNOWN_DATE_LABEL
    existing_date = date_rag.corpora.get(date_key, [])
    date_rag.add_speaker(date_key, existing_date + segments)

    iso_value = _label_to_iso(date_key) if date_key else None
    dropdowns = _current_choices(
        speaker_label,
        iso_value if iso_value is not None else current_start,
        iso_value if iso_value is not None else current_end,
    )
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
        with gr.Row():
            calendar_html = gr.HTML(
                value=_render_calendar_html([], None, None),
                **_HTML_COMPONENT_KWARGS,
            )
            with gr.Column():
                date_start_value = gr.Textbox(
                    label='Selected start date',
                    interactive=False,
                    lines=1,
                    value='',
                    elem_id='date-start-value',
                )
                date_end_value = gr.Textbox(
                    label='Selected end date',
                    interactive=False,
                    lines=1,
                    value='',
                    elem_id='date-end-value',
                )
                include_unknown_cb = gr.Checkbox(
                    label='Include "Unknown date" documents',
                    value=False,
                )
                gr.Markdown(
                    'Pilki langeliai rodo datas, kurioms jau yra dokumentų. '
                    'Spustelėkite pasirinktą datą dar kartą, kad ją nuimtumėte.'
                )
        date_chatbot = gr.Chatbot(height=400)
        date_msg = gr.Textbox(label='Your question')
        date_msg.submit(
            date_chat_fn,
            [date_chatbot, date_msg, date_start_value, date_end_value, include_unknown_cb],
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
            [file_input, date_start_value, date_end_value],
            [speaker_dd, speakers_box, date_start_value, date_end_value, calendar_html, dates_box],
        )
    with gr.Tab('YouTube'):
        youtube_url = gr.Textbox(label='YouTube URL')
        youtube_btn = gr.Button('Transcribe video')
        youtube_status = gr.Textbox(label='Status', interactive=False)
        youtube_btn.click(
            ingest_youtube,
            [youtube_url, date_start_value, date_end_value],
            [
                speaker_dd,
                speakers_box,
                date_start_value,
                date_end_value,
                calendar_html,
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
            [
                speaker_dd,
                status,
                date_start_value,
                date_end_value,
                calendar_html,
                speakers_box,
                dates_box,
            ],
        )
        save_btn.click(save_settings, [url_box, model_box], [status])

if __name__ == '__main__':
    demo.launch()
