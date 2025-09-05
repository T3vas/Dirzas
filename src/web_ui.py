from typing import List, Tuple

import gradio as gr

from rag_speaker import SpeakerRAG, load_text, parse_speakers, call_ollama

rag = SpeakerRAG()
ollama_url = 'http://localhost:11434/api/generate'
model_name = 'phi4'

def chat_fn(history: List[Tuple[str, str]], question: str, speaker: str):
    if not speaker:
        history.append((question, 'Please select a speaker'))
        return history, ''
    if speaker not in rag.corpora:
        history.append((question, f"Speaker '{speaker}' not found"))
        return history, ''
    segments = rag.retrieve(speaker, question)
    prompt = '\n\n'.join(segments) + f"\n\nQuestion: {question}"
    answer = call_ollama(model=model_name, prompt=prompt, url=ollama_url)
    history.append((question, answer))
    return history, ''

def upload_fn(files: List[gr.File]) -> Tuple[gr.Dropdown, str]:
    for file in files:
        try:
            text = load_text(file.name)
        except ValueError:
            # Skip files with unsupported extensions
            continue
        parsed = parse_speakers(text)
        for sp, segs in parsed.items():
            existing = rag.corpora.get(sp, [])
            rag.add_speaker(sp, existing + segs)
    speakers = list(rag.corpora.keys())
    return gr.Dropdown.update(choices=speakers), ', '.join(speakers)

def clear_db() -> Tuple[gr.Dropdown, str]:
    rag.corpora.clear()
    rag.tokens.clear()
    return gr.Dropdown.update(choices=[]), 'Database cleared'

def save_settings(url: str, model: str) -> str:
    global ollama_url, model_name
    ollama_url = url
    model_name = model
    return 'Settings saved'

with gr.Blocks() as demo:
    gr.Markdown('# RAG Web UI')
    with gr.Tab('Chat'):
        speaker_dd = gr.Dropdown(label='Speaker', choices=[])
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label='Your question')
        msg.submit(chat_fn, [chatbot, msg, speaker_dd], [chatbot, msg])
    with gr.Tab('Documents'):
        # Gradio 4 does not recognise ``docx`` in ``file_types``. Allow any
        # file to be uploaded and rely on ``load_text`` to filter by
        # extension instead.
        file_input = gr.File(file_types=['file'], file_count='multiple', label='Upload documents')
        upload_btn = gr.Button('Add to database')
        speakers_box = gr.Textbox(label='Known speakers', interactive=False)
        upload_btn.click(upload_fn, [file_input], [speaker_dd, speakers_box])
    with gr.Tab('Settings'):
        url_box = gr.Textbox(label='Ollama URL', value=ollama_url)
        model_box = gr.Textbox(label='LLM Model', value=model_name)
        clear_btn = gr.Button('Clear database')
        save_btn = gr.Button('Save settings')
        status = gr.Textbox(label='Status', interactive=False)
        clear_btn.click(lambda: clear_db(), [], [speaker_dd, status])
        save_btn.click(save_settings, [url_box, model_box], [status])

if __name__ == '__main__':
    demo.launch()
