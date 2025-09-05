import argparse
import json
import os
import re
import zipfile
from collections import Counter
from math import sqrt
from typing import Dict, List
from urllib.request import Request, urlopen

import xml.etree.ElementTree as ET


def load_text(path: str) -> str:
    """Load text content from a .txt or .docx file using only stdlib."""
    if path.endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    if path.endswith('.docx'):
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        paragraphs = []
        for p in tree.iter(f'{ns}p'):
            texts = [node.text for node in p.iter(f'{ns}t') if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        return '\n'.join(paragraphs)
    raise ValueError(f"Unsupported file type: {path}")


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def cosine_similarity(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[x] * b[x] for x in common)
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def parse_speakers(text: str) -> Dict[str, List[str]]:
    """Split transcript text into segments grouped by speaker.

    The function recognises lines where a speaker name (all caps) is
    followed by either a colon or a period, for example::

        PIRMININKAS: Sveiki.
        V. ALEKNAVIČIENĖ (LSDPF*). Labas.

    ``parse_speakers`` is careful not to split inside initials such as
    ``V. ALEKNAVIČIENĖ`` by selecting the first ``:`` or ``.`` whose
    succeeding text does not start with another all-caps word. Lines that
    do not match this pattern are appended to the current speaker's last
    segment.
    """

    speakers: Dict[str, List[str]] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        found = False
        for match in re.finditer(r"[\.:]", line):
            speaker = line[:match.start()].strip()
            remainder = line[match.end():].strip()
            if speaker and speaker.isupper() and not speaker[0].isdigit():
                words = remainder.split()
                if words and words[0].isupper():
                    # Likely an initial, keep searching
                    continue
                current = speaker
                speakers.setdefault(current, []).append(remainder)
                found = True
                break
        if not found and current:
            speakers[current][-1] += ' ' + line
    return speakers


class SpeakerRAG:
    def __init__(self):
        self.corpora: Dict[str, List[str]] = {}
        self.tokens: Dict[str, List[Counter]] = {}

    def add_speaker(self, speaker: str, texts: List[str]) -> None:
        self.corpora[speaker] = texts
        self.tokens[speaker] = [Counter(tokenize(t)) for t in texts]

    def retrieve(self, speaker: str, query: str, top_k: int = 3) -> List[str]:
        q_tokens = Counter(tokenize(query))
        scores = [cosine_similarity(q_tokens, t) for t in self.tokens[speaker]]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.corpora[speaker][i] for i in ranked[:top_k]]


def call_ollama(model: str, prompt: str, url: str = 'http://localhost:11434/api/generate') -> str:
    try:
        data = json.dumps({'model': model, 'prompt': prompt}).encode('utf-8')
        req = Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
        )
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        return result.get('response', '')
    except Exception as exc:
        return f"[Ollama error: {exc}]"


def main() -> None:
    parser = argparse.ArgumentParser(description='Speaker-based RAG using Ollama')
    parser.add_argument('docs', help='Directory with documents')
    parser.add_argument('--speaker', required=True, help='Speaker to query')
    parser.add_argument('--query', required=True, help='Question to ask')
    parser.add_argument('--model', default='phi4', help='Ollama model name')
    parser.add_argument('--top-k', type=int, default=3, help='Number of context chunks')
    parser.add_argument('--dry-run', action='store_true', help='Do not call LLM')
    args = parser.parse_args()

    rag = SpeakerRAG()
    all_speakers: Dict[str, List[str]] = {}

    for fname in os.listdir(args.docs):
        if not (fname.endswith('.txt') or fname.endswith('.docx')):
            continue
        text = load_text(os.path.join(args.docs, fname))
        parsed = parse_speakers(text)
        for sp, segs in parsed.items():
            all_speakers.setdefault(sp, []).extend(segs)

    for sp, segs in all_speakers.items():
        rag.add_speaker(sp, segs)

    if args.speaker not in rag.corpora:
        raise ValueError(f"Speaker '{args.speaker}' not found")

    segments = rag.retrieve(args.speaker, args.query, args.top_k)
    prompt = '\n\n'.join(segments) + f"\n\nQuestion: {args.query}"

    if args.dry_run:
        print('Retrieved context:\n' + prompt)
    else:
        print(call_ollama(args.model, prompt))


if __name__ == '__main__':
    main()
