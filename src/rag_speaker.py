import argparse
import json
import os
import re
import zipfile
from collections import Counter
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse
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


_DATE_PATTERNS = [
    re.compile(
        r"\b\d{4}\s*m\.\s*[A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž]+(?:\s+[A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž]+)*\s+\d{1,2}\s*d\.?",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"),
]


def extract_document_date(text: str, max_lines: int = 40) -> Optional[str]:
    """Return the first date-like string found near the start of ``text``."""

    lines = text.splitlines()[:max_lines]
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        for pattern in _DATE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            found = match.group(0).strip().rstrip(',;')
            if pattern is _DATE_PATTERNS[1]:
                parts = re.split(r"[-/.]", found)
                year, month, day = parts[0], int(parts[1]), int(parts[2])
                return f"{year}-{month:02d}-{day:02d}"
            return ' '.join(found.split())
    return None


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
    """Call the Ollama API and return the concatenated response text.

    Ollama streams multiple JSON objects separated by newlines.  The previous
    implementation attempted to ``json.loads`` the entire payload at once and
    failed with ``Extra data`` errors.  Here we parse each line individually and
    join the ``response`` fragments.
    """

    try:
        data = json.dumps({'model': model, 'prompt': prompt}).encode('utf-8')
        req = Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
        )
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8')

        parts: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'response' in obj:
                parts.append(obj['response'])
            if obj.get('done'):
                break
        return ''.join(parts)
    except Exception as exc:
        return f"[Ollama error: {exc}]"


def extract_youtube_video_id(url: str) -> str:
    """Return the 11-character video ID from a YouTube URL or ID string."""

    candidate = url.strip()
    if not candidate:
        raise ValueError('YouTube URL is empty')

    parsed = urlparse(candidate)
    if parsed.scheme and parsed.netloc:
        host = parsed.netloc.lower()
        path = parsed.path
        if host in {'youtu.be', 'www.youtu.be'}:
            video_id = path.lstrip('/')
        elif 'youtube.com' in host:
            if path == '/watch':
                query = parse_qs(parsed.query)
                ids = query.get('v', [])
                video_id = ids[0] if ids else ''
            elif path.startswith('/shorts/') or path.startswith('/embed/'):
                parts = path.split('/')
                video_id = parts[2] if len(parts) > 2 else ''
            else:
                video_id = ''
        else:
            video_id = ''
    else:
        video_id = candidate

    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id or ''):
        raise ValueError('Unable to extract video ID from URL')
    return video_id


def fetch_youtube_metadata(url: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Retrieve the video ID, title and optional date extracted from the title."""

    video_id = extract_youtube_video_id(url)
    title = video_id
    date_label: Optional[str] = None
    warning: Optional[str] = None
    oembed_url = (
        'https://www.youtube.com/oembed?format=json&url='
        + quote(url, safe=':/?=&')
    )

    try:
        req = Request(
            oembed_url,
            headers={'User-Agent': 'Mozilla/5.0'},
        )
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        fetched_title = data.get('title')
        if fetched_title:
            title = fetched_title.strip()
        date_label = extract_document_date(title)
    except Exception as exc:
        warning = f'Nepavyko gauti vaizdo įrašo pavadinimo: {exc}'

    return video_id, title, date_label, warning


_DEFAULT_TRANSCRIPT_LANGS = ['lt', 'lt-LT', 'en', 'en-US']


def fetch_youtube_transcript(video_id: str, languages: Optional[List[str]] = None) -> List[str]:
    """Download transcript segments for ``video_id`` using YouTube's API."""

    try:
        from youtube_transcript_api import (  # type: ignore
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeTranscriptApi,
        )
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError('youtube-transcript-api package is required') from exc

    langs = languages or _DEFAULT_TRANSCRIPT_LANGS
    api_instance: Optional[Any] = None

    def _call_api(method: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal api_instance

        attr = getattr(YouTubeTranscriptApi, method, None)
        if callable(attr):
            try:
                return attr(*args, **kwargs)
            except TypeError as exc:
                if 'self' not in str(exc):
                    raise

        if api_instance is None:
            try:
                api_instance = YouTubeTranscriptApi()  # type: ignore[call-arg]
            except TypeError as exc:
                raise RuntimeError(
                    'Nesuderinama youtube-transcript-api versija: '
                    'nepavyko inicijuoti API kliento.',
                ) from exc

        inst_attr = getattr(api_instance, method, None)
        if callable(inst_attr):
            return inst_attr(*args, **kwargs)

        raise RuntimeError(
            'Nesuderinama youtube-transcript-api versija: '
            f"nerastas metodas '{method}'.",
        )

    try:
        entries = _call_api('get_transcript', video_id, languages=langs)
    except TranscriptsDisabled as exc:
        raise ValueError('Transkripsijos yra išjungtos šiam vaizdo įrašui') from exc
    except VideoUnavailable as exc:
        raise ValueError('Vaizdo įrašas nepasiekiamas arba ištrintas') from exc
    except RuntimeError as exc:
        raise RuntimeError('Nepavyko nuskaityti YouTube transkripcijos: ' + str(exc)) from exc
    except NoTranscriptFound:
        try:
            transcript_list = _call_api('list_transcripts', video_id)
            transcript = transcript_list.find_transcript(langs)
            entries = transcript.fetch()
        except Exception as exc:  # pragma: no cover - fallback path
            raise ValueError('Transkripcija nerasta šiam vaizdo įrašui') from exc

    segments: List[str] = []
    for item in entries:
        text = item.get('text', '').replace('\n', ' ').strip()
        if text:
            segments.append(text)
    return segments


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
    context = '\n\n'.join(segments)
    prompt = (
        f"Use the following context from {args.speaker} to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {args.query}\nAnswer:"
    )

    if args.dry_run:
        print('Retrieved context:\n' + prompt)
    else:
        print(call_ollama(args.model, prompt))


if __name__ == '__main__':
    main()
