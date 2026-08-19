"""Mine (misheard -> correct) pairs by comparing what I said to what Claude wrote.

The lexicon's two halves answer different questions and neither is the one that
matters. `observe()` records what Claude wrote; `observe_said()` records what our
transcripts produced — but it compares by STRING EQUALITY, and the entire problem
is that the two sides do not look alike. When the recognizer turns `loss` into
「拉斯」, equality can only conclude "loss was never produced". It cannot say what
`loss` came out AS, so it cannot count how often the word is actually wrong, and
it cannot fix anything after the fact.

This joins each dictation transcript to the assistant reply it provoked — the
reply is the clean side, spelling the term correctly, having in effect already
corrected us once — and emits candidate pairs for judging.

Stage 1 (here, deterministic): find utterances where the reply uses a term our
transcript never produced, and carry the transcript along as evidence.
Stage 2 (a model reads the output): decide which are genuine misrecognitions and
what the wrong form was — 拉斯 for loss, H一百 for H100, Jason for JSON.

    pixi run python mine_corrections.py --terms 40 --examples 4
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, '/home/shuai/code/whisper_input')
import asr_context as A   # noqa: E402

REPO = '/home/shuai/code/whisper_input'
HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.expanduser('~/.claude/projects/*/*.jsonl')


def _log_path() -> str:
    """The dictation log, read from config so the two cannot drift apart."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(REPO, 'config.yaml'), encoding='utf-8'))
    return os.path.join(cfg.get('dictation_debug_dir') or '', 'dictation_log.tsv')


def transcripts(path=None):
    """{normalized -> raw} for every dictation output ever logged."""
    out = {}
    for line in open(path or _log_path(), encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 3 or not p[2].startswith('out='):
            continue
        try:
            text = eval(p[2][4:], {'__builtins__': {}})
        except Exception:
            continue
        if isinstance(text, str) and len(text) >= 6:
            out[A._norm_for_match(text)] = text
    return out


def pairs(said):
    """[(transcript, reply)] — every dictated turn joined to Claude's answer.

    Matched by normalized text rather than by timestamp: Claude Code stores the
    pasted transcript with its own trimming, but the text is ours, so the join is
    exact where it lands and simply misses where it does not.
    """
    out = []
    for f in glob.glob(SESSIONS):
        try:
            raw = open(f, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        pending = None
        for line in raw.splitlines():
            if not line.startswith('{'):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            kind = rec.get('type')
            if kind not in ('user', 'assistant'):
                continue
            c = rec.get('message', {}).get('content')
            blocks = [c] if isinstance(c, str) else [
                b.get('text', '') for b in c or []
                if isinstance(b, dict) and b.get('type') == 'text']
            text = ' '.join(t for t in blocks if t).strip()
            if not text or A._META_RE.search(text):
                continue
            if kind == 'user':
                pending = said.get(A._norm_for_match(text))
            elif pending:
                out.append((pending, text))
                pending = None
    return out


def _skeleton(tok: str) -> str:
    return re.sub(r'[^a-z0-9]', '', tok.lower())


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def near_miss(term: str, utt: str):
    """The fragment of `utt` that looks like a mangled `term`, or None.

    Presence/absence is not evidence of anything: Claude's reply mentions `epoch`
    after an utterance of 「跑怎么样了」, and nothing was misheard there — the word
    was simply never spoken. What makes a pair evidence is that the transcript
    contains something *shaped like* the term. Two families, both dependency-free:

      spelled-out Latin  — mclou/MLClaw, plug in/plugin, Jason/JSON, ULO/YOLO,
                           side car/sidecar, D E T R/DETR, I O U/IoU. Caught by
                           edit distance on the alphanumeric skeleton, over
                           single tokens and over 2-3 token windows (the
                           recognizer loves to insert spaces).
      digit-in-Chinese   — H一百/H100, E十三QH/e13qh: the letters survive and the
                           digits come out as 一二三…百千. Caught by rewriting
                           those numerals back and re-testing.

    NOT caught: a term rendered entirely in Chinese characters (拉斯 for loss,
    兀兀兀松 for 雾凇, 兰舍达格莫特 for "large language model"). Those need a
    pinyin table, which is a new dependency — see the note in the module header.
    """
    skel = _skeleton(term)
    if len(skel) < 3:
        return None
    budget = 1 if len(skel) <= 4 else (2 if len(skel) <= 7 else 3)

    words = re.findall(r'[A-Za-z0-9]+|[一-鿿]+', utt)
    for size in (1, 2, 3):
        for i in range(len(words) - size + 1):
            span = words[i:i + size]
            raw = ''.join(span)
            if raw.lower() == term.lower():
                continue          # written exactly right here, nothing to learn
            # Chinese numerals rewritten first, THEN skeletonized: the digits of
            # H100 come back as 一百, which stripping CJK would simply delete.
            for cand in (_skeleton(raw), _skeleton(_cn_digits(raw))):
                if not cand or not re.search(r'[a-z]', cand):
                    continue
                if _edit(cand, skel) <= budget or skel in cand:
                    return ' '.join(span)
    return None


_CN_NUM = {'零': '0', '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
           '六': '6', '七': '7', '八': '8', '九': '9', '十': '1', '百': '00',
           '千': '000'}


def _cn_digits(s: str) -> str:
    return ''.join(_CN_NUM.get(c, c) for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--terms', type=int, default=40)
    ap.add_argument('--examples', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(HERE, os.pardir,
                                                  'correction_candidates.json'))
    a = ap.parse_args()

    said = transcripts()
    joined = pairs(said)
    print(f'{len(said)} transcripts, {len(joined)} joined to a reply', file=sys.stderr)

    lex = A.Lexicon(os.path.expanduser('~/.local/state/whisper_input/lexicon.json'))
    targets = lex.corrections(a.terms, time.time(), speakable=A._speakable)

    cand = {}
    for term in targets:
        tl = term.lower()
        ev = []
        for utt, reply in joined:
            if term not in reply:
                continue
            if tl in {t.lower() for t in A._LATIN_RE.findall(utt)} or term in utt:
                continue    # said correctly here — not evidence of an error
            got = near_miss(term, utt)
            if not got:
                continue    # the word simply was not spoken — see near_miss()
            ev.append({'heard': got, 'utterance': utt})
            if len(ev) >= a.examples:
                break
        if ev:
            e = lex.terms.get(term, {})
            cand[term] = {'k': e.get('k'), 'n': e.get('n'), 'evidence': ev}
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(cand, f, ensure_ascii=False, indent=1)
    print(f'{len(cand)} terms with evidence -> {a.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
