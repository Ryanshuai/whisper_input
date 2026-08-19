"""Count how often each judged (correct <- heard) pair actually occurred.

Stage 3 decides *whether* a pair is a real misrecognition; this counts it. The
count is the point — it is what makes the table rankable and what a post-hoc
corrector needs to decide whether a rewrite is safe. A pair judged real but seen
once is a guess; one seen twenty times over three months is a fact.

`k` (distinct Claude sessions that used the correct spelling) comes from the
lexicon and says the term is real vocabulary. `heard` counts come from the
dictation log and say how often we failed to produce it. Both are needed: a high
`k` with no misses is a word we say fine, and a miss with no `k` is noise.

    pixi run python build_verified.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/shuai/code/whisper_input')

import asr_context as A          # noqa: E402
import mine_corrections as M     # noqa: E402

OUT = os.path.join(HERE, os.pardir, 'verified_corrections.json')

# Stage 3 verdicts. Keep the rejects — deleting them means the next mining pass
# re-proposes them and someone re-adjudicates from scratch. `why` is for the
# rejects, where the reason is not self-evident from the pair alone.
JUDGED = {
    # --- real misrecognitions, found by the miner -------------------------
    'Claude':       {'heard': ['Cloud', 'cloud', 'CLAW', 'Claw'], 'verdict': 'error'},
    'Claude Code':  {'heard': ['CLAW CODE', 'Claw Code'], 'verdict': 'error'},
    'MLClaw':       {'heard': ['M R Claw', 'ML CLAW', 'ML Claw', 'mclou'], 'verdict': 'error'},
    'e13qh':        {'heard': ['E十三QH', 'e十三qh'], 'verdict': 'error'},
    'H100':         {'heard': ['H一百', 'H 一百'], 'verdict': 'error'},
    'SKU':          {'heard': ['S K U'], 'verdict': 'error'},
    'IoU':          {'heard': ['I O U'], 'verdict': 'error'},
    'CDN':          {'heard': ['C D N'], 'verdict': 'error'},
    'anyware-core': {'heard': ['anywhere core', 'anywhere-core'], 'verdict': 'error'},
    'Orbbec':       {'heard': ['Orbic', 'orbic'], 'verdict': 'error'},
    's000':         {'heard': ['S 零零零'], 'verdict': 'error'},
    'SfM':          {'heard': ['sim'], 'verdict': 'error',
                     'why': 'domain term; "sim" is a real word, so a corrector '
                            'needs context — biasing-only until then'},
    'train-init':   {'heard': ['train init'], 'verdict': 'error'},

    # --- real, seeded by hand: entirely-Chinese renderings the miner cannot
    # localize without a pinyin table (see near_miss()). Sources are the
    # hand-checked probes in context_replay_eval.py.
    'loss':         {'heard': ['拉斯'], 'verdict': 'error', 'source': 'probe'},
    'JSON':         {'heard': ['Jason'], 'verdict': 'error', 'source': 'probe'},
    'YOLO':         {'heard': ['ULO'], 'verdict': 'error', 'source': 'probe'},
    'DETR':         {'heard': ['D E T R'], 'verdict': 'error', 'source': 'probe'},
    'plugin':       {'heard': ['plug in', 'Plug in'], 'verdict': 'error', 'source': 'probe'},
    'sidecar':      {'heard': ['side car'], 'verdict': 'error', 'source': 'probe'},
    '雾凇':          {'heard': ['兀兀兀松', '雾松', '吴松', '五松'], 'verdict': 'error',
                     'source': 'probe'},
    'large language model': {'heard': ['兰舍达格莫特'], 'verdict': 'error', 'source': 'probe'},

    # --- rejected: phonetically close but a different word the user meant ---
    'CPU':      {'heard': ['GPU'], 'verdict': 'not-an-error',
                 'why': 'both are said constantly and mean different things; '
                        'auto-correcting either way corrupts correct text'},
    'mode':     {'heard': ['code'], 'verdict': 'not-an-error', 'why': 'distinct words'},
    'batch':    {'heard': ['Match'], 'verdict': 'not-an-error',
                 'why': 'feature matching is a real topic here'},
    'shot':     {'heard': ['short'], 'verdict': 'not-an-error', 'why': 'distinct words'},
    'location': {'heard': ['motion'], 'verdict': 'not-an-error', 'why': 'distinct words'},
    'never':    {'heard': ['server'], 'verdict': 'not-an-error', 'why': 'distinct words'},
    'skills':   {'heard': ['skill'], 'verdict': 'not-an-error', 'why': 'plural, not an error'},
    'hooks':    {'heard': ['hook'], 'verdict': 'not-an-error', 'why': 'plural, not an error'},
    'train-run': {'heard': ['train'], 'verdict': 'not-an-error',
                  'why': 'truncated compound, not a misrecognition'},
}


def main():
    said = M.transcripts()
    corpus = list(said.values())
    lex = A.Lexicon(os.path.expanduser('~/.local/state/whisper_input/lexicon.json'))

    table = {}
    for term, spec in JUDGED.items():
        counts = {}
        for form in spec['heard']:
            # Case-SENSITIVE on purpose: the forms are listed with the casing the
            # recognizer actually emitted, and 'Cloud'/'cloud' are two separate
            # observations. Matching case-insensitively counts the same
            # occurrence under both entries and doubles the total.
            n = sum(len(re.findall(re.escape(form), utt)) for utt in corpus)
            if n:
                counts[form] = n
        right = sum(len(re.findall(re.escape(term), utt)) for utt in corpus)
        e = lex.terms.get(term) or lex.terms.get(term.lower()) or {}
        entry = {'verdict': spec['verdict'], 'heard': counts,
                 'said_right': right, 'k': e.get('k'), 'claude_wrote': e.get('n')}
        for extra in ('why', 'source'):
            if extra in spec:
                entry[extra] = spec[extra]
        table[term] = entry

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False, indent=1, sort_keys=True)

    errs = {t: e for t, e in table.items() if e['verdict'] == 'error'}
    ranked = sorted(errs.items(), key=lambda kv: -sum(kv[1]['heard'].values()))
    print(f'{len(table)} judged, {len(errs)} real errors -> {OUT}\n')
    print(f'{"correct":22s}{"wrong":>6s}{"right":>7s}  heard as')
    for t, e in ranked:
        wrong = sum(e['heard'].values())
        forms = ', '.join(f'{k}×{v}' for k, v in
                          sorted(e['heard'].items(), key=lambda kv: -kv[1]))
        print(f'{t:22s}{wrong:6d}{e["said_right"]:7d}  {forms or "(none in log)"}')


if __name__ == '__main__':
    main()
