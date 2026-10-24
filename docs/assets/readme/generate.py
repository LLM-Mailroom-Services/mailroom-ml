#!/usr/bin/env python3
"""Generate the README artwork for mailroom-ml (constellation design v2).

    python3 docs/assets/readme/generate.py          # rewrite the SVGs beside this file
    python3 docs/assets/readme/generate.py --check  # read-only drift gate (exit 1 on drift)

`_art.py` is the shared kit (identical in every constellation repo); this file
owns only this repository's labels, scene and facts. Facts come from the README
and the committed M9c evaluation: one shared encoder with six heads, 8,192-token
windows with 512 overlap, CPU ONNX (fp32 preferred), document-type accuracy
0.9598 and window ECE 0.0189 on the 323-document held-out test.
Standard library only, deterministic output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _art as art  # noqa: E402  (shared kit lives beside this file)

HERO = {
    'title': 'Mailroom ML',
    'kicker': 'THE MAILROOM CONSTELLATION · THE FAST PATH',
    'tagline': 'A fast, calibrated first look at every document — and a polite handoff to the LLM when it is not sure.',
    'eyebrow': 'MODERNBERT · ONE ENCODER · SIX HEADS · CALIBRATED GATE · CPU ONNX',
    'caption': 'standalone contractor repo · consumes the constellation taxonomy, never redefines it',
    'alt': 'Animated banner: a document is cut into overlapping 8,192-token windows, encoded by ModernBERT, '
           'classified by a document-type head and subclass heads, then a calibrated gate sends it down the CPU '
           'fast path or hands it to the LLM sorter.',
    'tagline_lines': ['A fast, calibrated first look —', 'and a polite handoff to the LLM.'],
    'mobile_cards': [('Window', '8,192 tokens · 512 overlap'), ('Classify', 'one encoder · six heads'),
                     ('Calibrate', 'temperature + OOD probe'), ('Route', 'fast path or LLM sorter')],
    'mobile_note': 'the LLM sorter stays the authority for the messy tail',
}
STATS = [('6', 'heads · one shared encoder', art.SKY), ('8,192', 'token windows · 512 overlap', art.VIOLET),
         ('0.9598', 'doc-type accuracy · 323 held out', art.GREEN), ('0.0189', 'window ECE · calibrated', art.PINK),
         ('fp32', 'ONNX on CPU, preferred', art.AMBER), ('LLM', 'takes the uncertain tail', art.LIME)]
STAGES = [('Normalize', 'clerk-normalized', 'leak-free inputs'), ('Window', '8,192 tokens', '512 overlap'),
          ('Encode', 'ModernBERT', 'shared by all heads'), ('Heads', 'doc_type + 5 subclass', 'plurality merge'),
          ('Calibrate', 'temperature per head', '+ OOD energy probe'), ('Gate', 'p · agreement · margin', 'fast path or LLM')]
SIDE = [('Train', 'local or Modal GPU', 'operator-budgeted'), ('Evaluate', 'held-out test + ECE', 'paired bootstrap compare'),
        ('Export', 'ONNX + parity check', 'fp32 preferred'), ('Serve', 'CPU ONNX', 'Modal app is a fallback'),
        ('Enrich', 'tiered augmentation', 'Tier 1 → Tier 3 gates')]
TERMINAL = [('cmd', 'git clone https://github.com/LLM-Mailroom-Services/mailroom-ml.git'),
            ('cmd', 'cd mailroom-ml && uv sync --extra dev'),
            ('note', '# pytest + ruff · no torch, modal or onnxruntime required'),
            ('cmd', 'uv run pytest -m "not fullcorpus"'),
            ('note', '# deploy-layer tests importorskip the heavy extras'),
            ('cmd', 'uv run ruff check src tests training deploy'),
            ('cmd', 'uv run python training/eval/eval_modernbert.py --help'),
            ('note', '# calibration, selective-risk and routing-threshold options')]

TUI_GAUGES = [('doc-type acc.', '0.9598', 0.9598, art.GREEN), ('subclass acc.', '0.7387', 0.7387, art.SKY),
              ('window ECE', '0.0189', 0.0189, art.PINK), ('gate accept.', '0.3901', 0.3901, art.AMBER)]
TUI_LOG = [('ok', 'doc-type head: 310 / 323 documents'), ('info', 'subclass: 229 / 310 scorable documents'),
           ('info', 'window ECE 0.0189 on the held-out test'), ('warn', 'gate acceptance is an upper bound'),
           ('warn', 'production guards not applied here'), ('info', 'ONNX fp32 preferred · CPU serving')]
TUI_STATUS = 'M9c evaluation · n = 323 held-out documents · lower ECE is better'


def scene() -> str:
    """A document cut into windows, encoded, headed, then gated: fast path or LLM."""
    y0 = 306
    out = [art.text(80, 246, 'ONE DOCUMENT', 11, '#8b9bb4', 700, spacing=2)]
    for k in range(6):
        w = 220 if k % 3 else 190
        out.append(f'<rect x="84" y="{262 + k * 14}" width="{w}" height="8" rx="4" fill="#334155"/>')
    out.append('<rect x="78" y="256" width="240" height="94" rx="10" fill="none" stroke="#475569"/>')
    for k, (dx, color) in enumerate(((0, art.SKY), (46, art.VIOLET))):
        out.append(f'<rect class="motion" x="{82 + dx}" y="258" width="130" height="90" rx="8" fill="{color}" opacity=".22" '
                   f'stroke="{color}" stroke-width="2"><animate attributeName="x" values="{82 + dx};{82 + dx + 70};{82 + dx}" '
                   'dur="4s" repeatCount="indefinite"/></rect>')
    out.append(art.text(198, 372, '8,192-token windows · 512 overlap', 11, art.DIM, anchor='middle'))
    out.append(art.arrow(326, y0, 384, y0, art.MUTED))
    out.append(f'<rect x="388" y="262" width="140" height="88" rx="14" fill="{art.PANEL}" stroke="{art.VIOLET}" stroke-width="2"/>')
    out.append(art.text(458, 298, 'ModernBERT', 16, art.INK, 700, 'middle'))
    out.append(art.text(458, 320, 'shared encoder', 12, art.MUTED, anchor='middle'))
    out.append(art.arrow(536, y0, 584, y0, art.MUTED))
    heads = ['doc_type', 'contract', 'merger_agreement', 'corporate_record', 'correspondence', 'insurance_claim']
    for k, head in enumerate(heads):
        y = 248 + k * 20
        color = art.PINK if k == 0 else art.SKY
        out.append(f'<rect x="590" y="{y}" width="160" height="17" rx="6" fill="{art.PANEL}" stroke="{color}" stroke-opacity=".8"/>')
        out.append(art.text(670, y + 12.5, head, 11, art.INK, 600, 'middle', family=art.MONO))
    out.append(art.arrow(758, y0, 800, y0, art.MUTED))
    out.append(f'<rect x="804" y="262" width="170" height="88" rx="14" fill="{art.PANEL}" stroke="{art.AMBER}" stroke-width="2"/>')
    out.append(art.text(889, 296, 'calibrated gate', 15, art.INK, 700, 'middle'))
    out.append(art.text(889, 318, 'S = p·agreement·margin', 11, art.MUTED, anchor='middle', family=art.MONO))
    out.append(art.text(889, 338, 'OOD flag fails the fast path', 11, art.DIM, anchor='middle'))
    out.append(f'<path id="fast" d="M974 284 C1000 284 1004 266 1030 266" fill="none" stroke="{art.GREEN}" stroke-width="2.2"/>')
    out.append(f'<path id="slow" d="M974 328 C1000 328 1004 346 1030 346" fill="none" stroke="{art.AMBER}" stroke-width="2.2" stroke-dasharray="6 5"/>')
    out.append(f'<rect x="1030" y="246" width="130" height="40" rx="12" fill="{art.PANEL}" stroke="{art.GREEN}" stroke-width="2"/>')
    out.append(art.text(1095, 271, 'CPU fast path', 14, '#86efac', 700, 'middle'))
    out.append(f'<rect x="1030" y="326" width="130" height="40" rx="12" fill="{art.PANEL}" stroke="{art.AMBER}" stroke-width="2" stroke-dasharray="7 5"/>')
    out.append(art.text(1095, 351, 'LLM sorter', 14, '#fcd34d', 700, 'middle'))
    out.append(f'<path id="run" d="M326 {y0} H800 L974 284" fill="none"/>')
    out.append(f'<path id="run2" d="M326 {y0} H800 L974 328" fill="none"/>')
    out.append(art.particle('run', 3.2, 0, '#86efac'))
    out.append(art.particle('run2', 3.2, 1.6, '#fcd34d'))
    return '\n'.join(out)


def build() -> dict[str, str]:
    """Return every generated asset keyed by file name."""
    flow_desc = ('Inputs are clerk-normalized, windowed, encoded by ModernBERT, scored by six heads, calibrated and '
                 'gated; around the model sit training, evaluation, ONNX export, serving and enrichment.')
    return {
        'hero.svg': art.hero(HERO, scene),
        'hero-mobile.svg': art.hero(HERO, mobile=True),
        'stats.svg': art.stats(STATS, 'mailroom-ml by the numbers'),
        'stats-mobile.svg': art.stats(STATS, 'mailroom-ml by the numbers', mobile=True),
        'flow.svg': art.flow('From raw text to a routing decision', STAGES, SIDE, 'Around the model', desc=flow_desc,
                             side_color=art.PINK),
        'flow-mobile.svg': art.flow('From raw text to a routing decision', STAGES, SIDE, 'Around the model',
                                    mobile=True, desc=flow_desc, side_color=art.PINK),
        'constellation.svg': art.constellation('mailroom-ml'),
        'constellation-mobile.svg': art.constellation('mailroom-ml', mobile=True),
        'tui.svg': art.tui('~/mailroom-ml — evaluation dashboard', TUI_GAUGES, TUI_LOG, TUI_STATUS,
                           'Dashboard of the committed M9c evaluation: document-type accuracy 0.9598, conditional subclass accuracy 0.7387, window ECE 0.0189 and an upper-bound gate acceptance of 0.3901 on 323 held-out documents.', left='M9C EVIDENCE', right='NOTES'),
        'terminal.svg': art.terminal('~/mailroom-ml — zsh', TERMINAL,
                                     'Animated terminal: clone, sync the dev extra, run the core tests and the linter, '
                                     'and list the evaluation options.'),
    }


def main() -> None:
    """Write the assets, or compare them byte-for-byte with --check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='fail on drift without writing')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    drift = []
    for name, content in build().items():
        path = base / name
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8') != content:
                drift.append(name)
        else:
            path.write_text(content, encoding='utf-8')
    if drift:
        parser.exit(1, 'README asset drift: ' + ', '.join(drift) + '\n')
    print('README assets match' if args.check else 'README assets written')


if __name__ == '__main__':
    main()
