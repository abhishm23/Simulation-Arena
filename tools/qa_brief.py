"""Generate a scoped markdown worklist for one agent in one round.

  python qa_brief.py dev  N [--out file.md]
  python qa_brief.py test N [--out file.md]

The brief contains only what that agent needs: the open items, their newest
verdict / dev status, and (for the tester) what the developer claims this round.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_rounds import WB, TRACKED, _header_map, _txt, ensure_round, open_items  # noqa: E402
from openpyxl import load_workbook  # noqa: E402


def latest(wb, sheet, hdr_row, row, labels):
    """Newest non-empty value among labels (already newest-first)."""
    ws = wb[sheet]
    hmap = _header_map(ws, hdr_row)
    for lab in labels:
        c = hmap.get(lab)
        if c:
            v = ws.cell(row=row, column=c).value
            if v is not None and str(v).strip():
                return str(v)
    return ''


def build(kind, n):
    wb = load_workbook(WB)
    items = open_items(wb, n)
    lines = []
    if kind == 'dev':
        lines.append(f'# Round {n} - DEVELOPER worklist')
        lines.append('')
        lines.append(f'{len(items)} open items. Fix each one, then record your result '
                     'with tools/qa_rounds.py.')
    else:
        lines.append(f'# Round {n} - TEST ENGINEER worklist')
        lines.append('')
        lines.append(f'{len(items)} items to retest through the real UI. Each needs a '
                     'verdict recorded with tools/qa_rounds.py.')
    lines.append('')
    lines.append('| ID | Severity | Last verdict | R{n} dev status | Summary |'.replace('{n}', str(n)))
    lines.append('|---|---|---|---|---|')
    for it in items:
        ws = wb[it['sheet']]
        hdr_row = TRACKED[it['sheet']][1]
        devstat = latest(wb, it['sheet'], hdr_row, it['row'],
                         [f'R{k} Dev Status' for k in range(n, 0, -1)])
        devcom = latest(wb, it['sheet'], hdr_row, it['row'],
                        [f'R{k} Dev Comment' for k in range(n, 0, -1)])
        dev_ev = latest(wb, it['sheet'], hdr_row, it['row'],
                        [f'R{k} Dev Evidence' for k in range(n, 0, -1)])
        sev = it['severity'] or '-'
        summ = it['summary'].replace('|', '/').replace('\n', ' ')[:220]
        lines.append(f"| {it['id']} | {sev} | {it['last_verdict'] or '-'} | "
                     f"{devstat or '-'} | {summ} |")
        if kind == 'test' and devcom:
            lines.append('')
            lines.append(f"**{it['id']} - developer says (round {n}):** "
                         + devcom.replace('\n', ' ')[:700])
            if dev_ev:
                lines.append(f"  - evidence: {dev_ev.replace(chr(10), ' ')[:400]}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['dev', 'test'])
    ap.add_argument('round', type=int)
    ap.add_argument('--out')
    a = ap.parse_args()
    md = build(a.kind, a.round)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, 'w', encoding='utf-8').write(md)
        print('wrote', a.out, len(md), 'chars')
    else:
        print(md)


if __name__ == '__main__':
    main()
