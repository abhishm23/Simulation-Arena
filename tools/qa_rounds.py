"""Round-tracking helper for SimArena_QA_Test_Deliverable.xlsx.

Used by BOTH the developer agent and the test agent. It is the only supported
way to write round results, and it guarantees:

  * existing cells / rows / sheets are NEVER modified or deleted
  * each round N adds its own columns  R{N} Dev Comment / R{N} Dev Status /
    R{N} Dev Evidence / R{N} Test Verdict / R{N} Regression Findings /
    R{N} Test Notes / R{N} Test Evidence
  * each round gets a "Round N Log" sheet

Usage
-----
  python qa_rounds.py ensure N
  python qa_rounds.py open N [--out file.json]
  python qa_rounds.py dev  N --data records.json
  python qa_rounds.py test N --data records.json
  python qa_rounds.py addreg N --data records.json
  python qa_rounds.py log  N --data pairs.json
  python qa_rounds.py summary N

records.json layout
-------------------
  dev   : [{"id": "DF-08", "comment": "...", "status": "...", "evidence": "..."}]
  test  : [{"id": "DF-08", "verdict": "Pass|Fail|Blocked|Not retested",
            "regression": "...", "notes": "...", "evidence": "..."}]
  addreg: [{"id": "NEW-08", "severity": "Critical|High|Medium|Low",
            "summary": "...", "observed": "...", "steps": "...",
            "impact": "...", "evidence": "..."}]
  log   : [["Field", "Value"], ...]

Only the keys you supply are written; omitted keys leave the cell untouched.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

WB = r'C:/Simulation arena/SimArena_QA_Test_Deliverable.xlsx'
BACKUP_DIR = r'C:/SimQA/deliverable/backups'

# sheet -> (id column header, header row number)
TRACKED = {
    'Retest Results': ('Issue ID', 1),
    'New Regressions': ('ID', 1),
}
DEV_COLS = ['R{n} Dev Comment', 'R{n} Dev Status', 'R{n} Dev Evidence']
TEST_COLS = ['R{n} Test Verdict', 'R{n} Regression Findings',
             'R{n} Test Notes', 'R{n} Test Evidence']
WIDTHS = {'Dev Comment': 62, 'Dev Status': 13, 'Dev Evidence': 44,
          'Test Verdict': 16, 'Regression Findings': 34,
          'Test Notes': 78, 'Test Evidence': 44}

HDR_FILL = PatternFill('solid', fgColor='1F3864')
DEV_HDR_FILL = PatternFill('solid', fgColor='2E5C8A')
TEST_HDR_FILL = PatternFill('solid', fgColor='375623')
HDR_FONT = Font(color='FFFFFF', bold=True, size=11)
WRAP = Alignment(wrap_text=True, vertical='top')
THIN = Border(*[Side(style='thin', color='D0D0D0')] * 4)
VERDICT_FILL = {
    'pass': 'C6EFCE', 'verified': 'C6EFCE',
    'fail': 'FFC7CE',
    'blocked': 'FFEB9C',
    'not retested': 'D9D9D9',
}
STATUS_FILL = {
    'fixed': 'C6EFCE', 'verified': 'C6EFCE', 'closed': 'C6EFCE',
    'won t fix': 'D9D9D9', 'wont fix': 'D9D9D9', 'deferred': 'D9D9D9',
    'cannot reproduce': 'FFEB9C', 'needs clarification': 'FFEB9C',
}
SEV_FILL = {'Critical': 'FFC7CE', 'High': 'FFD9B3',
            'Medium': 'FFEB9C', 'Low': 'DDEBF7'}

MAX_CELL = 30000


def _txt(v):
    if v is None:
        return ''
    s = str(v)
    return s[:MAX_CELL] + ' …[truncated]' if len(s) > MAX_CELL else s


def _last_used_col(ws):
    last = 0
    for row in ws.iter_rows():
        for c in row:
            if c.value is not None and str(c.value).strip() != '':
                if c.column > last:
                    last = c.column
    return last


def _header_map(ws, hdr_row):
    m = {}
    for c in range(1, _last_used_col(ws) + 1):
        v = ws.cell(row=hdr_row, column=c).value
        if v is not None and str(v).strip():
            m[str(v).strip()] = c
    return m


def _backup(tag):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, f'_backup_{tag}.xlsx')
    shutil.copy2(WB, dst)
    return dst


def ensure_round(wb, n):
    created = []
    for name, (id_hdr, hdr_row) in TRACKED.items():
        if name not in wb.sheetnames:
            continue
        ws = wb[name]
        hmap = _header_map(ws, hdr_row)
        next_c = _last_used_col(ws) + 1
        for tmpl in DEV_COLS + TEST_COLS:
            label = tmpl.format(n=n)
            if label in hmap:
                continue
            cell = ws.cell(row=hdr_row, column=next_c, value=label)
            cell.fill = DEV_HDR_FILL if tmpl in DEV_COLS else TEST_HDR_FILL
            cell.font = HDR_FONT
            cell.alignment = Alignment(wrap_text=True, vertical='center')
            cell.border = THIN
            key = label.split(' ', 1)[1]
            ws.column_dimensions[get_column_letter(next_c)].width = WIDTHS.get(key, 40)
            hmap[label] = next_c
            created.append(f'{name}!{label}')
            next_c += 1
        if ws.freeze_panes is None:
            ws.freeze_panes = f'A{hdr_row + 1}'

    logname = f'Round {n} Log'
    if logname not in wb.sheetnames:
        ws = wb.create_sheet(logname)
        ws['A1'] = f'Round {n} - Developer / Test agent log'
        ws['A1'].font = Font(bold=True, size=13, color='1F3864')
        ws['A2'] = f'Started {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        ws['A2'].font = Font(italic=True, size=10)
        ws.column_dimensions['A'].width = 34
        ws.column_dimensions['B'].width = 118
        created.append(logname)
    return created


def _id_to_sheet(wb):
    m = {}
    ws = wb['Retest Results']
    idc = _header_map(ws, 1).get('Issue ID', 1)
    shc = _header_map(ws, 1).get('Sheet', 2)
    for r in range(2, ws.max_row + 1):
        i = ws.cell(row=r, column=idc).value
        s = ws.cell(row=r, column=shc).value
        if i:
            m[str(i).strip()] = ('Retest Results', r)
    ws2 = wb['New Regressions']
    idc2 = _header_map(ws2, 1).get('ID', 1)
    for r in range(2, ws2.max_row + 1):
        i = ws2.cell(row=r, column=idc2).value
        if i:
            m[str(i).strip()] = ('New Regressions', r)
    return m


def _verdict_state(v):
    """Return 'pass' | 'open' | 'unknown' for a verdict string."""
    s = _txt(v).strip().lower()
    if not s:
        return 'unknown'
    if 'fail' in s or 'blocked' in s or 'not retested' in s or 'regression' in s:
        return 'open'
    if s.startswith('pass') or s.startswith('verified') or s.startswith('closed'):
        return 'pass'
    return 'unknown'


def open_items(wb, n):
    """Items not yet confirmed closed, with the newest evidence available."""
    idmap = _id_to_sheet(wb)
    out = []
    for name, (id_hdr, hdr_row) in TRACKED.items():
        ws = wb[name]
        hmap = _header_map(ws, hdr_row)
        idc = hmap.get(id_hdr, 1)
        sumc = hmap.get('Issue Summary') or hmap.get('Summary')
        sevc = hmap.get('Severity') or hmap.get('Priority')
        # newest-first list of round columns
        round_v, round_s, round_tn = [], [], []
        for k in range(n, 0, -1):
            c = hmap.get(f'R{k} Test Verdict')
            if c:
                round_v.append(c)
            c = hmap.get(f'R{k} Dev Status')
            if c:
                round_s.append(c)
            c = hmap.get(f'R{k} Test Notes')
            if c:
                round_tn.append(c)
        base_v = hmap.get('Retest Result')  # only on Retest Results
        for r in range(hdr_row + 1, ws.max_row + 1):
            i = ws.cell(row=r, column=idc).value
            if not i or not str(i).strip():
                continue
            verdict = ''
            for c in round_v:
                v = ws.cell(row=r, column=c).value
                if v is not None and str(v).strip():
                    verdict = str(v)
                    break
            base = ws.cell(row=r, column=base_v).value if base_v else None
            state = _verdict_state(verdict) if verdict else (
                _verdict_state(base) if base else 'unknown')
            if state == 'pass':
                continue
            devstat = ''
            for c in round_s:
                v = ws.cell(row=r, column=c).value
                if v is not None and str(v).strip():
                    devstat = str(v)
                    break
            notes = ''
            for c in round_tn:
                v = ws.cell(row=r, column=c).value
                if v is not None and str(v).strip():
                    notes = str(v)
                    break
            out.append({
                'id': str(i).strip(),
                'sheet': name,
                'row': r,
                'summary': _txt(ws.cell(row=r, column=sumc).value)[:300] if sumc else '',
                'severity': _txt(ws.cell(row=r, column=sevc).value) if sevc else '',
                'last_verdict': verdict or _txt(base),
                'last_dev_status': devstat,
                'last_test_notes': notes[:400],
                'state': state,
            })
    return out


def write_records(wb, n, records, kind):
    idmap = _id_to_sheet(wb)
    wrote, missing = [], []
    for rec in records:
        key = str(rec.get('id', '')).strip()
        if key not in idmap:
            missing.append(key)
            continue
        sheet, row = idmap[key]
        ws = wb[sheet]
        hmap = _header_map(ws, TRACKED[sheet][1])
        if kind == 'dev':
            fields = [('comment', f'R{n} Dev Comment'), ('status', f'R{n} Dev Status'),
                      ('evidence', f'R{n} Dev Evidence')]
        else:
            fields = [('verdict', f'R{n} Test Verdict'),
                      ('regression', f'R{n} Regression Findings'),
                      ('notes', f'R{n} Test Notes'),
                      ('evidence', f'R{n} Test Evidence')]
        for rk, label in fields:
            if rk not in rec or rec[rk] is None:
                continue
            col = hmap.get(label)
            if not col:
                continue
            cell = ws.cell(row=row, column=col, value=_txt(rec[rk]))
            cell.alignment = WRAP
            cell.border = THIN
            if rk == 'verdict':
                cell.font = Font(bold=True)
                key2 = _txt(rec[rk]).strip().lower()
                for k, colr in VERDICT_FILL.items():
                    if key2.startswith(k):
                        cell.fill = PatternFill('solid', fgColor=colr)
                        break
            if rk == 'status':
                cell.font = Font(bold=True)
                key2 = _txt(rec[rk]).strip().lower().replace("'", "")
                for k, colr in STATUS_FILL.items():
                    if key2.startswith(k.replace("'", "")):
                        cell.fill = PatternFill('solid', fgColor=colr)
                        break
        wrote.append(key)
    return wrote, missing


def add_regressions(wb, n, records):
    ws = wb['New Regressions']
    hmap = _header_map(ws, 1)
    cols = ['ID', 'Severity', 'Summary', 'What was observed', 'Reproduction steps',
            'Impact', 'Evidence']
    added = []
    for rec in records:
        row = ws.max_row + 1
        for i, c in enumerate(cols, start=1):
            key = ['id', 'severity', 'summary', 'observed', 'steps', 'impact', 'evidence'][i - 1]
            cell = ws.cell(row=row, column=i, value=_txt(rec.get(key, '')))
            cell.alignment = WRAP
            cell.border = THIN
        sev = _txt(rec.get('severity', ''))
        if sev in SEV_FILL:
            ws.cell(row=row, column=2).fill = PatternFill('solid', fgColor=SEV_FILL[sev])
            ws.cell(row=row, column=2).font = Font(bold=True)
        vcol = hmap.get(f'R{n} Test Verdict')
        if vcol:
            c = ws.cell(row=row, column=vcol, value='New regression - open')
            c.font = Font(bold=True)
            c.fill = PatternFill('solid', fgColor='FFC7CE')
            c.alignment = WRAP
            c.border = THIN
        ncol = hmap.get(f'R{n} Test Notes')
        if ncol and rec.get('notes'):
            c = ws.cell(row=row, column=ncol, value=_txt(rec['notes']))
            c.alignment = WRAP
            c.border = THIN
        added.append(str(rec.get('id', '')))
    return added


def write_log(wb, n, pairs):
    ws = wb[f'Round {n} Log']
    row = max(ws.max_row + 1, 3)
    for field, value in pairs:
        ws.cell(row=row, column=1, value=_txt(field)).font = Font(bold=True)
        c = ws.cell(row=row, column=2, value=_txt(value))
        c.alignment = WRAP
        row += 1
    return len(pairs)


def summary(wb, n):
    items = open_items(wb, n)
    total = 0
    for name, (id_hdr, hdr_row) in TRACKED.items():
        ws = wb[name]
        idc = _header_map(ws, hdr_row).get(id_hdr, 1)
        for r in range(hdr_row + 1, ws.max_row + 1):
            if ws.cell(row=r, column=idc).value:
                total += 1
    return {'round': n, 'tracked_items': total, 'open_items': len(items),
            'closed_items': total - len(items),
            'open_ids': [i['id'] for i in items]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['ensure', 'open', 'dev', 'test', 'addreg',
                                    'log', 'summary'])
    ap.add_argument('round', type=int, nargs='?', default=None)
    ap.add_argument('--data', help='path to JSON records file')
    ap.add_argument('--out', help='write JSON output here')
    a = ap.parse_args()

    wb = load_workbook(WB)
    n = a.round or 1

    if a.cmd == 'ensure':
        created = ensure_round(wb, n)
        _backup(f'pre_round{n}')
        wb.save(WB)
        print(json.dumps({'created': created}, indent=2))

    elif a.cmd == 'open':
        items = open_items(wb, n)
        payload = json.dumps(items, indent=2)
        if a.out:
            open(a.out, 'w', encoding='utf-8').write(payload)
            print(f'wrote {len(items)} open items -> {a.out}')
        else:
            print(payload)

    elif a.cmd in ('dev', 'test'):
        recs = json.load(open(a.data, encoding='utf-8'))
        ensure_round(wb, n)
        wrote, missing = write_records(wb, n, recs, a.cmd)
        _backup(f'pre_round{n}')
        wb.save(WB)
        print(json.dumps({'wrote': wrote, 'missing_ids': missing}, indent=2))

    elif a.cmd == 'addreg':
        recs = json.load(open(a.data, encoding='utf-8'))
        ensure_round(wb, n)
        added = add_regressions(wb, n, recs)
        _backup(f'pre_round{n}')
        wb.save(WB)
        print(json.dumps({'added': added}, indent=2))

    elif a.cmd == 'log':
        pairs = json.load(open(a.data, encoding='utf-8'))
        ensure_round(wb, n)
        k = write_log(wb, n, pairs)
        _backup(f'pre_round{n}')
        wb.save(WB)
        print(json.dumps({'logged': k}, indent=2))

    elif a.cmd == 'summary':
        print(json.dumps(summary(wb, n), indent=2))


if __name__ == '__main__':
    main()
