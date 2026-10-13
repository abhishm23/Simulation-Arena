"""Process + remnant-file cleanup for the Simulation Arena QA cycle.

Both the developer agent and the test agent must run this at the END of their
turn, so the next agent starts from a clean machine.

  python qa_cleanup.py kill              # stop app servers, vite, stray browsers
  python qa_cleanup.py scan --root DIR   # list leftover files (report only)
  python qa_cleanup.py tidy --files a,b  # delete exactly the listed paths
  python qa_cleanup.py ps                # show what is currently running

Safety: only processes whose command line matches the app / the QA harness are
touched. The IDE, the agent runtime and the shell are never killed.
"""
import argparse
import json
import os
import subprocess
import sys

PS = 'powershell'
PORTS = [8000, 8123, 5173, 5174]

# cmdline fragments -> process is part of the app / QA harness.
# NOTE: 'simarena' is deliberately NOT a pattern - every venv python call
# carries it in the path, which would kill the agent's own tooling.
KILL_PATTERNS = [
    'app.main',            # uvicorn backend ("python -m app.main")
    'uvicorn',             # backend launched any other way
    'vite',                # frontend dev server
    'qa.js', 'pwtest.js',  # playwright harness scripts
]

# Never kill these: the querying shell matches its own patterns, and the
# agent runtime / IDE must survive.
EXCLUDE_PATTERNS = [
    'Get-CimInstance',     # the powershell running this very query
    'qa_cleanup.py',       # this script
    'qa_rounds.py',        # the round helper
    'workbuddy', 'codebuddy',
    'PortableGit',
]


def _ps(cmd):
    r = subprocess.run([PS, '-NoProfile', '-NonInteractive', '-Command', cmd],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip()


def _procs():
    """One CIM query, parsed in python, self-matches filtered out."""
    out = _ps("Get-CimInstance Win32_Process | "
              "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress -Depth 3")
    if not out:
        return []
    data = json.loads(out)
    if isinstance(data, dict):
        data = [data]
    res = []
    for p in data:
        cl = p.get('CommandLine') or ''
        pid = str(p.get('ProcessId') or '')
        if not pid.isdigit():
            continue
        if any(e.lower() in cl.lower() for e in EXCLUDE_PATTERNS):
            continue
        res.append({'pid': pid, 'name': p.get('Name') or '', 'cmd': cl})
    return res


def _port_owners():
    out = []
    for p in PORTS:
        o = _ps(f"(Get-NetTCPConnection -LocalPort {p} -State Listen -ErrorAction SilentlyContinue "
                f"| Select-Object -ExpandProperty OwningProcess) -join ','")
        if not o:
            continue
        for pid in o.split(','):
            pid = pid.strip()
            if pid.isdigit():
                nm = _ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).ProcessName")
                out.append({'kind': f'port:{p}', 'pid': pid, 'name': nm or '?'})
    return out


def list_running():
    me = os.getpid()
    out = list(_port_owners())
    for p in _procs():
        cl = p['cmd'].lower()
        nm = (p['name'] or '').lower()
        hit = None
        for pat in KILL_PATTERNS:
            if pat.lower() in cl:
                hit = pat
                break
        # playwright browser leftovers only - never the user's own Chrome
        if hit is None and (nm == 'headless_shell.exe' or 'ms-playwright' in cl):
            hit = 'browser'
        if hit is None:
            continue
        if int(p['pid']) <= 4 or int(p['pid']) == me:
            continue
        pm = _ps(f"(Get-CimInstance Win32_Process -ProcessId {p['pid']} "
                 f"-ErrorAction SilentlyContinue).ParentProcessId")
        if pm.strip().isdigit() and int(pm.strip()) == me:
            continue
        out.append({'kind': f'cmd:{hit}', 'pid': p['pid'], 'name': p['name'],
                    'cmd': p['cmd'][:160]})
    seen, uniq = set(), []
    for x in out:
        if x['pid'] in seen:
            continue
        seen.add(x['pid'])
        uniq.append(x)
    return uniq


def kill_all():
    killed, failed = [], []
    for x in list_running():
        try:
            r = subprocess.run([PS, '-NoProfile', '-NonInteractive', '-Command',
                                f"Stop-Process -Id {x['pid']} -Force -ErrorAction Stop"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                killed.append(x)
            else:
                failed.append({**x, 'err': r.stderr.strip()[:200]})
        except Exception as e:
            failed.append({**x, 'err': str(e)[:200]})
    return killed, failed


def scan(root, max_entries=400):
    """Report files that look like leftovers. Never deletes."""
    hits = []
    tmp_ext = {'.tmp', '.temp', '.log', '.part', '.crdownload', '.bak'}
    tmp_names = {'nul', 'thumbs.db', '.ds_store'}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ('.git', 'node_modules', '__pycache__', '.venv')]
        for f in filenames:
            low = f.lower()
            ext = os.path.splitext(low)[1]
            if ext in tmp_ext or low in tmp_names:
                p = os.path.join(dirpath, f)
                try:
                    hits.append({'path': p, 'mb': round(os.path.getsize(p) / 1e6, 2)})
                except OSError:
                    pass
            if len(hits) >= max_entries:
                return hits
    return hits


def tidy(paths):
    removed, failed = [], []
    for p in paths:
        p = p.strip()
        if not p:
            continue
        try:
            if os.path.isdir(p):
                os.rmdir(p)
            elif os.path.exists(p):
                os.remove(p)
                removed.append(p)
        except Exception as e:
            failed.append({'path': p, 'err': str(e)[:150]})
    return removed, failed


def purge():
    """Delete large, REGENERABLE QA residue so every round ends clean.
    Targets: the app's .store upload cache and the QA working datasets.
    NEVER touches: C:/SimQA/gen, C:/SimQA/harness, C:/SimQA/evidence,
    C:/SimQA/deliverable, or any application source."""
    targets = [
        r'C:/Simulation arena/backend/.store',
        r'C:/SimQA/data',
        r'C:/SimQA/runs',
        r'C:/SimQA/_samp',
        r'C:/SimQA/r2',
    ]
    removed, failed, freed = [], [], 0
    for t in targets:
        if not os.path.exists(t):
            continue
        for root, dirs, files in os.walk(t, topdown=False):
            for f in files:
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p)
                    os.remove(p)
                    removed.append(p)
                    freed += sz
                except Exception as e:
                    failed.append({'path': p, 'err': str(e)[:120]})
            for d in dirs:
                dp = os.path.join(root, d)
                try:
                    os.rmdir(dp)
                except OSError:
                    pass
        # top-level target itself (keep dirs that are persistent, kill r2)
        if os.path.basename(t) in ('r2', '_samp'):
            try:
                os.rmdir(t)
            except OSError:
                pass
    return removed, failed, freed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['kill', 'scan', 'tidy', 'ps', 'purge'])
    ap.add_argument('--root')
    ap.add_argument('--files')
    a = ap.parse_args()

    if a.cmd == 'ps':
        print(json.dumps(list_running(), indent=2))
    elif a.cmd == 'kill':
        k, f = kill_all()
        print(json.dumps({'killed': k, 'failed': f}, indent=2))
    elif a.cmd == 'scan':
        print(json.dumps(scan(a.root or '.'), indent=2))
    elif a.cmd == 'tidy':
        files = a.files.split(',') if a.files else []
        r, f = tidy(files)
        print(json.dumps({'removed': r, 'failed': f}, indent=2))
    elif a.cmd == 'purge':
        r, f, freed = purge()
        print(json.dumps({'files_removed': len(r),
                          'gb_freed': round(freed / 1e9, 3),
                          'failed': f}, indent=2))


if __name__ == '__main__':
    main()
