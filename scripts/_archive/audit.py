import ast, os, re, json, sys
from pathlib import Path
from collections import defaultdict, Counter

SRC = Path("src/orius")
issues = []

def add(sev, file, line, msg):
    issues.append({"severity": sev, "file": str(file), "line": line, "msg": msg})

for path in sorted(SRC.rglob("*.py")):
    rel = str(path.relative_to(SRC.parent))
    try:
        src = path.read_text(errors="replace")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        continue
    lines = src.splitlines()

    # 1. Third-party / copied code markers
    for i, l in enumerate(lines, 1):
        ls = l.strip().lower()
        if any(p in ls for p in ["copied from", "adapted from", "based on http", "source: http",
                                  "credit:", "stackoverflow", "stack overflow", "github.com/",
                                  "from https://", "originally from",
                                  "mit license", "apache license", "bsd license", "gpl",
                                  "copyright (c)", "all rights reserved"]):
            add("🔴 CRITICAL", rel, i, f"Third-party/attribution marker: {l.strip()[:80]}")

    # 2. Hardcoded secrets / API keys / passwords
    for i, l in enumerate(lines, 1):
        if re.search(r'(api_key|password|secret|token)\s*=\s*["\'][^"\']{8,}', l, re.I):
            if "TODO" not in l.upper():
                add("🔴 CRITICAL", rel, i, f"Possible hardcoded secret: {l.strip()[:60]}")

    # 3. TODO/FIXME/HACK/XXX
    for i, l in enumerate(lines, 1):
        for tag in ["TODO", "FIXME", "HACK", "XXX", "TEMP", "WORKAROUND"]:
            if tag in l and not l.strip().startswith("#!"):
                add("🟡 MEDIUM", rel, i, f"{tag}: {l.strip()[:70]}")
                break

    # 4. Bare except
    for i, l in enumerate(lines, 1):
        if re.match(r'\s*except\s*:', l):
            add("🔴 CRITICAL", rel, i, "Bare except: catches SystemExit, KeyboardInterrupt")

    # 5. exec/eval usage
    for i, l in enumerate(lines, 1):
        ls = l.strip()
        if re.match(r'(exec|eval)\s*\(', ls) and not ls.startswith('#'):
            add("🔴 CRITICAL", rel, i, f"Dynamic code execution: {ls[:60]}")

    # 6. os.system / subprocess without shell=False
    for i, l in enumerate(lines, 1):
        if "os.system(" in l:
            add("🟡 MEDIUM", rel, i, "os.system() — use subprocess instead")
        if "subprocess" in l and "shell=True" in l:
            add("🟡 MEDIUM", rel, i, "subprocess with shell=True — injection risk")

    # 7. Import * (wildcard imports)
    for i, l in enumerate(lines, 1):
        if re.match(r'\s*from\s+\S+\s+import\s+\*', l):
            add("🟡 MEDIUM", rel, i, f"Wildcard import: {l.strip()[:60]}")

    # 8. Print statements (should use logging)
    print_count = sum(1 for l in lines if re.match(r'\s*print\s*\(', l))
    if print_count > 3:
        add("🟡 MEDIUM", rel, 0, f"{print_count} print() calls — should use logging")

    # 9. Mutable default arguments
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        add("🔴 CRITICAL", rel, node.lineno,
                            f"Mutable default arg in {node.name}()")
    except SyntaxError:
        pass

    # 10. Long functions (>80 lines)
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, 'end_lineno', node.lineno + 50)
                length = end - node.lineno
                if length > 80:
                    add("🟡 MEDIUM", rel, node.lineno,
                        f"Function {node.name}() is {length} lines — consider breaking up")
    except SyntaxError:
        pass

# Sort by severity
order = {"🔴 CRITICAL": 0, "🟡 MEDIUM": 1, "🟢 LOW": 2}
issues.sort(key=lambda x: (order.get(x["severity"], 9), x["file"], x["line"]))

crit = sum(1 for i in issues if "CRITICAL" in i["severity"])
med = sum(1 for i in issues if "MEDIUM" in i["severity"])
print(f"=== HARSH CODE REVIEW: {len(issues)} issues found ===")
print(f"  🔴 CRITICAL: {crit}")
print(f"  🟡 MEDIUM:   {med}")
print()

for iss in issues:
    loc = f"{iss['file']}:{iss['line']}" if iss['line'] else iss['file']
    print(f"  {iss['severity']}  {loc}")
    print(f"    {iss['msg']}")
print()
print(f"Total files scanned: {len(list(SRC.rglob('*.py')))}")
