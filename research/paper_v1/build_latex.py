"""Build the arXiv-ready LaTeX source and PDF from paper/writeup.md.

The Markdown stays the single source (research.paper_v1.claims checks it); this
script derives paper/latex/main.tex, which compiles with plain pdflatex, and
copies the result to paper/writeup.pdf. It also writes paper/kaggle/: the
Writeup body (title and subtitle are separate Kaggle fields) and its fields.
  python -m research.paper_v1.build_latex [--author NAME] [--date TEXT]
"""
import argparse
from datetime import date
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper'
OUT = PAPER / 'latex'
TEMPLATE = Path(__file__).with_name('latex_template.tex')
FIGURE = 'figure-decomposition.png'
CITE = re.compile(r'\[(\d{1,2}(?:\s*[,–]\s*\d{1,2})*)\](?!\()')
TEX_SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def section(text, start, end=None):
    a = text.index(start) + len(start)
    return text[a:text.index(end, a)] if end else text[a:]


def cite(text, n_refs):
    def keys(m):
        nums = []
        for part in re.split(r'\s*,\s*', m.group(1)):
            lo, _, hi = part.partition('–')
            nums += range(int(lo), int(hi or lo) + 1)
        if not all(1 <= k <= n_refs for k in nums):
            return m.group(0)
        return '\\cite{' + ','.join(f'r{k}' for k in nums) + '}'
    return CITE.sub(keys, text)


def tex_escape(text):
    return ''.join(TEX_SPECIAL.get(ch, ch) for ch in text)


def bibliography(refs):
    items = []
    for line in refs.strip().splitlines():
        m = re.match(r'\[(\d+)\] (.*)', line.strip())
        if not m:
            continue
        body = tex_escape(m.group(2))
        # Bare dataset hosts become breakable URLs.
        body = re.sub(r'(huggingface\.co/\S+?)(?=[\s,)]|\.?$)', lambda u: r'\url{' + u.group(1).replace(r'\_', '_') + '}', body)
        items.append(f'\\bibitem{{r{m.group(1)}}} {body}')
    return items


def build(author, when):
    md = (PAPER / 'writeup.md').read_text()
    title = re.search(r'^# (.+)$', md, re.M).group(1)
    subtitle = re.search(r'^### (.+)$', md, re.M).group(1)
    abstract = section(md, '## Abstract', '## 1. ').strip()
    body = '## 1. ' + section(md, '## 1. ', '## References')
    kaggle(md, title, subtitle)
    items = bibliography(section(md, '## References'))
    n = len(items)
    # LaTeX numbers sections itself; the Markdown numbering matches its order.
    body = re.sub(r'^(#{2,3}) \d+(?:\.\d+)?\.? ', r'\1 ', body, flags=re.M)
    body = re.sub(r'!\[Figure 1\]\(' + re.escape(FIGURE) + r'\)\n\*Figure 1\. (.+?)\*\n',
                  lambda m: ('```{=latex}\n\\begin{figure}[t]\n\\centering\n'
                             f'\\includegraphics[width=0.92\\linewidth]{{{FIGURE}}}\n'
                             f'\\caption{{{" ".join(m.group(1).split())}}}\n\\label{{fig:decomposition}}\n'
                             '\\end{figure}\n```\n'), body, flags=re.S)
    assert '![Figure 1]' not in body, 'figure block not converted'
    body = cite(body, n) + ('\n```{=latex}\n\\begin{thebibliography}{99}\n\\small\n'
                            + '\n'.join(items) + '\n\\end{thebibliography}\n```\n')
    meta = {'title': title, 'subtitle': subtitle, 'author': author, 'date': when, 'abstract': cite(abstract, n)}
    front = '---\n' + ''.join(f'{k}: {json.dumps(v, ensure_ascii=False)}\n' for k, v in meta.items()) + '---\n\n'
    OUT.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False) as f:
        f.write(front + body)
        source = f.name
    subprocess.run(['pandoc', source, '-f', 'markdown', '-t', 'latex', '--template', str(TEMPLATE),
                    '--shift-heading-level-by=-1', '--wrap=preserve', '-o', str(OUT / 'main.tex')], check=True)
    Path(source).unlink()
    tex = (OUT / 'main.tex').read_text()
    # Long identifiers and paths in code font may break after _, / and . .
    tex = re.sub(r'\\texttt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                 lambda m: '\\texttt{' + re.sub(r'(\\_|/|\.)(?=\w)', r'\1\\allowbreak{}', m.group(1)) + '}', tex)
    # "~17" means "about 17".
    tex = tex.replace(r'\textasciitilde{}', r'\ensuremath{\sim}')
    # The tables are short: typeset them as unbreakable tabulars instead of longtables.
    tex = re.sub(r'\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\endhead\n(?:\\endfirsthead\n)?'
                 r'\\bottomrule\\noalign\{\}\n\\endlastfoot\n(.*?)\\end\{longtable\}',
                 lambda m: ('\\begin{center}\\small\n\\begin{tabular}{' + m.group(1) + '}\n' + m.group(2)
                            + m.group(3) + '\\bottomrule\n\\end{tabular}\n\\end{center}'), tex, flags=re.S)
    assert 'longtable}' not in tex.split('\\begin{document}')[1], 'unconverted longtable'
    # Natural-width columns: these tables fit the text width once set in \\small.
    tex = re.sub(r'>\{\\(raggedright|raggedleft|centering)\\arraybackslash\}p\{\(\\linewidth - \d+\\tabcolsep\) \* \\real\{[\d.]+\}\}',
                 lambda m: {'raggedright': 'l', 'raggedleft': 'r', 'centering': 'c'}[m.group(1)], tex)
    tex = re.sub(r'\\begin\{minipage\}\[[a-z]\]\{\\linewidth\}\\(?:raggedright|raggedleft|centering)\n?(.*?)\\end\{minipage\}',
                 lambda m: ' '.join(m.group(1).split()), tex, flags=re.S)
    (OUT / 'main.tex').write_text(tex)
    shutil.copyfile(PAPER / FIGURE, OUT / FIGURE)
    for _ in range(2):  # second pass resolves citations and the figure reference
        run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                             cwd=OUT, capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(run.stdout[-3000:])
    log = (OUT / 'main.log').read_text(errors='replace')
    shutil.copyfile(OUT / 'main.pdf', PAPER / 'writeup.pdf')
    for ext in ['aux', 'log', 'out', 'pdf']:
        (OUT / f'main.{ext}').unlink(missing_ok=True)
    warnings = [l for l in log.splitlines() if re.match(r'(Overfull|LaTeX Warning|Package \w+ Warning)', l)]
    print(PAPER / 'writeup.pdf', f'{len(warnings)} warnings')
    for w in warnings:
        print(' ', w)


def kaggle(md, title, subtitle):
    # The editor takes everything from the Abstract on; Figure 1 is uploaded in the editor.
    out = PAPER / 'kaggle'
    out.mkdir(exist_ok=True)
    (out / 'body.md').write_text(md[md.index('## Abstract'):])
    (out / 'fields.json').write_text(json.dumps({'title': title, 'subtitle': subtitle, 'figure': FIGURE},
                                                indent=1, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--author', default='Xining Li')
    ap.add_argument('--date', default=date.today().strftime('%B %Y'))
    args = ap.parse_args()
    build(args.author, args.date)


if __name__ == '__main__':
    main()
