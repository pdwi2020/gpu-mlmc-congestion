#!/usr/bin/env python3
"""Produce the IEEE Access "Highlighted PDF" deliverable.

The portal asks for the revised manuscript with changes highlighted in yellow --
not a latexdiff strikethrough document. So the clean PDF stays the source of
truth for layout, and highlights are applied to it mechanically:

  1. latexdiff gives us the added text as \\DIFadd{...} blocks (already produced
     into diff.tex by the caller).
  2. Each block is stripped of LaTeX markup down to plain prose fragments.
  3. Each fragment is located in the CLEAN pdf with page.search_for() and a
     yellow highlight annotation is drawn over the hit.

The point of doing it this way rather than compiling a marked-up .tex is that the
submitted PDF is then byte-for-byte the same document as the clean manuscript,
with annotations layered on top -- no risk of the highlighted and clean versions
differing in content, pagination or numbering.

Fragments that cannot be located are COUNTED AND REPORTED, never silently
dropped. Some misses are unavoidable: text inside tables and equations, and text
whose line-wrapping differs from the source, will not match a plain-string
search. A coverage figure that is quietly 60% would misrepresent the deliverable,
so the number is printed and written to the sidecar JSON.

Run (after building main.pdf and diff.tex):
    python scripts/make_highlighted_pdf.py --diff-tex <path> --pdf <path> --out <path>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def extract_difadd_blocks(tex: str):
    """Return the contents of every \\DIFadd{...}, brace-balanced."""
    blocks, i, token = [], 0, "\\DIFadd{"
    while True:
        i = tex.find(token, i)
        if i < 0:
            break
        j = i + len(token)
        depth, start = 1, j
        while j < len(tex) and depth:
            if tex[j] == "\\":
                j += 2
                continue
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
            j += 1
        blocks.append(tex[start:j - 1])
        i = j
    return blocks


SEP = "\u0000"


def to_plain(s: str):
    """Reduce a LaTeX fragment to prose, marking discontinuities with SEP.

    Inline math, citations and refs become SEP rather than a space: the prose
    on either side of a formula is not contiguous in the rendered PDF, so
    joining across it produces a token sequence that exists nowhere on the page
    and can never match. Splitting there is what lifted coverage from 41%.
    """
    s = re.sub(r"\$[^$]*\$", SEP, s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}", SEP, s)
    s = re.sub(r"\\(cite|ref|eqref|label|includegraphics)\{[^}]*\}", SEP, s)
    s = re.sub(r"\\[a-zA-Z@]+\s*", SEP, s)
    s = s.replace("~", " ")
    s = re.sub(r"[{}\\&%#_^]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def norm_tokens(text: str):
    """Lowercase alphanumeric tokens; the unit both sides of the match use."""
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower())]


def fragments(text: str, min_len: int, max_len: int):
    """Split prose into contiguous runs, breaking at SEP and sentence ends."""
    out = []
    for run in text.split(SEP):
        for sentence in re.split(r"(?<=[.;:])\s+", run):
            sentence = sentence.strip()
            if len(sentence) < min_len:
                continue
            if len(sentence) <= max_len:
                out.append(sentence)
                continue
            words, cur = sentence.split(), ""
            for w in words:
                if len(cur) + len(w) + 1 > max_len:
                    if len(cur.strip()) >= min_len:
                        out.append(cur.strip())
                    cur = w
                else:
                    cur += " " + w
            if len(cur.strip()) >= min_len:
                out.append(cur.strip())
    return out


def page_index(page):
    """Word rects plus the normalised token stream they correspond to."""
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,word_no
    toks, rects = [], []
    for w in words:
        for t in norm_tokens(w[4]):
            toks.append(t)
            rects.append(w[:4])
    return toks, rects


def find_sequence(hay, needle, start=0):
    """Index of the first contiguous occurrence of `needle` in `hay`."""
    n = len(needle)
    if not n or n > len(hay):
        return -1
    first = needle[0]
    i = start
    while True:
        try:
            i = hay.index(first, i)
        except ValueError:
            return -1
        if hay[i:i + n] == needle:
            return i
        i += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff-tex", required=True)
    ap.add_argument("--pdf", required=True, help="the CLEAN compiled manuscript")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=28)
    ap.add_argument("--max-len", type=int, default=70)
    ap.add_argument("--min-tokens", type=int, default=4,
                    help="shorter runs risk false-positive highlights")
    args = ap.parse_args()

    import fitz

    tex = Path(args.diff_tex).read_text(errors="ignore")
    blocks = extract_difadd_blocks(tex)
    frags = []
    for b in blocks:
        frags.extend(fragments(to_plain(b), args.min_len, args.max_len))
    # de-duplicate, keep order
    seen, uniq = set(), []
    for f in frags:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    print(f"{len(blocks)} DIFadd blocks -> {len(uniq)} unique search fragments")

    doc = fitz.open(args.pdf)
    # Match on the page's WORD SEQUENCE rather than with search_for(). A literal
    # string search fails whenever the PDF wraps a fragment across a line or
    # column break, which in a two-column layout is most of them; matching token
    # runs and highlighting the union of their word rects is wrap-invariant.
    pages = [page_index(doc[i]) for i in range(doc.page_count)]
    hits = misses = 0
    per_page = [0] * doc.page_count
    unmatched = []
    for frag in uniq:
        needle = norm_tokens(frag)
        if len(needle) < args.min_tokens:
            continue
        found = False
        for pno, (toks, rects) in enumerate(pages):
            k = find_sequence(toks, needle)
            if k < 0:
                continue
            page = doc[pno]
            # group the matched words into per-line rectangles
            spans, cur = [], None
            for r in rects[k:k + len(needle)]:
                rect = fitz.Rect(r)
                if cur is not None and abs(rect.y0 - cur.y0) < 2.0:
                    cur |= rect
                else:
                    if cur is not None:
                        spans.append(cur)
                    cur = fitz.Rect(rect)
            if cur is not None:
                spans.append(cur)
            for sp in spans:
                a = page.add_highlight_annot(sp)
                a.set_colors(stroke=(1, 1, 0.25))
                a.update()
            per_page[pno] += len(spans)
            found = True
            break
        hits += found
        if not found:
            misses += 1
            unmatched.append(frag)

    doc.save(args.out, garbage=3, deflate=True)
    coverage = hits / max(len(uniq), 1)
    print(f"located {hits}/{len(uniq)} fragments ({coverage:.1%}); "
          f"{misses} not found")
    print("highlights per page: " +
          ", ".join(f"p{i+1}:{c}" for i, c in enumerate(per_page) if c))
    print("\nUnlocated fragments are expected for text inside tables, equations "
          "and captions, which a plain-string search cannot match. Review the "
          "output visually before uploading.")
    side = Path(args.out).with_suffix(".coverage.json")
    side.write_text(json.dumps({
        "difadd_blocks": len(blocks), "fragments": len(uniq),
        "located": hits, "not_located": misses, "coverage": coverage,
        "highlights_per_page": per_page,
        "unmatched_examples": unmatched[:25],
    }, indent=2))
    print(f"\nWrote {args.out} and {side.name}")


if __name__ == "__main__":
    main()
