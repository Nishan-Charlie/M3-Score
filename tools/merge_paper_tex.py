"""
Inline every \\input{...} in a LaTeX document into a single self-contained file.
===============================================================================

The manuscript was written as a master file plus one file per section, which is
convenient while drafting and inconvenient when the paper has to travel as a
single source file -- which is what most journal submission systems and
co-authors want.

This inlines each \\input in place, wrapping the inserted body in comment
banners so the original section boundaries stay visible and the merge can be
undone by hand. Line endings are preserved byte-for-byte (the source uses CRLF,
and rewriting it as LF would show up as a whole-file diff).

Only \\input lines that sit alone on their own line are replaced, which is how
they appear in this document; an \\input embedded mid-line is left alone rather
than guessed at. Missing targets are left as-is and reported.

Usage
-----
    python tools/merge_paper_tex.py --src paper_cmig/paper.tex \\
                                    --out paper_cmig/paper_merged.tex
"""

from __future__ import annotations

import argparse
import os
import re

# The source uses CRLF, so the trailing \r must be consumed explicitly:
# in MULTILINE mode $ matches before \n, leaving the \r unmatched by [ \t]*.
INPUT_RE = re.compile(r"^[ \t]*\\input\{([^}]+)\}[ \t]*\r?$", re.MULTILINE)


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def merge(src: str, recursive: bool = True) -> tuple[str, list[str]]:
    """Return (merged_text, list_of_inlined_paths)."""
    base = os.path.dirname(src)
    text = read(src)
    inlined: list[str] = []
    # Detect the document's dominant line ending so banners match it.
    eol = "\r\n" if "\r\n" in text else "\n"

    def repl(m: re.Match) -> str:
        name = m.group(1)
        path = name if name.endswith(".tex") else name + ".tex"
        full = os.path.join(base, path) if base else path
        if not os.path.exists(full):
            print(f"  [keep]   {name}: not found, leaving \\input in place")
            return m.group(0)

        if recursive:
            body, nested = merge(full, recursive=True)
            inlined.extend(nested)
        else:
            body = read(full)

        body = body.rstrip("\r\n")
        inlined.append(path)
        rule = "=" * max(4, 66 - len(path))
        return (f"% ===== begin {path} {rule}{eol}"
                f"{body}{eol}"
                f"% ===== end {path} {rule}")

    return INPUT_RE.sub(repl, text), inlined


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="paper_cmig/paper.tex")
    ap.add_argument("--out", default="paper_cmig/paper_merged.tex")
    args = ap.parse_args()

    merged, inlined = merge(args.src)
    write(args.out, merged)

    print(f"[merge_paper_tex] {args.src} -> {args.out}")
    for p in inlined:
        print(f"  inlined  {p}")
    print(f"  {len(inlined)} file(s), {merged.count(chr(10)) + 1} lines, "
          f"{len(merged)} chars")
    remaining = INPUT_RE.findall(merged)
    print(f"  remaining \\input: {remaining if remaining else 'none'}")


if __name__ == "__main__":
    main()
