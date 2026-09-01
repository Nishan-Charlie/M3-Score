"""
Apply the round-3 review corrections to the manuscript source.
==============================================================

Each entry is a literal search/replace against paper.tex. Kept as a script
rather than done by hand so the exact set of wording changes is reviewable and
repeatable, and so a failed match is reported loudly instead of silently
skipped. Raw strings throughout: the manuscript is full of backslashes, and
shell heredocs mangle them.

Run once; re-running is a no-op because the old strings are gone.
"""

from __future__ import annotations

import sys

PATH = "paper_cmig/paper.tex"

FIXES: list[tuple[str, str, str]] = [
    # --- representativeness is not established -----------------------------
    ("representativeness overclaim (results)",
     r"""to $89.0$, correctly reporting that a more representative reference resolves
the generator gap more sharply.""",
     r"""to $89.0$. The standardisation should not be over-read: the numerator is
almost unchanged across cohort sizes ($0.0776$, $0.0752$, $0.0766$), and the
rise is driven by the null spread contracting $5.4\times$. What a more
subject-diverse reference buys is resolution, not a larger effect."""),

    # --- site attribution in the protocol list -----------------------------
    ("site attribution in protocol",
     r"      Sorted-order convenience selects a site.",
     r"""      Sorted-order convenience can select a structured subject subset with
      systematic feature differences."""),

    # --- "common convenience" is not established by our audit --------------
    ("common convenience",
     r"""common convenience --- selects a $17$-subject block that is systematically
atypical of the cohort,""",
     r"""plausible but reproducibility-poor convenience --- selects a $17$-subject
block that is systematically atypical of the cohort,"""),

    # --- unbiasedness does not license cross-N comparison ------------------
    ("unbiasedness overclaim",
     r"""unbiased, so values at different $N$ are comparable in expectation --- which is
precisely why the residual dependence on \emph{which} images form the reference
is worth isolating.""",
     r"""unbiased for a fixed kernel and a fixed pair of sampling distributions. That
is a weaker guarantee than it is often read as: it says nothing about whether
the empirical reference represents the population it stands for. We therefore
hold the image count fixed in every cohort-composition experiment, so that what
varies is which subjects constitute the reference and not how many images it
contains."""),

    # --- coverage claim precision ------------------------------------------
    ("coverage contribution bullet",
     r"""\item We report that $k$-NN coverage is invalid at every depth in this feature""",
     r"""\item We report that $k$-NN recall fails its mode-drop sanity check at every
      evaluated depth in this feature"""),
    ("coverage subsection heading",
     r"\subsection{$k$-NN coverage is invalid at every depth}",
     r"\subsection{$k$-NN recall fails its mode-drop check at every depth}"),

    # --- permutation-test rhetoric -----------------------------------------
    ("competent generator phrasing",
     r"""samples were drawn from the same distribution. For any competent medical
generator that null is false, and in every configuration we tested the test
rejected it at overwhelming significance regardless of generator quality; we measure""",
     r"""samples were drawn from the same distribution. For realistic generative
models, exact equality of the generated and real distributions is not the
hypothesis of practical interest, and in every configuration we tested the test
rejected it at overwhelming significance regardless of generator quality; we measure"""),

    # --- do not lean on an undocumented earlier analysis -------------------
    ("earlier analysis attribution",
     r"""This result contradicts an earlier analysis of ours conducted on a
\textsc{head} reference, which reported that shallow blocks outperformed
mid-depth ones for this task. That conclusion was an artefact of the
$17$-subject reference:""",
     r"""Running the identical sweep under the \textsc{head} protocol reverses this
ordering: there, shallow blocks appear to outperform mid-depth ones. Both
protocols are reported here from the same code and the same generated set, so
the reversal can be checked directly rather than taken on trust. The
\textsc{head} result is an artefact of the $17$-subject reference:"""),
]


def main() -> int:
    with open(PATH, "r", encoding="utf-8", newline="") as f:
        text = f.read()

    # The manuscript uses CRLF; the patterns above are written with plain \n.
    # Convert the patterns to the file's line ending rather than rewriting the
    # file, so the merge does not show up as a whole-file diff.
    eol = "\r\n" if "\r\n" in text else "\n"

    def to_eol(s: str) -> str:
        return s.replace("\r\n", "\n").replace("\n", eol)

    failed = []
    for name, old_raw, new_raw in FIXES:
        old, new = to_eol(old_raw), to_eol(new_raw)
        if old in text:
            text = text.replace(old, new, 1)
            print(f"  ok      {name}")
        elif new in text:
            print(f"  already {name}")
        else:
            failed.append(name)
            print(f"  MISS    {name}")

    with open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    if failed:
        print(f"\n{len(failed)} unmatched: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
