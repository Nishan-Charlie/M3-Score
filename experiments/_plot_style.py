"""
Shared figure style for the manuscript figures.
===============================================

One place to set typography so every panel in the paper matches:

  * Times New Roman throughout, including inside math, so ``$\\rho$`` and
    ``MMD$^2$`` do not fall back to a different face mid-label.
  * Large, bold titles, axis labels and legends -- these are read at column
    width in a two-column journal, so they are sized to survive reduction.
  * Tick labels large enough that every number stays legible.

Call :func:`apply_paper_style` before creating any figure. Use
:func:`sentence_case` for labels so titles and axis names get a capital first
letter without upper-casing acronyms or math.

Times New Roman ships with Windows and is present on this machine; on a system
without it, ``FONT_STACK`` falls through to the next available serif face
rather than silently rendering in the sans-serif default.
"""

from __future__ import annotations

import re

import matplotlib

# Fallbacks in preference order; matplotlib takes the first one installed.
FONT_STACK = [
    "Times New Roman",
    "Nimbus Roman",
    "Liberation Serif",
    "STIX Two Text",
    "DejaVu Serif",
]

# Point sizes. These are deliberately large: journal figures are reduced to
# column width, and the reviewer complaint that motivated this module was that
# numbers were not legible.
SIZES = {
    "base": 15,      # fallback for anything unlisted
    "title": 19,     # panel titles
    "label": 17,     # axis labels
    "tick": 14,      # tick numbers
    "legend": 13,
    "suptitle": 21,
}


def apply_paper_style(bold: bool = True) -> None:
    """Install the manuscript typography into the global rcParams."""
    weight = "bold" if bold else "normal"

    matplotlib.rcParams.update({
        # ---- family -------------------------------------------------------
        "font.family": "serif",
        "font.serif": FONT_STACK,
        # Render math in the same face; 'custom' lets us point mathtext at the
        # text font instead of its own Computer Modern default.
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_STACK[0],
        "mathtext.it": f"{FONT_STACK[0]}:italic",
        "mathtext.bf": f"{FONT_STACK[0]}:bold",
        "mathtext.default": "regular",

        # ---- sizes --------------------------------------------------------
        "font.size": SIZES["base"],
        "axes.titlesize": SIZES["title"],
        "axes.labelsize": SIZES["label"],
        "xtick.labelsize": SIZES["tick"],
        "ytick.labelsize": SIZES["tick"],
        "legend.fontsize": SIZES["legend"],
        "figure.titlesize": SIZES["suptitle"],

        # ---- weights ------------------------------------------------------
        "axes.titleweight": weight,
        "axes.labelweight": weight,
        "font.weight": weight,

        # ---- readability --------------------------------------------------
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.6",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
    })


_PANEL_MARKER = re.compile(r"^\s*\(\s*[a-zA-Z0-9]\s*\)\s*")


def sentence_case(text: str) -> str:
    """Capitalise the first letter of the label, leaving the rest alone.

    Acronyms (FID, MMD, AUC), math segments and layer names such as ``L12``
    must survive untouched, so this deliberately does not use ``str.capitalize``
    which would lower-case the remainder.

    A leading panel marker is stepped over rather than capitalised: subfigure
    labels are conventionally lower case and are referenced that way from the
    manuscript text, so ``(a) fidelity ...`` becomes ``(a) Fidelity ...`` and
    not ``(A) Fidelity ...``.
    """
    if not text:
        return text

    m = _PANEL_MARKER.match(text)
    start = m.end() if m else 0
    head, rest = text[:start], text[start:]

    for i, ch in enumerate(rest):
        if ch.isalpha():
            return head + rest[:i] + rest[i].upper() + rest[i + 1:]
        # Stop at anything that is not leading punctuation or a math delimiter,
        # so a label that genuinely starts with a number is left as written.
        if ch not in " $\\":
            break
    return text


def style_legend(ax, **kwargs):
    """Legend with bold text at the configured size."""
    leg = ax.legend(**kwargs)
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontweight("bold")
    return leg


def finalize(ax, title=None, xlabel=None, ylabel=None, legend=False, **legend_kw):
    """Apply sentence case and bold weight to one axes' text elements."""
    if title is not None:
        ax.set_title(sentence_case(title), fontweight="bold",
                     fontsize=SIZES["title"])
    if xlabel is not None:
        ax.set_xlabel(sentence_case(xlabel), fontweight="bold",
                      fontsize=SIZES["label"])
    if ylabel is not None:
        ax.set_ylabel(sentence_case(ylabel), fontweight="bold",
                      fontsize=SIZES["label"])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    if legend:
        style_legend(ax, **legend_kw)
    ax.grid(alpha=0.3)
    return ax
