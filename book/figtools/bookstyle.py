"""Figure style for the book.

Deliberately different from the style of the source papers. Those figures were
built for a two column journal page at 88 mm and 180 mm, in a sans serif face,
with the perceptually uniform sequential maps that journal house styles favour.

A book page is a single column 117.5 mm wide, set in a Times-like serif. Figures
that sit inside that column should look like they belong to the text around
them, so everything here is serif, quieter, and laid out one panel at a time
instead of as multi panel strips. Colour choices also differ from the papers,
so that no figure in this book is a reproduction of a published one.
"""

import matplotlib
matplotlib.use("pdf")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Springer type area for a 155 by 235 mm book.
MM = 1.0 / 25.4
TEXT_WIDTH = 117.5 * MM          # 4.63 in, the full measure
HALF_WIDTH = 56.0 * MM           # two figures side by side

# A restrained palette. The four spheres each keep one colour throughout the
# book, so a reader learns to read the figures without checking the legend.
SPHERE = {
    "atmosphere": "#2E6F95",     # blue
    "ocean":      "#1B7A6E",     # teal
    "hydrosphere": "#B07A2B",    # ochre
    "solid":      "#8C4A32",     # earth brown
}

# Scenario colours, cool to warm, distinct from the papers' choices.
SSP = {
    "ssp126": "#2E6F95",
    "ssp245": "#5E9C76",
    "ssp370": "#D4913A",
    "ssp585": "#A8402F",
}
SSP_LABEL = {
    "ssp126": "SSP1 2.6",
    "ssp245": "SSP2 4.5",
    "ssp370": "SSP3 7.0",
    "ssp585": "SSP5 8.5",
}

# Sequential map for hazard: pale sand through ochre to deep brown red. Chosen
# to read as "dryness and ground" and to differ from the papers' magma and
# plasma ramps.
HAZARD = LinearSegmentedColormap.from_list("hazard", [
    "#F4F1E8", "#E8DCC0", "#D9BE86", "#C79A55", "#AE7238",
    "#8E4B2C", "#6B2C24", "#43161A",
])

# Sequential map for time, pale recent to dark distant.
WHEN = LinearSegmentedColormap.from_list("when", [
    "#3B1F2B", "#7A2F3A", "#B85A3C", "#DE9A54", "#F0CE8E", "#F7EBD2",
])

LAND_GREY = "#DCDCD6"
SEA = "#FFFFFF"
RULE = "#4A4A4A"


def apply():
    """Install the book rc parameters. Call once before plotting."""
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "figure.facecolor": "white",

        "font.family": "serif",
        "font.serif": ["Nimbus Roman No9 L", "Times New Roman",
                       "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,

        "axes.linewidth": 0.6,
        "axes.edgecolor": RULE,
        "axes.labelcolor": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,

        "lines.linewidth": 1.1,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": RULE,
        "ytick.color": RULE,

        "legend.frameon": False,
        "figure.autolayout": False,
    })


def figure(width=TEXT_WIDTH, height=None, ratio=0.62):
    """A figure sized to the book measure."""
    if height is None:
        height = width * ratio
    return plt.subplots(figsize=(width, height))


def finish(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path
