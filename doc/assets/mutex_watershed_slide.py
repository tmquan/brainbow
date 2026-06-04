"""Generate the pedagogical Mutex Watershed slide (doc/assets/mutex_watershed_slide.png).

A 6-voxel chain worked example whose merge / mutex decisions follow
``brainbow/inference/mutex_watershed.py::_mws_core`` exactly:

* attractive (nearest-neighbour, "single-step") edges carry priority = affinity;
* repulsive (long-range, "multi-step") edges carry priority = 1 - affinity;
* all edges are processed in descending priority with a union-find +
  mutual-exclusion constraints.

Run: python doc/assets/mutex_watershed_slide.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Worked example
# ---------------------------------------------------------------------------
# True segmentation: {1,2,3} | {4,5,6}.  Boundary is between node 3 and 4.
POS = {i: (i - 1, 0.0) for i in range(1, 7)}

# (u, v, priority) where priority = affinity for attractive edges.
ATTR = [(1, 2, 0.95), (2, 3, 0.90), (3, 4, 0.08), (4, 5, 0.92), (5, 6, 0.93)]
# (u, v, priority) where priority = 1 - affinity (repulsion strength).
REP = [(2, 4, 0.88), (3, 5, 0.85), (1, 3, 0.12), (4, 6, 0.10)]

GREEN = "#2e8b57"
RED = "#c0392b"
GREY = "#888888"
CLUST_A = "#4c78c8"
CLUST_B = "#e0843a"

# Sorted edge list (descending priority) with the union-find decision.
QUEUE = [
    ("a", 1, 2, 0.95, "merge"),
    ("a", 5, 6, 0.93, "merge"),
    ("a", 4, 5, 0.92, "merge"),
    ("a", 2, 3, 0.90, "merge"),
    ("r", 2, 4, 0.88, "mutex"),
    ("r", 3, 5, 0.85, "skip (mutex)"),
    ("r", 1, 3, 0.12, "skip (same)"),
    ("r", 4, 6, 0.10, "skip (same)"),
    ("a", 3, 4, 0.08, "BLOCKED"),
]


def draw_node(ax, i, color, r=0.20, edge="k", text="k"):
    x, y = POS[i]
    ax.add_patch(plt.Circle((x, y), r, facecolor=color, edgecolor=edge,
                            lw=1.8, zorder=3))
    ax.text(x, y, str(i), ha="center", va="center", color=text,
            fontsize=12, fontweight="bold", zorder=4)


def straight_edge(ax, u, v, color, weight=None, lw=2.2, ls="-", alpha=1.0):
    (x0, y0), (x1, y1) = POS[u], POS[v]
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, alpha=alpha,
            zorder=1, solid_capstyle="round")
    if weight is not None:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 0.28, f"{weight:.2f}",
                ha="center", va="center", color=color, fontsize=8.5)


def arc_edge(ax, u, v, color, weight=None, lw=2.0, ls=(0, (4, 3)), rad=0.45):
    # Always bulge the arc UPWARD (negative rad for a left->right path) so
    # repulsive long-range arcs sit above the chain, clear of the
    # attractive nn edges and their weight labels.
    (x0, y0), (x1, y1) = POS[u], POS[v]
    p = FancyArrowPatch((x0, y0), (x1, y1),
                        connectionstyle=f"arc3,rad={-abs(rad)}",
                        arrowstyle="-", color=color, lw=lw, ls=ls, zorder=1)
    ax.add_patch(p)
    if weight is not None:
        xm = (x0 + x1) / 2
        ym = max(y0, y1) + abs(x1 - x0) * abs(rad) * 0.9 + 0.08
        ax.text(xm, ym, f"{weight:.2f}", ha="center", va="center",
                color=color, fontsize=8.5)


def base_axes(ax, title, ylim=(-1.25, 2.3)):
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlim(-0.7, 5.7)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9), dpi=130)
fig.suptitle(
    "Mutex Watershed: attractive (single-step) + repulsive (multi-step) "
    "graph partition",
    fontsize=17, fontweight="bold", y=0.975,
)
gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.12,
                      left=0.04, right=0.975, top=0.9, bottom=0.04)

# ---- Panel 1: the affinity graph --------------------------------------
ax = fig.add_subplot(gs[0, 0])
base_axes(ax, "1.  Affinity graph (per-voxel edges)")
for u, v, w in ATTR:
    straight_edge(ax, u, v, GREEN, w)
for u, v, w in REP:
    arc_edge(ax, u, v, RED, w, rad=0.5 if abs(u - v) == 2 else 0.32)
for i in range(1, 7):
    draw_node(ax, i, "#eeeeee")
ax.text(2.5, -1.05,
        "solid green = attractive nearest-neighbour (priority = affinity)\n"
        "dashed red = repulsive long-range / multi-step (priority = 1 - affinity)",
        ha="center", va="center", fontsize=9.0)

# ---- Panel 2: sorted priority queue -----------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.set_title("2.  Sort ALL edges by priority (desc)", fontsize=12.5,
             fontweight="bold", loc="left", pad=6)
ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
y = 0.92
ax.text(0.02, y, "prio  edge        type        decision", fontsize=10.5,
        family="monospace", fontweight="bold")
y -= 0.10
for kind, u, v, p, dec in QUEUE:
    col = GREEN if kind == "a" else RED
    sym = "attract" if kind == "a" else "repulse"
    decol = {"merge": GREEN, "mutex": RED}.get(dec, "#333333")
    if dec == "BLOCKED":
        decol = RED
    ax.text(0.02, y, f"{p:.2f}", fontsize=10, family="monospace", color=col)
    ax.text(0.16, y, f"({u},{v})", fontsize=10, family="monospace", color=col)
    ax.text(0.40, y, sym, fontsize=10, family="monospace", color=col)
    ax.text(0.66, y, dec, fontsize=10, family="monospace", color=decol,
            fontweight="bold" if dec in ("merge", "mutex", "BLOCKED") else "normal")
    y -= 0.094

# ---- Panel 3: the rule ------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
ax.set_title("3.  The single rule (Kruskal + mutex)", fontsize=12.5,
             fontweight="bold", loc="left", pad=6)
ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
rule = (
    "for each edge, in descending priority:\n\n"
    "   find roots ru, rv\n"
    "   if ru == rv:            skip\n"
    "   if mutex(ru, rv):       skip\n\n"
    "   if edge is ATTRACTIVE:\n"
    "        union(ru, rv)\n"
    "        (merge clusters)\n\n"
    "   if edge is REPULSIVE:\n"
    "        add_mutex(ru, rv)\n"
    "        (forbid future merge)\n\n"
    "no threshold, no seeds\n"
    "-> parameter-free partition"
)
ax.text(0.04, 0.96, rule, fontsize=10.5, family="monospace", va="top")

# ---- Panel 4: attractive merges ---------------------------------------
ax = fig.add_subplot(gs[1, 0])
base_axes(ax, "4.  High-priority attractive edges -> MERGE")
group = {1: CLUST_A, 2: CLUST_A, 3: CLUST_A, 4: CLUST_B, 5: CLUST_B, 6: CLUST_B}
for u, v, w in ATTR:
    if (u, v) in [(1, 2), (2, 3), (4, 5), (5, 6)]:
        straight_edge(ax, u, v, GREEN, lw=4.5)
    else:
        straight_edge(ax, u, v, GREY, lw=1.5, ls=(0, (2, 3)), alpha=0.5)
for i in range(1, 7):
    draw_node(ax, i, group[i], text="white")
ax.text(1.0, 1.5, "{1,2,3}", ha="center", color=CLUST_A, fontsize=12, fontweight="bold")
ax.text(4.0, 1.5, "{4,5,6}", ha="center", color=CLUST_B, fontsize=12, fontweight="bold")
ax.text(2.5, -1.05, "the four strongest edges are attractive within-object\n"
        "edges -> union-find merges them into two clusters",
        ha="center", va="center", fontsize=9.0)

# ---- Panel 5: repulsive mutex + blocked merge -------------------------
ax = fig.add_subplot(gs[1, 1])
base_axes(ax, "5.  Repulsive edge -> MUTEX")
for u, v, w in ATTR:
    if (u, v) == (3, 4):
        straight_edge(ax, u, v, RED, lw=2.0, ls=(0, (1, 2)), alpha=0.8)
    else:
        straight_edge(ax, u, v, GREEN, lw=3.0, alpha=0.5)
arc_edge(ax, 2, 4, RED, 0.88, lw=3.2, rad=0.5)
# mutex wall between cluster A and B (between nodes 3 and 4 -> x=2.5)
ax.plot([2.5, 2.5], [-0.7, 1.15], color=RED, lw=3.5, ls=(0, (3, 2)), zorder=5)
ax.text(2.5, 1.4, "MUTEX", ha="center", color=RED, fontsize=11, fontweight="bold")
# blocked X on the (3,4) attractive edge
ax.text(2.5, 0.0, "x", ha="center", va="center", color=RED, fontsize=20,
        fontweight="bold", zorder=6)
for i in range(1, 7):
    draw_node(ax, i, group[i], text="white")
ax.text(2.5, -1.05,
        "r(2,4)=0.88 adds a mutex between {1,2,3} and {4,5,6};\n"
        "r(3,5), r(1,3), r(4,6) are skipped; a(3,4)=0.08 is BLOCKED",
        ha="center", va="center", fontsize=9.0)

# ---- Panel 6: final partition -----------------------------------------
ax = fig.add_subplot(gs[1, 2])
base_axes(ax, "6.  Result: 2 instances (parameter-free)")
ax.add_patch(FancyBboxPatch((-0.45, -0.45), 2.9, 0.9, boxstyle="round,pad=0.05",
             facecolor=CLUST_A, alpha=0.18, edgecolor=CLUST_A, lw=2))
ax.add_patch(FancyBboxPatch((2.55, -0.45), 2.9, 0.9, boxstyle="round,pad=0.05",
             facecolor=CLUST_B, alpha=0.18, edgecolor=CLUST_B, lw=2))
for u, v, w in [(1, 2, 0), (2, 3, 0), (4, 5, 0), (5, 6, 0)]:
    straight_edge(ax, u, v, GREEN, lw=4.5)
ax.plot([2.5, 2.5], [-0.7, 1.15], color=RED, lw=3.5, ls=(0, (3, 2)), zorder=5)
for i in range(1, 7):
    draw_node(ax, i, group[i], text="white")
ax.text(1.0, 1.5, "instance 1", ha="center", color=CLUST_A, fontsize=12, fontweight="bold")
ax.text(4.0, 1.5, "instance 2", ha="center", color=CLUST_B, fontsize=12, fontweight="bold")
ax.text(2.5, -1.05,
        "attractive edges set what merges;\n"
        "repulsive (multi-step) edges set where it must NOT",
        ha="center", va="center", fontsize=9.0)

out = __import__("os").path.join(__import__("os").path.dirname(__file__),
                                 "mutex_watershed_slide.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
