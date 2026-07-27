"""
src/visualization.py
--------------------
Shared dataset visualization utilities for the Week 2 IR notebooks.

All three notebooks display the same SciFact dataset overview plots before
their model-specific experiments. This module provides that plot as a
reusable function so it only needs to be maintained in one place.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter


def plot_dataset_overview(
    corpus: dict,
    queries: dict,
    qrels_train: dict,
    qrels_test: dict,
    qrels_all: dict,
    save_path: str = None,
) -> None:
    """
    Render a 6-panel overview of the SciFact dataset.

    Panels
    ------
    1. Document length distribution (words per abstract)
    2. Query length distribution
    3. Relevant documents per query (bar chart)
    4. Top-15 most frequent words in the corpus
    5. Train / Test query split (pie chart)
    6. Key dataset statistics (text summary card)

    Parameters
    ----------
    corpus      : dict   { doc_id: doc_dict }  — from data_loader.load_corpus()
    queries     : dict   { query_id: text }     — from data_loader.load_queries()
    qrels_train : dict   { query_id: {doc_id: score} }
    qrels_test  : dict   { query_id: {doc_id: score} }
    qrels_all   : dict   merged qrels (train + test)
    save_path   : str, optional
        If provided, save the figure to this path instead of (or in addition to)
        displaying it. E.g. ``save_path="results/dataset_overview.png"``
    """
    # ── 1. Compute stats ────────────────────────────────────────────────────
    doc_lengths = [
        len((doc.get("title", "") + " " + doc.get("text", "")).split())
        for doc in corpus.values()
    ]
    query_lengths = [len(q.split()) for q in queries.values()]

    relevant_per_query = [
        sum(1 for score in docs.values() if score > 0)
        for docs in qrels_all.values()
    ]
    query_coverage = Counter(relevant_per_query)

    all_words = [
        w
        for doc in corpus.values()
        for w in (doc.get("title", "") + " " + doc.get("text", "")).lower().split()
        if w.isalpha() and len(w) > 3
    ]
    top_words = Counter(all_words).most_common(15)
    words, freqs = zip(*top_words)

    # ── 2. Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "SciFact Dataset — Visual Overview",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # ── Plot 1: Document length ──────────────────────────────────────────────
    ax1.hist(doc_lengths, bins=40, color="steelblue", edgecolor="white")
    ax1.axvline(
        np.mean(doc_lengths),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(doc_lengths):.0f}",
    )
    ax1.set_title("Document Length Distribution")
    ax1.set_xlabel("Words per Document")
    ax1.set_ylabel("Number of Documents")
    ax1.legend()

    # ── Plot 2: Query length ─────────────────────────────────────────────────
    ax2.hist(query_lengths, bins=20, color="darkorange", edgecolor="white")
    ax2.axvline(
        np.mean(query_lengths),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(query_lengths):.1f} words",
    )
    ax2.set_title("Query Length Distribution")
    ax2.set_xlabel("Words per Query")
    ax2.set_ylabel("Number of Queries")
    ax2.legend()

    # ── Plot 3: Relevant docs per query ─────────────────────────────────────
    labels = sorted(query_coverage.keys())
    counts = [query_coverage[l] for l in labels]
    ax3.bar([str(l) for l in labels], counts, color="mediumseagreen", edgecolor="white")
    ax3.set_title("Relevant Documents per Query")
    ax3.set_xlabel("Number of Relevant Docs")
    ax3.set_ylabel("Number of Queries")

    # ── Plot 4: Top-15 words ─────────────────────────────────────────────────
    ax4.barh(list(words)[::-1], list(freqs)[::-1], color="slateblue")
    ax4.set_title("Top 15 Words in Corpus")
    ax4.set_xlabel("Frequency")

    # ── Plot 5: Train / Test split ───────────────────────────────────────────
    split_data   = [len(qrels_train), len(qrels_test)]
    split_labels = [
        f"Train\n({len(qrels_train)} queries)",
        f"Test\n({len(qrels_test)} queries)",
    ]
    ax5.pie(
        split_data,
        labels=split_labels,
        autopct="%1.1f%%",
        colors=["steelblue", "darkorange"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax5.set_title("Train / Test Query Split")

    # ── Plot 6: Summary card ─────────────────────────────────────────────────
    ax6.axis("off")
    summary = [
        ("Total Documents",  f"{len(corpus):,}"),
        ("Total Queries",    f"{len(queries):,}"),
        ("Avg Doc Length",   f"{np.mean(doc_lengths):,.0f} words"),
        ("Avg Query Length", f"{np.mean(query_lengths):.1f} words"),
        ("Train Queries",    f"{len(qrels_train):,}"),
        ("Test Queries",     f"{len(qrels_test):,}"),
    ]
    y = 0.92
    for label, value in summary:
        ax6.text(0.05, y, label, fontsize=11, color="gray",  transform=ax6.transAxes)
        ax6.text(0.95, y, value, fontsize=11, color="black", fontweight="bold",
                 transform=ax6.transAxes, ha="right")
        ax6.plot([0.05, 0.95], [y - 0.04, y - 0.04],
                 color="#eeeeee", linewidth=0.8, transform=ax6.transAxes)
        y -= 0.15
    ax6.set_title("Dataset Summary", fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    print("Dataset visualization complete.")
