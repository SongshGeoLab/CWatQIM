#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Where the "friend" network comes from.

The social term and social learning both run over `City.friends`. Until now
that network had exactly one source: `Province.update_graph` linked every pair
of cities **inside a province** with probability `l_p` (1.0 in the shipped
configuration), so the graph was a disjoint union of eight province-sized
cliques with no cross-province edge at all.

This module adds a second source — the **observed** collaboration network: 69
prefectures and 200 undirected edges taken from 76 formal cooperation documents
published on prefecture government websites between 2020 and 2024. It is
selected by `model.network`:

* `within_province` — the clique-per-province graph above. The default, and
  bit-identical to the model before this module existed.
* `observed` — the collaboration network, cropped to the city agents.

Warning:
    **Switching to `observed` replaces a modelling assumption, not a parameter,
    and three mismatches come with it. They are quantified in
    `water_quota_analysis.analysis.collab_network`'s module docstring (the
    single home for that account) and in issue #139; the two that change what
    the model does are repeated here because they decide how results should be
    read:**

    1. *Coverage.* The network's 69 prefectures and the model's 58 canonical
       names share 57. Cropping to the agents keeps 145 of 200 edges (72.5%),
       and leaves **four agents isolated** — Gannan, Jiyuan, Laiwu and
       Yangquan. Isolation is safe but not neutral: an agent with no friends
       has `s = 1` in every branch, so the social channel is switched off for
       it entirely, and it never learns from anyone.
    2. *Time.* The network covers 2020-2024; the model runs 1980-2012. The two
       do not overlap. Nothing in the data can fix this, so any result under
       `observed` is a statement about *this topology*, not about the network
       that existed during the simulated period.

    The third mismatch is the point of the exercise rather than an obstacle:
    the observed network is cross-regional, while `within_province` has no
    cross-province edge, so the two graphs differ in kind.

Warning:
    **This switch reaches further than the culture parameters do, and the
    difference is easy to get wrong.** `City.s_grid` / `City.s_group` enter only
    the social term, so a scenario that keeps the social term out of utility
    (`never`) is provably immune to them. The network is not confined that way:
    `City.change_mind` runs over `City.friends` **unconditionally** at the end
    of every `City.step`, whatever `include_s` says. Social learning therefore
    rides on this graph in every scenario, and `never` is **not** immune to it.

    Do not carry the `never`-is-invariant reasoning over from
    `cwatqim.core.culture`. Any comparison across network modes has to include
    the control arm rather than reuse it.

Note:
    The edge list is built by `scripts/build_social_network_edges.py` into
    `data/processed/city_network_edges.csv`. That script prints what cropping
    costs on every run — the shrinkage is a property of the data and must not
    become invisible.

Google-style docs are used throughout.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd

#: `model.network` 的取值域。
NETWORK_MODES: tuple[str, ...] = ("within_province", "observed")

#: 边表的两列，与 `collab_network.MODEL_EDGE_COLUMNS` 同一套 schema。
EDGE_COLUMNS: tuple[str, ...] = ("City_ID_a", "City_ID_b")


@lru_cache(maxsize=None)
def load_city_links(path: str) -> Mapping[int, tuple[int, ...]]:
    """Read the observed network as an adjacency mapping, once per run.

    Args:
        path: Path to `city_network_edges.csv`.

    Returns:
        Mapping from `City_ID` to its neighbours' `City_ID`s, ascending. Only
        cities that have at least one edge appear; the caller must treat a
        missing key as "no friends" rather than as an error, because isolation
        is a real property of this data (four agents have it).

    Raises:
        FileNotFoundError: If the table is missing.
        KeyError: If a required column is absent.
        ValueError: If an edge is a self-loop, or the same pair appears twice.

    Note:
        The returned mapping is symmetric: each undirected row is expanded into
        both directions here, so callers never have to remember to do it. The
        file itself stores each edge once, with `City_ID_a < City_ID_b`.

    Note:
        `lru_cache` keeps this to one file read per process rather than one per
        simulated year — `update_network` runs every step.
    """
    table = Path(path)
    if not table.exists():
        raise FileNotFoundError(
            f"协作网络的边表不存在：{table}。"
            "先跑 `python scripts/build_social_network_edges.py` 生成。"
        )
    frame = pd.read_csv(table)
    missing = [c for c in EDGE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"{table} 里没有这些列：{missing}")

    pairs: dict[int, list[int]] = {}
    seen: set[tuple[int, int]] = set()
    for a, b in zip(frame[EDGE_COLUMNS[0]], frame[EDGE_COLUMNS[1]]):
        a, b = int(a), int(b)
        if a == b:
            raise ValueError(f"{table} 里有自环：C{a}。城市不能是自己的邻居")
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError(f"{table} 里有重复的边：C{key[0]}–C{key[1]}")
        seen.add(key)
        pairs.setdefault(a, []).append(b)
        pairs.setdefault(b, []).append(a)
    return {city: tuple(sorted(set(nbrs))) for city, nbrs in pairs.items()}


def isolated_city_ids(
    links: Mapping[int, tuple[int, ...]], city_ids: "list[int] | tuple[int, ...]"
) -> list[int]:
    """Which agents this network leaves with no friends at all.

    Deliberately a query rather than a guard: isolation is a genuine property
    of the observed network (four agents have it), so it must be reportable
    without being fatal. What would be a bug is *silent* isolation, which is
    why the model logs this at setup.

    Args:
        links: `load_city_links`'s product.
        city_ids: Every agent's `City_ID`.

    Returns:
        Sorted `City_ID`s absent from `links`.
    """
    return sorted(set(int(c) for c in city_ids) - set(links))
