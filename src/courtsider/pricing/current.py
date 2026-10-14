"""Pre-match fair 1X2 for today's matches from the Dixon–Coles book fit on the latest football-data seasons."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process

from courtsider.paths import PROCESSED
from courtsider.pricing.dixon_coles import ModelBook

# ESPN display names -> football-data's abbreviations, where fuzzy matching alone is unreliable
ALIASES = {
    r"\bmanchester\b": "man",
    r"\batl[eé]tico( de)? madrid\b": "ath madrid",
    r"\bathletic (club|bilbao)\b": "ath bilbao",
    r"\breal sociedad\b": "sociedad",
    r"\breal betis\b": "betis",
    r"\brayo vallecano\b": "vallecano",
    r"\bcelta vigo\b": "celta",
    r"\bespanyol\b": "espanol",
    r"\bdeportivo alav[eé]s\b|\balav[eé]s\b": "alaves",
    r"\breal oviedo\b": "oviedo",
    r"\bnottingham forest\b": "nott'm forest",
    r"\bwolverhampton wanderers\b": "wolves",
    r"\btottenham hotspur\b": "tottenham",
    r"\bbrighton & hove albion\b": "brighton",
    r"\bparis saint-germain\b": "paris sg",
    r"\binternazionale\b|\binter milan\b": "inter",
    r"\bac milan\b": "milan",
    r"\bbayern munich\b": "bayern munich",
    r"\bborussia dortmund\b": "dortmund",
    r"\bbayer leverkusen\b": "leverkusen",
    r"\bborussia m[oö]nchengladbach\b": "m'gladbach",
    r"\beintracht frankfurt\b": "ein frankfurt",
}


def _norm(name: str) -> str:
    n = name.lower()
    for pat, rep in ALIASES.items():
        n = re.sub(pat, rep, n)
    return n


@lru_cache(maxsize=1)
def _book() -> tuple[ModelBook, dict[str, list[str]]]:
    r = pd.read_parquet(PROCESSED / "results.parquet")
    recent = r[r.date >= r.date.max() - pd.Timedelta(days=365)]
    teams = {lg: sorted(set(g.home) | set(g.away)) for lg, g in recent.groupby("league")}
    return ModelBook(r), teams


def _match(name: str, teams: list[str]) -> str | None:
    hit = process.extractOne(_norm(name), [t.lower() for t in teams], scorer=fuzz.WRatio, score_cutoff=88)
    return teams[hit[2]] if hit else None


def fair_1x2(home: str, away: str) -> dict | None:
    """{'league', 'home', 'draw', 'away'} if both teams are in the same one of the five leagues, else None."""
    book, teams = _book()
    for league, names in teams.items():
        h, a = _match(home, names), _match(away, names)
        if h and a and h != a:
            p = book.model_for(league, pd.Timestamp.today()).outcome_probs(h, a)
            return dict(
                league=league, home=round(float(p[0]), 3), draw=round(float(p[1]), 3), away=round(float(p[2]), 3)
            )
    return None
