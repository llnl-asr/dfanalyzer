from betterset import BetterSet as S


def unique_set_flatten_pd(s):
    return frozenset(S.flatten(s.dropna()))


def unique_set_pd(s):
    return frozenset(S(s.dropna().unique().tolist()))


unique_set_flatten_pd.__name__ = "unique"
unique_set_pd.__name__ = "unique"
