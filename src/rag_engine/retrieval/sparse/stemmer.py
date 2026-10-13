"""From-scratch implementation of the Porter (1980) stemming algorithm.

M.F. Porter, "An algorithm for suffix stripping", Program 14(3), 1980.
Dependency-free and deterministic. Works on lowercase ASCII words; any token
containing a non-letter (digits, underscores, dots...) is returned unchanged so
identifiers like `k8s` or `asyncio.gather` are never mangled.
"""

from __future__ import annotations

_VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Number of VC sequences in the stem: [C](VC)^m[V]."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        vowel = not _is_consonant(stem, i)
        if prev_vowel and not vowel:
            m += 1
        prev_vowel = vowel
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """consonant-vowel-consonant, where the last consonant is not w, x or y."""
    n = len(word)
    if n < 3:
        return False
    return _is_consonant(word, n - 3) and not _is_consonant(word, n - 2) and _is_consonant(word, n - 1) and word[-1] not in "wxy"


def _replace(word: str, suffix: str, repl: str, min_m: int) -> tuple[str, bool]:
    """If word ends with suffix, return (new_word, True) when the remaining stem's
    measure > min_m, else (word, True) meaning 'matched but condition failed'."""
    stem = word[: -len(suffix)]
    if _measure(stem) > min_m:
        return stem + repl, True
    return word, True


def _step1a(w: str) -> str:
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def _step1b(w: str) -> str:
    if w.endswith("eed"):
        stem = w[:-3]
        return stem + "ee" if _measure(stem) > 0 else w
    for suf in ("ed", "ing"):
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if not _has_vowel(stem):
                return w
            w = stem
            if w.endswith(("at", "bl", "iz")):
                return w + "e"
            if _ends_double_consonant(w) and w[-1] not in "lsz":
                return w[:-1]
            if _measure(w) == 1 and _ends_cvc(w):
                return w + "e"
            return w
    return w


def _step1c(w: str) -> str:
    if w.endswith("y") and _has_vowel(w[:-1]):
        return w[:-1] + "i"
    return w


_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
    ("abli", "able"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
    ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
)
_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
)
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
    "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def _longest_suffix_rule(w: str, rules: tuple[tuple[str, str], ...]) -> str:
    # Porter specifies the longest matching suffix is the one tested.
    best: tuple[str, str] | None = None
    for suf, repl in rules:
        if w.endswith(suf) and (best is None or len(suf) > len(best[0])):
            best = (suf, repl)
    if best is None:
        return w
    return _replace(w, best[0], best[1], 0)[0]


def _step4(w: str) -> str:
    best = None
    for suf in _STEP4:
        if w.endswith(suf) and (best is None or len(suf) > len(best)):
            best = suf
    if best is None:
        return w
    stem = w[: -len(best)]
    if best == "ion" and not (stem and stem[-1] in "st"):
        return w
    return stem if _measure(stem) > 1 else w


def _step5(w: str) -> str:
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            w = stem
    if _measure(w) > 1 and _ends_double_consonant(w) and w.endswith("l"):
        w = w[:-1]
    return w


def porter_stem(word: str) -> str:
    """Return the Porter stem of a lowercase alphabetic word; other tokens unchanged."""
    if len(word) <= 2 or not word.isascii() or not word.isalpha():
        return word
    w = _step1a(word)
    w = _step1b(w)
    w = _step1c(w)
    w = _longest_suffix_rule(w, _STEP2)
    w = _longest_suffix_rule(w, _STEP3)
    w = _step4(w)
    return _step5(w)
