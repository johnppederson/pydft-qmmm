"""A module containing helper functions accessed by multiple classes.

Attributes:
    SELECTORS: Pairs of VMD selection keywords and the corresponding
        attribute and type to check in a system.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..constants import Subsystem

if TYPE_CHECKING:
    from pydft_qmmm import System


SELECTORS = {
    "element": ("elements", str),
    "atom": ("atoms", int),
    "index": ("atoms", int),
    "name": ("names", str),
    "residue": ("residues", int),
    "resid": ("residues", int),
    "resname": ("residue_names", str),
    "subsystem": ("subsystems", Subsystem),
}


def decompose(text: str) -> list[str]:
    """Decompose an atom selection query into meaningful components.

    Args:
        text: The atom selection query.

    Returns:
        The atom selection query broken into meaningful parts,
        demarcated by keywords.
    """
    line = [a.strip() for a in re.split(r"(not|or|and|\(|\))", text)]
    while "" in line:
        line.remove("")
    return line


def evaluate(text: str, system: System) -> frozenset[int]:
    """Evaluate a part of an atom selection query.

    Args:
        text: A single contained statement from an atom selection query.
        system: The system whose atoms will be selected by evaluating
            a single query statement.

    Returns:
        The set of atom indices selected by the query statement.
    """
    line = text.split(" ")
    category = SELECTORS[line[0].lower()]
    if " ".join(line).lower().startswith("atom name"):
        category = SELECTORS["name"]
        del line[1]
    elif " ".join(line).lower().startswith("residue name"):
        category = SELECTORS["resname"]
        del line[1]
    ret: frozenset[int] = frozenset({})
    if category[0] == "atoms":
        for string in line[1:]:
            value = category[1](string)
            ret = ret | frozenset({value})
    else:
        population = getattr(system, category[0])
        for string in line[1:]:
            value = category[1](string)
            indices = {i for i, x in enumerate(population) if x == value}
            ret = ret | frozenset(indices)
    return ret


def parens_slice(line: list[str], start: int) -> slice:
    """Find the slice of a query within parentheses.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement within
            parentheses begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        contained by parentheses.
    """
    flag = True
    count = 1
    index = start
    while flag:
        if line[index] == "(":
            count += 1
        if line[index] == ")":
            count -= 1
        if count == 0:
            stop = index
            flag = False
        index += 1
    return slice(start, stop)


def not_slice(line: list[str], start: int) -> slice:
    """Find the slice of a query modified by the 'not' keyword.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement modified by the
            'not' keyword begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        modified by the 'not' keyword.
    """
    flag = True
    count = 0
    index = start
    while flag:
        if line[index] == "(":
            count += 1
        if line[index] == ")":
            count -= 1
        if count == 0:
            stop = index + 1
            flag = False
        index += 1
    return slice(start, stop)


def and_slice(line: list[str], start: int) -> slice:
    """Find the slice of a query modified by the 'and' keyword.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement modified by the
            'and' keyword begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        modified by the 'and' keyword.
    """
    flag = True
    count = 0
    index = start
    while flag:
        if line[index] == "(":
            count += 1
        if line[index] == ")":
            count -= 1
        if count == 0 and line[index] != "not":
            stop = index + 1
            flag = False
        index += 1
    return slice(start, stop)


def or_slice(line: list[str], start: int) -> slice:
    """Find the slice of a query modified by the 'or' keyword.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement modified by the
            'or' keyword begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        modified by the 'or' keyword.
    """
    flag = True
    count = 0
    index = start
    while flag:
        if line[index] == "(":
            count += 1
        if line[index] == ")":
            count -= 1
        if index < len(line) - 1:
            if line[index+1] == "and":
                count += 1
        if index >= 1:
            if line[index-1] == "and":
                count -= 1
        if count == 0 and line[index] != "not":
            stop = index + 1
            flag = False
        index += 1
    return slice(start, stop)


def interpret(line: list[str], system: System) -> frozenset[int]:
    """Interpret a line of atom selection query language.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        system: The system whose atoms will be selected by interpreting
            the selection query.

    Returns:
        The set of atom indices selected by the query.

    .. note:: Based on the VMD atom selection rules.
    """
    # Precedence: () > not > and > or
    full = frozenset(range(len(system)))
    selection: frozenset[int] = frozenset({})
    count = 0
    while count < len(line):
        entry = line[count]
        if entry == "all":
            selection = selection | full
        elif entry == "none":
            selection = selection | frozenset({})
        elif entry.split(" ")[0].lower() in SELECTORS:
            selection = selection | evaluate(entry, system)
        elif entry == "(":
            indices = parens_slice(line, count + 1)
            selection = selection | interpret(line[indices], system)
            count = indices.stop
        elif entry == "not":
            indices = not_slice(line, count + 1)
            selection = selection | (full - interpret(line[indices], system))
            count = indices.stop
        elif entry == "and":
            indices = and_slice(line, count + 1)
            selection = selection & interpret(line[indices], system)
            count = indices.stop
        elif entry == "or":
            indices = or_slice(line, count + 1)
            selection = selection | interpret(line[indices], system)
            count = indices.stop
        else:
            print(f"{entry=}")
            raise ValueError
        count += 1
    return selection
