"""Phase 14 - Lesson 10: Skill Libraries and Lifelong Learning (Voyager) - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import itertools
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from main import Skill, SkillLibrary, _craft_iron_pick_v2, _gather_sticks, _mine, _place_table


def noop(context: dict[str, Any]) -> str:
    return "ok"


def lesson_library() -> SkillLibrary:
    """The four skills main.py ends up with."""
    lib = SkillLibrary()
    lib.register(Skill("mine_ore", "mine iron ore from nearby rock formations", "mine(3)", _mine,
                       tags=("gather", "ore")))
    lib.register(Skill("place_crafting_table", "place a crafting table at current position", "place_table()",
                       _place_table, tags=("setup", "crafting")))
    lib.register(Skill("gather_sticks", "gather sticks from tree or broken planks", "gather(2, stick)",
                       _gather_sticks, tags=("gather", "stick")))
    lib.register(Skill("craft_iron_pickaxe", "craft an iron pickaxe using ore, sticks, and a crafting table",
                       "craft('iron_pickaxe')", _craft_iron_pick_v2, tags=("craft", "tool"),
                       depends_on=("mine_ore", "gather_sticks", "place_crafting_table")))
    return lib


# ---------------------------------------------------------------------------
# Exercise 1 - dependency-cycle detection when composing
#
# main has no compose(); composition is `depends_on` plus topo_order. With
# A -> B -> A, topo_order does not loop forever, which is worse: it returns an
# order and execute() runs B before the A it needs.
#
# Error, not warning. No valid order exists for a cycle, so every execution is
# wrong, and a warning in a log does not stop one. Refusing at write time puts
# the failure in front of whoever just created the loop.
# ---------------------------------------------------------------------------

class CycleError(ValueError):
    pass


def path_to(lib: SkillLibrary, start: str, target: str, seen: set[str] | None = None) -> list[str] | None:
    """Dependency path start -> ... -> target, or None."""
    seen = set() if seen is None else seen
    if start == target:
        return [start]
    if start in seen:
        return None
    seen.add(start)
    skill = lib.get(start)
    for dep in (skill.depends_on if skill else ()):
        rest = path_to(lib, dep, target, seen)
        if rest:
            return [start] + rest
    return None


def compose(lib: SkillLibrary, skill: Skill) -> str:
    """register(), refusing a skill whose dependencies lead back to itself."""
    for dep in skill.depends_on:
        path = path_to(lib, dep, skill.name)
        if path:
            raise CycleError(" -> ".join([skill.name] + path))
    return lib.register(skill)


def ex1_cycle_detection() -> None:
    wood = Skill("gather_wood", "gather wood with an axe", "chop()", noop, depends_on=("craft_axe",))
    axe = Skill("craft_axe", "craft an axe from wood", "craft('axe')", noop, depends_on=("gather_wood",))

    unchecked = SkillLibrary()
    unchecked.register(wood)
    unchecked.register(axe)
    print(f"  main, no check : order {unchecked.topo_order('craft_axe')} (gather_wood runs without its axe)")

    checked = SkillLibrary()
    compose(checked, wood)
    try:
        compose(checked, axe)
        raise AssertionError("the cycle should have been refused")
    except CycleError as error:
        print(f"  compose()      : CycleError: {error}")
        assert str(error) == "craft_axe -> gather_wood -> craft_axe"
    assert unchecked.topo_order("craft_axe") == ["gather_wood", "craft_axe"]
    assert checked.list_names() == ["gather_wood"]          # nothing half-registered
    assert compose(lesson_library(), Skill("mine_coal", "mine coal", "mine()", noop,
                                           depends_on=("craft_iron_pickaxe",))).startswith("registered")


# ---------------------------------------------------------------------------
# Exercise 2 - version pinning
#
# main.register overwrites a skill in place, so every parent picks up the new
# child on its next run. Here every version is kept, a dependency written as
# "crafting@1" resolves to that exact version, and a bare "crafting" still
# means latest. Upgrading a pinned parent becomes an explicit edit.
# ---------------------------------------------------------------------------

class VersionedLibrary(SkillLibrary):
    def __init__(self) -> None:
        super().__init__()
        self._versions: dict[tuple[str, int], Skill] = {}

    def register(self, skill: Skill, dedup: bool = True) -> str:
        message = super().register(skill, dedup)
        current = self._skills[skill.name]
        self._versions[(current.name, current.version)] = replace(current, history=list(current.history))
        return message

    def resolve(self, ref: str) -> Skill | None:
        name, _, pin = ref.partition("@")
        return self._versions.get((name, int(pin))) if pin else self._skills.get(name)

    def execute(self, name: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = {} if context is None else context
        context.setdefault("log", [])
        done: set[tuple[str, int]] = set()

        def run(ref: str) -> None:
            skill = self.resolve(ref)
            if skill is None:
                raise LookupError(f"missing skill: {ref}")
            if (skill.name, skill.version) in done:
                return
            done.add((skill.name, skill.version))
            for dep in skill.depends_on:
                run(dep)
            context["log"].append(f"ran {skill.name} v{skill.version}: {skill.fn(context)}")

        try:
            run(name)
            context["failed"] = False
        except Exception as error:
            context["log"].append(f"error: {type(error).__name__}: {error}")
            context["failed"] = True
        return context


def ex2_version_pinning() -> None:
    lib = VersionedLibrary()
    lib.register(Skill("crafting", "craft a tool", "recipe_v1()", lambda ctx: "used recipe v1"))
    lib.register(Skill("pinned_kit", "build the starter kit", "crafting@1(); pack()", lambda ctx: "kit packed",
                       depends_on=("crafting@1",)))
    lib.register(Skill("floating_kit", "build the starter kit", "crafting(); pack()", lambda ctx: "kit packed",
                       depends_on=("crafting",)))
    print("  " + lib.register(Skill("crafting", "craft a tool", "recipe_v2()", lambda ctx: "used recipe v2")))

    pinned, floating = lib.execute("pinned_kit")["log"], lib.execute("floating_kit")["log"]
    print(f"  pinned_kit   (crafting@1): {pinned[0]}")
    print(f"  floating_kit (crafting)  : {floating[0]}")
    assert pinned[0] == "ran crafting v1: used recipe v1"       # not silently upgraded
    assert floating[0] == "ran crafting v2: used recipe v2"
    assert lib.execute("pinned_kit")["failed"] is False
    assert lib.execute("crafting@7")["failed"] is True          # an unknown pin fails loudly


# ---------------------------------------------------------------------------
# Exercise 3 - BM25 instead of token overlap, retrieval@5 on 50 skills
#
# The exercise allows sentence-transformers or a stdlib BM25; this is the
# BM25 (the same scoring as lesson 07). The 50 skills are 10 verbs x 5 objects
# with descriptions of three lengths, which is what a real library looks like
# after a few authors. Token overlap ranks by shared words over total words, so
# boilerplate-heavy descriptions bury a terse one. BM25 weights the rare words
# (the verb, the object) and corrects for length.
# ---------------------------------------------------------------------------

VERBS = ["mine", "craft", "smelt", "gather", "build", "repair", "trade", "cook", "farm", "explore"]
OBJECTS = ["iron", "wood", "stone", "wheat", "diamond"]
TEMPLATES = ["{verb} {obj}",
             "use this skill to {verb} {obj} for the base",
             "a skill the agent can use when it needs to {verb} some {obj} in the world or for the base"]


def fifty_skill_library() -> SkillLibrary:
    lib = SkillLibrary()
    for i, (verb, obj) in enumerate(itertools.product(VERBS, OBJECTS)):
        lib.register(Skill(f"{verb}_{obj}", TEMPLATES[i % 3].format(verb=verb, obj=obj), f"{verb}('{obj}')", noop))
    return lib


def bm25_search(lib: SkillLibrary, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> list[str]:
    docs = {name: lib.get(name).description.lower().split() for name in lib.list_names()}
    n = len(docs)
    avg_len = sum(len(d) for d in docs.values()) / n
    df = Counter(term for d in docs.values() for term in set(d))
    terms = set(query.lower().split())

    def score(doc: list[str]) -> float:
        tf = Counter(doc)
        return sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                   * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avg_len))
                   for t in terms if t in tf)

    return sorted(docs, key=lambda name: -score(docs[name]))[:top_k]


def ex3_bm25_retrieval() -> None:
    lib = fifty_skill_library()
    overlap_hits = bm25_hits = 0
    for verb, obj in itertools.product(VERBS, OBJECTS):
        query = f"what skill do i use to {verb} some {obj} for the base"
        gold = f"{verb}_{obj}"
        overlap_hits += gold in [skill.name for _, skill in lib.search(query, top_k=5)]
        bm25_hits += gold in bm25_search(lib, query)
    print(f"  retrieval@5 over {len(lib.list_names())} skills: token overlap {overlap_hits / 50:.2f}   "
          f"BM25 {bm25_hits / 50:.2f}")
    assert bm25_hits == 50
    assert overlap_hits < bm25_hits


# ---------------------------------------------------------------------------
# Exercise 4 - a curriculum agent: propose 5 missing skills
#
# SCRIPTED PROPOSER: an LLM would read a prose description of the domain. Here
# the domain is a prerequisite table and the proposer is a sort, which keeps
# Voyager's idea intact: propose what is just beyond current ability first
# (prerequisites already in the library), then what unlocks the most.
#
# "Call it weekly" is a scheduler entry (cron `0 9 * * 1`, or a Windows
# scheduled task) that calls propose_skills. No scheduler is built here.
# ---------------------------------------------------------------------------

MINECRAFT = {       # skill -> prerequisites
    "mine_ore": (), "gather_sticks": (), "place_crafting_table": (), "gather_wood": (), "mine_stone": (),
    "craft_iron_pickaxe": ("mine_ore", "gather_sticks", "place_crafting_table"),
    "build_furnace": ("mine_stone", "place_crafting_table"),
    "smelt_iron": ("mine_ore", "build_furnace"),
    "build_shelter": ("gather_wood",),
    "mine_coal": ("craft_iron_pickaxe",),
    "craft_torch": ("gather_sticks", "mine_coal"),
    "mine_diamond": ("craft_iron_pickaxe", "craft_torch"),
    "craft_diamond_pickaxe": ("mine_diamond", "gather_sticks", "place_crafting_table"),
    "craft_iron_armor": ("smelt_iron", "place_crafting_table"),
}


def propose_skills(lib: SkillLibrary, domain: dict[str, tuple[str, ...]], k: int = 5) -> list[str]:
    have = set(lib.list_names())
    missing = [skill for skill in domain if skill not in have]

    def blocked(skill: str) -> bool:
        return any(pre not in have for pre in domain[skill])

    def unlocks(skill: str) -> int:
        return sum(skill in domain[other] for other in missing)

    return sorted(missing, key=lambda s: (blocked(s), -unlocks(s), s))[:k]


def ex4_curriculum() -> None:
    lib = lesson_library()
    week_1 = propose_skills(lib, MINECRAFT)
    learnable = [s for s in week_1 if all(pre in lib.list_names() for pre in MINECRAFT[s])]
    print(f"  week 1 proposals : {week_1}")
    print(f"  learnable now    : {learnable}")
    for name in learnable:
        lib.register(Skill(name, name.replace("_", " "), f"{name}()", noop, depends_on=MINECRAFT[name]))
    week_2 = propose_skills(lib, MINECRAFT)
    print(f"  week 2 proposals : {week_2}")
    assert len(week_1) == 5 and not set(week_1) & {"mine_ore", "craft_iron_pickaxe"}
    assert week_1[:3] == learnable == ["gather_wood", "mine_coal", "mine_stone"]
    assert not set(week_2) & set(learnable)
    assert all(pre in lib.list_names() for pre in MINECRAFT[week_2[0]])     # the frontier moved


# ---------------------------------------------------------------------------
# Exercise 5 - port the library to the Claude Agent SDK skill schema
#
# In the SDK a skill is a file: .claude/skills/<name>/SKILL.md, with YAML
# frontmatter (name, description) and Markdown instructions. There is no call
# to register one; the SDK finds them on disk.
#
# What changes about discoverability: the toy library is pulled - the runtime
# runs search(query) and a skill the query does not match stays invisible. In
# the SDK the name and description of every skill are discovered at startup,
# the model decides when one applies, and only then is the body loaded. So
# there is no similarity score and no top-k. The description stops being a
# search index and becomes a trigger the model reads, which is why it has to
# say when to use the skill. Dependency order has no field; it becomes
# instructions in the body.
#
# The export is written to a temp folder and read back. Loading it in a live
# SDK session was NOT done here.
# ---------------------------------------------------------------------------

def to_skill_md(skill: Skill) -> tuple[str, str]:
    """(folder name, SKILL.md text) for one skill."""
    name = skill.name.replace("_", "-")
    when = f" Use when the task involves {', '.join(skill.tags)}." if skill.tags else ""
    description = f"{skill.description[0].upper()}{skill.description[1:]}.{when}"
    lines = ["---", f"name: {name}", f"description: {json.dumps(description)}", "---", "", f"# {name}", ""]
    if skill.depends_on:
        lines += ["Run these skills first, in this order:",
                  *[f"- {dep.replace('_', '-')}" for dep in skill.depends_on], ""]
    lines += ["Then run:", "", "```", skill.code, "```", "", f"Library version: {skill.version}"]
    return name, "\n".join(lines) + "\n"


def export_library(lib: SkillLibrary, root: Path) -> list[Path]:
    paths = []
    for skill_name in lib.list_names():
        name, text = to_skill_md(lib.get(skill_name))
        path = root / ".claude" / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def ex5_agent_sdk_skills() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_library(lesson_library(), Path(tmp))
        for path in paths:
            frontmatter = path.read_text(encoding="utf-8").split("---")[1]
            fields = dict(line.split(": ", 1) for line in frontmatter.strip().splitlines())
            description = json.loads(fields["description"])
            assert fields["name"] == path.parent.name
            assert re.fullmatch(r"[a-z0-9-]{1,64}", fields["name"])
            assert len(description) <= 1024 and "Use when" in description
        print(f"  wrote {len(paths)} skills under .claude/skills/: {[p.parent.name for p in paths]}")
        print("  " + paths[0].read_text(encoding="utf-8").replace("\n", "\n  ").rstrip())
    assert len(paths) == 4


if __name__ == "__main__":
    print("Phase 14 - Lesson 10: Skill Libraries (Voyager) - exercises")
    for exercise in (ex1_cycle_detection, ex2_version_pinning, ex3_bm25_retrieval,
                     ex4_curriculum, ex5_agent_sdk_skills):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
