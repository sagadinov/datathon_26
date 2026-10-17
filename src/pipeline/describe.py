"""Russian descriptions of events: only what is visible, templated from rule details.

Normative sources are attached from docs/legal_sources.json when present (verified by the team).
"""
from __future__ import annotations

import json
from functools import lru_cache

from src.common import ROOT
from src.pipeline.rules import Event
from src.rules_catalog import RULES

OBJ_RU = {"cash": "наличные деньги", "money_bag": "инкассаторская сумка", "bag": "сумка", "jacket": "куртка",
          "cup": "стакан/чашка с напитком", "book": "журнал"}


@lru_cache(maxsize=1)
def legal_sources() -> dict:
    p = ROOT / "docs" / "legal_sources.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def describe(ev: Event) -> str:
    d, t0 = ev.details, f"{ev.t_start:.1f} с"
    dur = f"{max(0.5, ev.t_end - ev.t_start):.1f} с"
    if ev.type_id == "T1":
        return (f"На рабочем столе кассира видны {OBJ_RU.get(d.get('object'), 'ценности')}, при этом кассир отсутствует "
                f"на рабочем месте начиная с {t0} (не менее {dur}). Ценности остаются без присмотра.")
    if ev.type_id == "T2":
        lights = "свет в помещении выключен, " if d.get("lights_off") else ""
        return (f"Кассир покинул рабочее место с {t0}: {lights}экран ноутбука остаётся включённым "
                f"(видно свечение экрана). Рабочая система не закрыта перед уходом.")
    if ev.type_id == "T3":
        return (f"С {t0} дверца сейфа (металлического шкафа) открыта не менее {dur}, рядом с сейфом никого нет: "
                f"кассир занят другими действиями.")
    if ev.type_id == "T4":
        return (f"С {t0} в кассовой зоне одновременно находятся {d.get('persons', 3)} человека: помимо кассира и "
                f"клиента у рабочего стола присутствует посторонний во время операции (не менее {dur}).")
    if ev.type_id == "T5":
        return (f"С {t0} на рабочем столе кассира лежит {OBJ_RU.get(d.get('object'), 'личная вещь')} "
                f"(не менее {dur}): личные вещи находятся в кассовом помещении.")
    if ev.type_id == "T6":
        return (f"С {t0} на рабочем столе кассира находится {OBJ_RU.get(d.get('object'), 'посторонний предмет')} "
                f"рядом с наличными и документами (не менее {dur}).")
    if ev.type_id == "T7":
        return (f"С {t0} руки кассира опускаются под стол / к телу сразу после работы с наличными; купюры выходят "
                f"из зоны обзора камеры.")
    if ev.type_id == "T8":
        return (f"С {t0} клиент отсутствует у стола, на столе остаются наличные, кассир продолжает работу с ними "
                f"без клиента (не менее {dur}).")
    if ev.type_id == "T9":
        left = d.get("visitor_left_at")
        return (f"С {t0} на столе кассира появилась инкассаторская сумка, посетитель покинул кассу"
                f"{f' на {left:.1f} с' if left is not None else ''}; передача и проверка сумки в кадре не видны.")
    return f"С {t0} наблюдается нетипичная ситуация в кассовой зоне (не менее {dur})."


def rule_block(ev: Event) -> dict:
    info = RULES.get(ev.type_id, RULES["T0"])
    return {"id": info.id, "name": info.name, "score": round(ev.score, 2), "t_start_sec": round(ev.t_start, 1),
            "t_end_sec": round(ev.t_end, 1), "sources": legal_sources().get(ev.type_id, [])}
