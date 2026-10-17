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


def _s(t: float | None) -> str:
    return f"{t:.1f} с" if t is not None else "—"


def describe(ev: Event) -> str:
    d, t0 = ev.details, _s(ev.t_start)
    dur = f"{max(0.5, ev.t_end - ev.t_start):.1f} с"
    obj = OBJ_RU.get(d.get("object"), "ценности")
    if ev.type_id == "T1":
        if d.get("variant") == "B":
            return (f"На рабочем столе лежат {obj} предыдущего клиента; в {_s(d.get('entered_t'))} в помещение входит "
                    f"следующий посетитель, пока {obj} остаются на столе.")
        return (f"На рабочем столе кассира видны {obj}, при этом кассир покинул рабочее место начиная с {t0} "
                f"(отсутствует не менее {dur}, у стола никого нет). Ценности остаются без присмотра.")
    if ev.type_id == "T2":
        if d.get("why") == "items":
            return (f"Кассир встал и ушёл с рабочего места с {t0} (выход из помещения: {_s(d.get('exit_t'))}); "
                    f"на столе остались посторонние предметы (стакан/журнал), ноутбук открыт.")
        lights = "свет в помещении выключен, " if d.get("lights_off") else ""
        return (f"Кассир покинул рабочее место с {t0} (выход из помещения: {_s(d.get('exit_t'))}): {lights}"
                f"ноутбук открыт, экран остаётся включённым. Рабочее место не закрыто перед уходом.")
    if ev.type_id == "T3":
        return (f"С {t0} дверца сейфа (металлического шкафа) открыта не менее {dur}, у сейфа никого нет: "
                f"кассир занят другими действиями.")
    if ev.type_id == "T4":
        if d.get("variant") == "b":
            return (f"С {t0} посторонний находится на стороне кассира (за рабочим столом, у кресла кассира) во время "
                    f"работы кассира (не менее {dur}).")
        return (f"С {t0} в кассовом помещении одновременно находятся два посетителя: второй человек подошёл к столу, "
                f"пока обслуживается первый (не менее {dur}).")
    if ev.type_id == "T5":
        return (f"Вошедший положил на рабочий стол кассира {obj} (с {t0}, не менее {dur}): личные вещи "
                f"находятся в кассовом помещении.")
    if ev.type_id == "T6":
        return (f"С {t0} на рабочем столе кассира находится {obj} рядом с наличными во время работы с ними "
                f"(не менее {dur}).")
    if ev.type_id == "T7":
        v = d.get("variant")
        if v == "B":
            return (f"Пачка купюр, лежавшая на столе, исчезает со стола в {t0}; у её места нет рук клиента, сейф закрыт, "
                    f"руки кассира ниже края стола. Наличные вышли из зоны обзора камеры.")
        if v == "C":
            return (f"С {t0} кассир стоит с наличными, прижатыми к телу ниже линии стола, без пересчёта на машинке; "
                    f"купюры вне зоны обзора камеры.")
        return (f"В {t0} купюры в руках кассира уходят ниже края стола / за корпус, после чего руки не видны; "
                f"на столе и у клиента наличных нет. Наличные вне зоны обзора камеры.")
    if ev.type_id == "T8":
        return (f"Клиент отошёл от стола в {_s(d.get('client_left_t'))}; с {t0} кассир продолжает пересчёт наличных "
                f"на машинке без клиента (не менее {dur}).")
    if ev.type_id == "T9":
        return (f"В {t0} посетитель положил инкассаторскую сумку на стол и покинул помещение в "
                f"{_s(d.get('visitor_left_at'))}; сумка остаётся на столе, передача и проверка в кадре не видны.")
    if ev.type_id == "T10":
        return (f"Кассир входит в пустое кассовое помещение и отпускает дверь через {d.get('dwell_s', 0)} с после "
                f"порога ({t0}), не закрыв и не заперев её.")
    return f"С {t0} наблюдается нетипичная ситуация в кассовой зоне (не менее {dur})."


def rule_block(ev: Event) -> dict:
    info = RULES.get(ev.type_id, RULES["T0"])
    return {"id": info.id, "name": info.name, "score": round(ev.score, 2), "t_start_sec": round(ev.t_start, 1),
            "t_end_sec": round(ev.t_end, 1), "sources": legal_sources().get(ev.type_id, [])}
