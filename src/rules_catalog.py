"""Catalog of violation types T1-T9: JSON display name, what is visible, normative source.

`source` entries are filled by the team from adilet.zan.kz only after checking the text.
An entry without a clause number is allowed; an invented clause number is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuleInfo:
    id: str
    name: str                 # value of the "type" field in the API JSON (Russian)
    visible: str              # what is actually visible on the video (template, Russian)
    source: dict = field(default_factory=dict)  # {"document": ..., "clause": ..., "url": ...}


RULES: dict[str, RuleInfo] = {
    "T1": RuleInfo("T1", "Наличные/ценности без присмотра",
                   "Деньги или инкассаторская сумка лежат на столе, кассир вне рабочего места"),
    "T2": RuleInfo("T2", "Уход с рабочего места без закрытия",
                   "Кассир покинул кассу, ноутбук остался включённым с открытой системой"),
    "T3": RuleInfo("T3", "Открытый сейф",
                   "Дверца сейфа остаётся открытой, пока кассир занят другим"),
    "T4": RuleInfo("T4", "Посторонний в кассовой зоне",
                   "Второй человек находится в зоне кассира / за столом во время операции"),
    "T5": RuleInfo("T5", "Личные вещи в кассовом помещении",
                   "Кассир вошёл с сумкой или положил куртку/сумку на рабочий стол"),
    "T6": RuleInfo("T6", "Посторонние предметы / напитки на рабочем месте",
                   "Стакан, напиток или журнал на столе рядом с наличными"),
    "T7": RuleInfo("T7", "Наличные вне зоны видимости",
                   "Купюры уходят под стол, к телу или в карман, вне зоны обзора камеры"),
    "T8": RuleInfo("T8", "Пересчёт без клиента",
                   "Клиент ушёл, кассир продолжает пересчитывать его деньги"),
    "T9": RuleInfo("T9", "Нарушение при приёме инкассации",
                   "Инкассатор оставил сумку и ушёл без передачи и проверки"),
    "T0": RuleInfo("T0", "Иное",
                   "Нетипичная ситуация, описанная по видимым признакам"),
}

NAME_TO_ID: dict[str, str] = {r.name: r.id for r in RULES.values()}


def type_id(type_name: str) -> str:
    """Map a JSON `type` string (or an id like 'T3') back to the catalog id; 'T0' if unknown."""
    if type_name in RULES:
        return type_name
    return NAME_TO_ID.get(type_name, "T0")
