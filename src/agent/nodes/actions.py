from typing import Any, Dict, List
from datetime import date as _date, datetime as _dt, timedelta

from langchain_core.messages import AIMessage

from ..prompts import TEMPLATES
from ..state import AgentState

# В каждом узле:
# 1) наружу возвращаем человеческую фразу (messages)
# 2) внутренние шаги пишем в scratch["ops_note"]


def _merge_ops(state: Dict[str, Any], new_ops: List[str], base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Объединяет список операционных шагов в scratch['ops'].
    ВАЖНО: если передан base — начинаем с него (сохраняя такие поля как resume_at),
    иначе берём scratch из state.
    """
    scratch = dict(base) if base is not None else dict(state.get("scratch", {}) or {})
    cur = list(scratch.get("ops", []))
    for item in new_ops:
        if item not in cur:
            cur.append(item)
    scratch["ops"] = cur
    return scratch

from datetime import date as _date, datetime as _dt, timedelta
from typing import Optional

# ——— дата → "шестнадцатого сентября [2025 года]" ————————————————

_MONTHS_GEN = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}

_ONES_GEN = {
    1: "первого", 2: "второго", 3: "третьего", 4: "четвёртого",
    5: "пятого", 6: "шестого", 7: "седьмого", 8: "восьмого", 9: "девятого",
}
_TEENS_GEN = {
    10: "десятого", 11: "одиннадцатого", 12: "двенадцатого", 13: "тринадцатого",
    14: "четырнадцатого", 15: "пятнадцатого", 16: "шестнадцатого",
    17: "семнадцатого", 18: "восемнадцатого", 19: "девятнадцатого",
}
_TENS_GEN = {20: "двадцатого", 30: "тридцатого"}

def _day_ordinal_genitive(n: int) -> str:
    if n <= 0 or n > 31:
        return "уточнить дату"
    if n in _TEENS_GEN:
        return _TEENS_GEN[n]
    if n in _TENS_GEN:
        return _TENS_GEN[n]
    if n < 10:
        return _ONES_GEN[n]
    # 21–29, 31
    tens = (n // 10) * 10
    ones = n % 10
    tens_word = _TENS_GEN.get(tens, "")
    if ones == 0:
        return tens_word or "уточнить дату"
    return f"{tens_word[:-2]} { _ONES_GEN[ones] }"  # "двадцать первого", "тридцать первого"

def _parse_any_date(value: Optional[str]) -> Optional[_date]:
    if not value:
        return None
    s = value.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return _dt.strptime(s, fmt).date()
        except Exception:
            pass
    return None

def _date_propisi(dt: _date, with_year: bool = False) -> str:
    day = _day_ordinal_genitive(dt.day)
    month = _MONTHS_GEN.get(dt.month, "уточнить месяц")
    if with_year:
        return f"{day} {month} {dt.year} года"
    return f"{day} {month}"

def _fmt_date_propisi(value: Optional[str], with_year: bool = False) -> str:
    """Принимает ISO/формат DD.MM.YYYY/DD/MM/YYYY и возвращает дату прописью."""
    d = _parse_any_date(value)
    if not d:
        return "уточнить дату"
    return _date_propisi(d, with_year=with_year)


# A — вступление
def intro_node(state: AgentState) -> Dict[str, Any]:
    meta = state.get("meta", {}) or {}
    msg = (
        f"Добрый день! Меня зовут {meta.get('agent_name', '<ИМЯ>')}, я сотрудник компании Яндекс, "
        f"звоню по договору с Яндекс.Доставкой. С кем я могу переговорить по поводу задолженности "
        f"{meta.get('company', '<НАЗВАНИЕ КОМПАНИИ>')} перед Яндекс Доставкой?"
    )
    scratch = dict(state.get("scratch", {}) or {})
    scratch["greeted"] = True
    # продолжим на следующем ходу вопросом об ЛПР
    scratch["resume_at"] = "classify_lpr"
    return {"messages": [AIMessage(content=msg)], "stage": "Приветствие", "scratch": scratch, "route": "unknown"}

def ask_lpr_node(state: AgentState) -> Dict[str, Any]:
    meta = state.get("meta", {}) or {}
    company = meta.get("company", "<НАЗВАНИЕ КОМПАНИИ>")
    scratch = dict(state.get("scratch", {}) or {})
    attempt = int(scratch.get("ask_lpr_attempts", 0)) + 1
    scratch["ask_lpr_attempts"] = attempt
    scratch["resume_at"] = "classify_lpr"  # ← на следующем ходе классифицируем ЛПР

    if attempt == 1:
        text = (
            f"Подскажите, пожалуйста, вы являетесь лицом, принимающим решения по оплате в {company}? "
            "Если да — уточните, пожалуйста. Если нет — можно контакт ЛПР? Если номера нет, укажите e-mail — пришлю закрывающие документы."
        )
    elif attempt == 2:
        text = "Чтоб не задерживать вас: вы отвечаете за оплату? Если нет — оставьте, пожалуйста, контакт ЛПР или e-mail."
    else:
        text = "Нужно понять, с кем согласовывать оплату. Вы ЛПР? Если нет — прошу контакт ЛПР (телефон/e-mail)."

    return {"messages": [AIMessage(content=text)], "stage": "Идентификация ЛПР", "scratch": scratch, "route": "unknown"}

# C — не ЛПР
def not_lpr_node(state: AgentState) -> Dict[str, Any]:
    """Ветка 'не ЛПР': просим контакт и продолжаем в classify_lpr на следующем ходе (вдруг ответит ЛПР)."""
    scratch = dict(state.get("scratch", {}) or {})
    scratch["resume_at"] = "classify_lpr"
    text = (
        "Подскажите, пожалуйста, номер лица, принимающего решения, либо e-mail, на который продублировать закрывающие."
    )
    return {"messages": [AIMessage(content=text)], "stage": "Не ЛПР — запрашиваем контакт", "scratch": scratch, "route": "unknown"}

# D — ЛПР: проговариваем сумму и спрашиваем дату/причину
def lpr_prompt_node(state: AgentState) -> Dict[str, Any]:
    """Первый смысловой вопрос ЛПР; далее ждём ответ → на следующем ходе классифицируем причину."""
    scratch = dict(state.get("scratch", {}) or {})
    scratch["resume_at"] = "classify_reason"   # ← продолжение следующего хода
    meta = state.get("meta", {}) or {}
    debt_date = meta.get("act_date", "01.01.2024")
    debt_sum = meta.get("amount", "100 000 ₽")
    text = (
        f"Сумма задолженности по акту от {debt_date} — {debt_sum}.\n\n"
        "Подскажите дату, когда планируете оплатить задолженность?"
    )
    return {"messages": [AIMessage(content=text)], "stage": "Вопрос о способе/дате оплаты", "scratch": scratch, "route": "unknown"}

# n5/n6 — обработка названной даты
def named_date_within_week_node(state: "AgentState") -> dict[str, Any]:
    scratch_in = dict(state.get("scratch", {}) or {})
    target_propisi = _fmt_date_propisi(scratch_in.get("payment_date"), with_year=False)

    msg = (
        f"Хорошо, спасибо. Будем ожидать оплату {target_propisi}. До свидания!\n"
    )
    ops = [
        "Этап «Названа дата»",
        f"Ожидаем оплату {target_propisi}",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Названа дата",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

def named_date_over_week_node(state: "AgentState") -> dict[str, Any]:
    # контрольная дата = сегодня + 3 дня (её и произносим прописью)
    control_dt = (_date.today() + timedelta(days=3))
    control_propisi = _date_propisi(control_dt, with_year=False)

    msg = (
        "Согласно условиям договора срок оплаты услуг до 15 числа, на данный момент прошло уже существенно больше времени. "
        "С чем связана такая поздняя дата названная вами?\n"
        "Прошу вас в течение ближайших трёх дней произвести оплату за услуги сервиса, у вас есть такая возможность?\n"
        f"Будем ожидать оплату {control_propisi}.\n"
        "В случае отсутствия оплаты мы будем вынуждены передать вас в работу юридического отдела для взыскания в судебном порядке.\n"
        "До свидания!"
    )
    ops = [
        f"Контроль через 3 дня — {control_dt.strftime('%d.%m.%Y')}",
        "При нарушении — эскалация в юротдел",
        "Этап «Названа дата» (ускорение, > 7 дней)",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Названа дата",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

# n7 — предложение оплатить в течение недели
def propose_pay_within_week_node(state: AgentState) -> Dict[str, Any]:
    """Предложение оплатить в течение недели — дальше ждём согласие/несогласие."""
    scratch = dict(state.get("scratch", {}) or {})
    scratch["resume_at"] = "classify_agreement"
    text = "Прошу в течение недели произвести оплату сервиса. У вас есть такая возможность?"
    return {"messages": [AIMessage(content=text)], "stage": "Предложение оплатить в течение недели", "scratch": scratch, "route": "unknown"}


# n8 — согласие
def agree_named_date_node(state: AgentState) -> dict[str, Any]:
    date = (state.get("scratch", {}) or {}).get("payment_date", "00/00/0000")
    msg = TEMPLATES["agree_named_date"].format(target_date=date)
    ops = ["Этап «Названа дата»", f"Ожидаем оплату {date}"]
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Названа дата",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

# n9 — реструктуризация (предложение)
def restructuring_node(state: "AgentState") -> Dict[str, Any]:
    """
    Делает предложение о реструктуризации и переводит диалог на классификацию согласия.
    """
    base = dict(state.get("scratch", {}) or {})
    base["resume_at"] = "classify_restructuring_agreement"  # <- следующий ход должен пойти сюда

    text = (
        "Подскажите, если сделаем реструктуризацию долга, вам удобнее будет оплачивать?\n"
        "Предлагаю разбить задолженность на несколько платежей в течение текущего месяца. "
        "Согласны рассмотреть такой вариант?"
    )
    ops = [
        "Предложение реструктуризации",
        "Ожидаем ответ клиента (согласен/не согласен)",
    ]

    return {
        "messages": [AIMessage(content=text)],
        "stage": "Реструктуризация — предложение",
        "scratch": _merge_ops(state, ops, base=base),
        "route": "unknown",
    }



def restructuring_confirm_node(state: "AgentState") -> Dict[str, Any]:
    """
    Клиент согласен на реструктуризацию: отправляем подробное подтверждение.
    Даты подставляем: первая оплата — через 7 дней, последняя — не позже чем через 30 дней.
    """
    first_dt = (_date.today() + timedelta(days=7)).strftime("%d.%m.%Y")
    last_dt = (_date.today() + timedelta(days=30)).strftime("%d.%m.%Y")

    text = (
        "Предлагаю разбить вашу задолженность на несколько платежей в течение текущего месяца, "
        "с первым платежом в течение недели и последним не позднее чем через месяц с момента звонка.\n"
        "Согласны с данным предложением?\n"
        f"Давайте зафиксируем в письме: первую оплату ожидаем от вас {first_dt}, "
        f"последняя оплата не позднее {last_dt}.\n"
        "Прошу вас ответным письмом подтвердить наши договоренности.\n"
        "В случае отсутствия оплаты мы будем вынуждены передать вас в работу юридического отдела "
        "для подготовки судебной претензии.\n"
        "До свидания!"
    )

    ops = [
        "Этап «Реструктуризация»",
        f"Первая оплата {first_dt}; последняя — {last_dt}",
        "Ждём подтверждение по e-mail",
    ]
    scratch = dict(state.get("scratch", {}) or {})
    # После подтверждения дальнейшая классификация не нужна
    scratch.pop("resume_at", None)

    return {
        "messages": [AIMessage(content=text)],
        "stage": "Реструктуризация — подтверждение",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

# n10/n11 — уведомление о претензии
def pretrial_notice_node(state: AgentState) -> dict[str, Any]:
    meta = state.get("meta", {}) or {}
    msg = TEMPLATES["pretrial_notice"].format(debt_sum=meta.get("debt_sum", "<СУММА>"))
    ops = ["Этап «Уведомление о предсудебной претензии»", "Подготовить и отправить претензию"]
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Уведомление о предсудебной претензии",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

# n13 — нет ответа
def no_answer_letter_node(state: AgentState) -> dict[str, Any]:
    msg = TEMPLATES["first_debtor_letter"]
    ops = [
        "Отправить первое письмо должнику (шаблон)",
        "Поставить задачу «Звонок через 3 дня»",
        "В реестре: письмо-информационное",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Информационное письмо",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

# n14–n15 — дублирование закрывающих
def send_closings_node(state: AgentState) -> Dict[str, Any]:
    """Повтор закрывающих — после ответа снова спросим про дату оплаты (classify_reason)."""
    scratch = dict(state.get("scratch", {}) or {})
    scratch["resume_at"] = "classify_reason"
    text = "Сегодня направлю закрывающие документы. После получения подскажите, когда планируете оплату?"
    return {"messages": [AIMessage(content=text)], "stage": "Дублирование закрывающих", "scratch": scratch, "route": "unknown"}


# n16–n18 — уже оплатил
def already_paid_node(state: AgentState) -> dict[str, Any]:
    msg = TEMPLATES["already_paid"]
    ops = [
        "Запросить платёжное поручение",
        "Передать в Баланс на розыск и зачисление",
        "Контроль через 3 дня",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

# n19–n20 — акт сверки
def need_recon_node(state: AgentState) -> Dict[str, Any]:
    scratch = dict(state.get("scratch", {}) or {})
    scratch["resume_at"] = "classify_reason"
    return {
        "messages": [AIMessage(content="Отправлю акт сверки за весь период. После получения, пожалуйста, сообщите дату оплаты.")],
        "stage": "Акт сверки",
        "scratch": scratch,
        "route": "unknown",
    }

# n21–n23 — счёт
def need_invoice_node(state: AgentState) -> Dict[str, Any]:
    scratch = dict(state.get("scratch", {}) or {})
    scratch["resume_at"] = "create_duplicate_request"  # сначала запрос на дублирование, затем — причина
    return {
        "messages": [AIMessage(content="Счёт вышлю на e-mail. Укажите, достаточно ли счёта или продублировать закрывающие?")],
        "stage": "Нужен счёт",
        "scratch": scratch,
        "route": "unknown",
    }

def create_duplicate_request_node(state: AgentState) -> Dict[str, Any]:
    scratch = dict(state.get("scratch", {}) or {})
    scratch["resume_at"] = "classify_reason"  # после заявки снова обсудим дату оплаты
    return {
        "messages": [AIMessage(content="Оставил заявку на дублирование закрывающих. После получения подскажите дату оплаты.")],
        "stage": "Заявка на дублирование закрывающих",
        "scratch": scratch,
        "route": "unknown",
    }
