from typing import Any

from langchain_core.messages import AIMessage

from ..prompts import TEMPLATES
from ..state import AgentState

# В каждом узле:
# 1) наружу возвращаем человеческую фразу (messages)
# 2) внутренние шаги пишем в scratch["ops_note"]

def _merge_ops(state: AgentState, notes: list[str]) -> dict[str, Any]:
    scratch = dict(state.get("scratch", {}) or {})
    scratch["ops_note"] = "; ".join(notes)
    return scratch

# A — вступление
def intro_node(state: AgentState) -> dict[str, Any]:
    meta = state.get("meta", {}) or {}
    msg = TEMPLATES["intro"].format(
        agent_name=meta.get("agent_name", "<ИМЯ>"),
        company=meta.get("company", "<НАЗВАНИЕ КОМПАНИИ>"),
    )
    # Внутренние шаги на старте обычно не требуются
    return {"messages": [AIMessage(content=msg)], "route": "classify_lpr"}

def ask_lpr_node(state: AgentState) -> dict[str, Any]:
    meta = state.get("meta", {}) or {}
    company = meta.get("company", "<НАЗВАНИЕ КОМПАНИИ>")
    client_text = (
        f"Подскажите, с кем можно обсудить вопрос по задолженности {company}? "
        "Вы являетесь лицом, принимающим решения?"
    )
    ops = [
        "Идентифицировать ЛПР: спросить прямо, при отказе — запросить контакт ЛПР",
        "При отсутствии номера — запросить e-mail для дублирования закрывающих",
    ]
    scratch = dict(state.get("scratch", {}) or {})
    scratch["ops_note"] = "; ".join(ops)
    return {
        "messages": [AIMessage(content=client_text)],
        "stage": "Идентификация ЛПР",
        "scratch": scratch,
        "route": "unknown",
    }

# C — не ЛПР
def not_lpr_node(state: AgentState) -> dict[str, Any]:
    client_text = (
        "Подскажите, пожалуйста, с кем можно обсудить вопрос по оплате? "
        "Можете оставить контакт ЛПР? Если номера нет, укажите e-mail — пришлю документы."
    )
    ops = [
        "Уточнить ФИО и статус собеседника",
        "Запросить номер ЛПР; при отсутствии — e-mail для дублирования закрывающих",
        "Отправить инфописьмо с закрывающими",
        "Перенести кейс на этап «Информационное письмо»",
        "CRM: результат к задаче «связаться – первое уведомление о просрочке», закрыть задачу",
        "Создать «Контроль оплаты» на следующий день (если названа дата)",
        "В реестре зафиксировать риск/статус/причину/этап",
    ]
    return {
        "messages": [AIMessage(content=client_text)],
        "stage": "Информационное письмо",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

# D — ЛПР: проговариваем сумму и спрашиваем дату/причину
def lpr_prompt_node(state: AgentState) -> dict[str, Any]:
    meta = state.get("meta", {}) or {}
    msg = TEMPLATES["lpr_prompt"].format(
        act_date=meta.get("act_date", "<ДАТА АКТА>"),
        act_amount=meta.get("act_amount", "<СУММА АКТА>"),
    )
    return {"messages": [AIMessage(content=msg)], "route": "unknown"}

# n5/n6 — обработка названной даты
def named_date_within_week_node(state: AgentState) -> dict[str, Any]:
    date = (state.get("scratch", {}) or {}).get("payment_date", "00/00/0000")
    msg = TEMPLATES["named_date_within_week"].format(target_date=date)
    ops = ["Этап «Названа дата»", f"Ожидаем оплату {date}"]
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Названа дата",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

def named_date_over_week_node(state: AgentState) -> dict[str, Any]:
    date = (state.get("scratch", {}) or {}).get("payment_date", "00/00/0000")
    msg = TEMPLATES["named_date_over_week"].format(target_date=date)
    ops = [
        "Согласовать перенос даты (>7 дней) и зафиксировать SLA",
        f"Контроль через 3 дня; целевая дата {date}",
        "При нарушении — эскалация в юротдел",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Названа дата",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

# n7 — предложение оплатить в течение недели
def propose_pay_within_week_node(state: AgentState) -> dict[str, Any]:
    ops = ["Предложено: оплата в течение недели", "Далее — классификация согласия"]
    return {
        "messages": [AIMessage(content=TEMPLATES['pay_within_week'])],
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

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
def restructuring_node(state: AgentState) -> dict[str, Any]:
    first_date = (state.get("scratch", {}) or {}).get("first_date", "00.00.0000")
    last_date = (state.get("scratch", {}) or {}).get("last_date", "00.00.0000")
    msg = TEMPLATES["restruct_offer"].format(first_date=first_date, last_date=last_date)
    ops = ["Перевести на этап «Реструктуризация»", "Письменно зафиксировать график платежей"]
    scratch = _merge_ops(state, ops)
    scratch["offered_restruct"] = True
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Реструктуризация",
        "scratch": scratch,
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
def send_closings_node(state: AgentState) -> dict[str, Any]:
    msg = TEMPLATES["duplicate_closings"]
    ops = [
        "Оставить заявку на дублирование закрывающих",
        "Отправить письмо клиенту с документами",
        "Вернуться к запросу даты оплаты",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "scratch": _merge_ops(state, ops),
        "route": "named_date",
    }

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
def need_recon_node(state: AgentState) -> dict[str, Any]:
    msg = TEMPLATES["need_recon"]
    ops = [
        "Сформировать и отправить акт(ы) сверки",
        "Вернуться к запросу даты оплаты",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "scratch": _merge_ops(state, ops),
        "route": "named_date",
    }

# n21–n23 — счёт
def need_invoice_node(state: AgentState) -> dict[str, Any]:
    msg = TEMPLATES["need_invoice_explain"]
    ops = [
        "Пояснить не обязательность счёта",
        "Предложить закрывающие/реквизиты",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }

def create_duplicate_request_node(state: AgentState) -> dict[str, Any]:
    msg = "Заявку на дублирование закрывающих оформим и подтвердим письмом."
    ops = [
        "Оставить заявку по форме",
        "Поставить «контроль оплаты» через 3 дня от названной даты",
        "Перенести кейс на этап «Названа дата»",
    ]
    return {
        "messages": [AIMessage(content=msg)],
        "stage": "Названа дата",
        "scratch": _merge_ops(state, ops),
        "route": "unknown",
    }
