import zoneinfo
from datetime import datetime

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Вычисляет арифметическое выражение (Python-подобный синтаксис), например: "(12+30)*2"."""
    try:
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {})  # noqa: S307 (контролируем ввод)
        return str(result)
    except Exception as e:
        return f"error: {e}"


@tool
def now_in_tz(tz: str = "Europe/Moscow") -> str:
    """Возвращает текущее время в указанном IANA-таймзоне (например, Europe/Moscow)."""
    try:
        dt = datetime.now(zoneinfo.ZoneInfo(tz))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except Exception as e:
        return f"error: {e}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Отправляет письмо (заглушка); возвращает идентификатор очереди."""
    return f"email(to={to}, subject={subject}) queued"


@tool
def generate_invoice(account_id: str) -> str:
    """Генерирует счёт (заглушка); возвращает идентификатор документа."""
    return f"invoice:{account_id}:001"


@tool
def generate_reconciliation(account_id: str, period: str = "all") -> str:
    """Генерирует акт сверки (заглушка) за указанный период; возвращает идентификатор документа."""
    return f"recon:{account_id}:{period}"


@tool
def schedule_followup(when: str, note: str = "") -> str:
    """Создаёт задачу «перезвонить/контроль оплаты» на указанное время (заглушка)."""
    return f"followup scheduled at {when}: {note}"


@tool
def crm_set_result(task: str, result: str) -> str:
    """Устанавливает результат по задаче в CRM (заглушка)."""
    return f"crm.result[{task}]={result}"


@tool
def crm_close_task(task: str) -> str:
    """Закрывает задачу в CRM (заглушка)."""
    return f"crm.close[{task}]"


@tool
def crm_create_task(title: str, due: str, meta: str = "") -> str:
    """Создаёт задачу в CRM (заглушка) с названием, сроком и дополнительными данными."""
    return f"crm.create[{title}]@{due} {meta}"


@tool
def registry_update(fields: dict[str, str]) -> str:
    """Обновляет поля в реестре (заглушка). Принимает словарь ключ=значение."""
    return "registry:" + ",".join(f"{k}={v}" for k, v in fields.items())


@tool
def validate_payment_order(pp_text: str) -> str:
    """Проверяет текст платёжного поручения на корректность (заглушка)."""
    return "pp:validated"


@tool
def balance_search_payment(pp_id: str) -> str:
    """Создаёт запрос в блок «Баланс» на розыск и зачисление платежа по номеру ПП (заглушка)."""
    return f"balance:search[{pp_id}]"


@tool
def create_backoffice_request(form: str, payload: str) -> str:
    """Оставляет заявку в бэкофис по указанной форме с полезной нагрузкой (заглушка)."""
    return f"backoffice:{form}:{payload}"


TOOLS = [
    calculator,
    now_in_tz,
    send_email,
    generate_invoice,
    generate_reconciliation,
    schedule_followup,
    crm_set_result,
    crm_close_task,
    crm_create_task,
    registry_update,
    validate_payment_order,
    balance_search_payment,
    create_backoffice_request,
]
