# src/agent/otel.py
from __future__ import annotations

import os
import socket

_initialized = False

def _parse_headers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    pairs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            pairs.append((k.strip(), v.strip()))
    return dict(pairs)

def _is_endpoint_up(url: str, timeout: float = 0.3) -> bool:
    try:
        # простой TCP-чек
        from urllib.parse import urlparse
        u = urlparse(url)
        host = u.hostname or "localhost"
        port = u.port or (443 if u.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def setup_otel() -> None:
    """Инициализирует OpenTelemetry. Безопасно вызывается многократно."""
    global _initialized
    if _initialized:
        return

    # Полное отключение через env
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true" or os.getenv("OTEL_DISABLED", "").lower() == "true":
        _initialized = True
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        # авто-инструментация httpx/логирования (best-effort)
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument()
        except Exception:
            pass
        try:
            from opentelemetry.instrumentation.logging import LoggingInstrumentation
            LoggingInstrumentation().instrument(set_logging_format=True)
        except Exception:
            pass

        service = os.getenv("OTEL_SERVICE_NAME", "langgraph-agent")
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS"))
        exporter_mode = os.getenv("OTEL_EXPORTER", "auto").lower()  # auto|otlp|console|none

        provider = TracerProvider(resource=Resource.create({"service.name": service}))

        if exporter_mode == "none":
            # без экспортёров, но SDK включён — спаны «в никуда»
            trace.set_tracer_provider(provider)
            _initialized = True
            return

        if exporter_mode == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter_mode == "otlp" or _is_endpoint_up(endpoint):
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers)))
        else:
            # endpoint недоступен — переключаемся на консоль
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _initialized = True
    except Exception:
        # если пакеты OTEL не установлены — не мешаем выполнению
        _initialized = True
