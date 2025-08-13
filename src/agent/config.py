from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic v2: конфиг через model_config
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",           # не падать, если в окружении есть лишние переменные
    )

    # LLM
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    # (опционально) списки для шардинга
    openai_api_keys: str | None = Field(default=None, alias="OPENAI_API_KEYS")
    model_names: str | None = Field(default=None, alias="MODEL_NAMES")

    # Checkpointer
    checkpoint_dsn: str = Field(default="sqlite:///./data/graph.db", alias="CHECKPOINT_DSN")
    checkpoint_dsn_shards: str | None = Field(default=None, alias="CHECKPOINT_DSN_SHARDS")

    # Sharding
    shard_count: int = Field(default=1, alias="SHARD_COUNT")

    # OpenTelemetry
    otel_service_name: str = Field(default="langgraph-agent", alias="OTEL_SERVICE_NAME")
    otel_endpoint: str = Field(default="http://localhost:4318", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_headers: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_HEADERS")
    otel_resource_attributes: str | None = Field(default=None, alias="OTEL_RESOURCE_ATTRIBUTES")

settings = Settings()
