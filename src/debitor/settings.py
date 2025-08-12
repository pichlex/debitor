"""Application settings loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Base application settings.

    Using pydantic ensures type validation and easy expansion when new
    configuration options are introduced during project growth.
    """

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
