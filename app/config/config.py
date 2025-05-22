from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    SMTP_HOST: str
    SMTP_PORT: int
    SMTP_USER: str
    SMTP_PASSWORD: str
    EMAIL_FROM: str
    POSTGRES_URL: str
    OPENAI_API_KEY: str
    DEEPGRAM_API_KEY: str
    class Config:
        env_file = ".env"

settings = Settings()