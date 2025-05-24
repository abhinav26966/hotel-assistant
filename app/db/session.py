from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("POSTGRES_URL")

if not DATABASE_URL:
    raise ValueError("POSTGRES_URL environment variable is not set")
engine = create_engine(DATABASE_URL)

# Set the search_path to hotelassistant schema
with engine.connect() as connection:
    connection.execute(text("SET search_path TO hotelassistant"))
    connection.commit()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)