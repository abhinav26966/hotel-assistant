from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
import os

password = quote_plus("Abhinav@2604")
DATABASE_URL = f"postgresql+psycopg2://abhinav:{password}@157.245.219.63:5432/hotelmanagement"

engine = create_engine(DATABASE_URL)

# Set the search_path to hotelassistant schema
with engine.connect() as connection:
    connection.execute(text("SET search_path TO hotelassistant"))
    connection.commit()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)