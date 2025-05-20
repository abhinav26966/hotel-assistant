from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.models import Conversation, Message, User
from app.schemas.schemas import MessageCreate, UserCreate, UserLogin
from uuid import uuid4
import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(db: Session, user: UserCreate) -> User:
    hashed = hash_password(user.password)
    db_user = User(id=uuid4(), email=user.email, hashpass=hashed)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, user: UserLogin) -> Optional[User]:
    hashed = hash_password(user.password)
    db_user = db.query(User).filter(User.email == user.email, User.hashpass == hashed).first()
    return db_user

def create_conversation(db: Session, user_id=None) -> Conversation:
    if user_id:
        conv = Conversation(id=uuid4(), user_id=user_id)
    else:
        conv = Conversation(id=uuid4())
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

def create_message(db: Session, message: MessageCreate) -> Message:
    mes = Message(**message.dict())
    db.add(mes)
    db.commit()
    db.refresh(mes)
    return mes

def get_messages(db: Session, conversation_id: str) -> List[Message]:
    return (db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all())