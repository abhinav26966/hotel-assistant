from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.schemas.schemas import MessageCreate, MessageResponse, UserCreate, UserLogin, UserResponse, ConversationCreate, ConversationResponse
from app.crud import crud
from app.vectorStore.vectorstore import get_vectorstore, embeddings
from app.models.models import Message, Conversation, User
from uuid import UUID
from datetime import datetime
from langchain_openai import ChatOpenAI
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user = crud.create_user(db, user)
    return UserResponse(id=db_user.id, email=db_user.email)

@app.post("/login", response_model=UserResponse)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = crud.authenticate_user(db, user)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return UserResponse(id=db_user.id, email=db_user.email)

@app.post("/conversations", response_model=ConversationResponse)
def create_conversation(conv: ConversationCreate, db: Session = Depends(get_db)):
    conversation = crud.create_conversation(db, user_id=conv.user_id)
    return ConversationResponse(id=conversation.id, user_id=conversation.user_id)

@app.get("/")
def read_root(db: Session = Depends(get_db)):
    # example query
    users = db.execute("SELECT * FROM hotelassistant.users").fetchall()
    return users

@app.post("/chat", response_model=MessageResponse)
def chat(
    message: MessageCreate,
    db: Session = Depends(get_db)
):
    # 1. Store user message in Postgres
    user_message = crud.create_message(db, message)

    # 2. Embed and store user message in Chroma
    vectorstore = get_vectorstore(str(message.conversation_id))
    user_emb = embeddings.embed_documents([message.message])[0]
    vectorstore.add_texts([message.message], metadatas=[{"sender": message.sender, "message_id": str(user_message.id)}])

    # 3. Retrieve conversation history for context
    messages = crud.get_messages(db, str(message.conversation_id))
    chat_history = [
        {"role": m.sender.lower(), "content": m.message}
        for m in messages
    ]

    # 4. Call LLM (OpenAI via LangChain)
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2, verbose=True)
    prompt = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
    ]) + f"\nAI:"
    ai_response = llm.invoke(prompt)
    ai_message_text = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)

    # 5. Store AI message in Postgres
    ai_message_obj = MessageCreate(
        conversation_id=message.conversation_id,
        message=ai_message_text,
        sender="AI",
        toolsused=None
    )
    ai_message = crud.create_message(db, ai_message_obj)

    # 6. Embed and store AI message in Chroma
    ai_emb = embeddings.embed_documents([ai_message_text])[0]
    vectorstore.add_texts([ai_message_text], metadatas=[{"sender": "AI", "message_id": str(ai_message.id)}])

    # 7. Return AI message
    return MessageResponse(
        id=ai_message.id,
        conversation_id=ai_message.conversation_id,
        message=ai_message.message,
        sender=ai_message.sender,
        toolsused=ai_message.toolsused,
        created_at=ai_message.created_at
    )