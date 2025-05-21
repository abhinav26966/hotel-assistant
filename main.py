from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.schemas.schemas import MessageCreate, MessageResponse, UserCreate, UserLogin, UserResponse, ConversationCreate, ConversationResponse
from app.crud import crud
from app.vectorStore.vectorstore import get_vectorstore, embeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.models.models import Message, Conversation, User
from uuid import UUID
from datetime import datetime
from langchain_openai import ChatOpenAI
import os
from fastapi.middleware.cors import CORSMiddleware
from app.tools.tools import make_get_room_types_tool, make_get_available_rooms_tool
import logging
import json
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

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
    lc_messages = []
    retrieved_docs = vectorstore.similarity_search(
        message.message,  # query
        k=5  # tune this for your context window
    )
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"Context[{i}]: {doc.page_content} | Meta: {doc.metadata}")
    current_year = datetime.now().year

    # 4. Add system prompt
    system_prompt = (
        "You are a hotel assistant. "
        "You can use the following tools to get information about the hotel: "
        "getRoomTypes: Get all different types of rooms provided by the hotel. Returns a list of room types with their details (type, description, capacity, cost). "
        "getRooms: Get all available rooms between check in date and check out date. Returns a list of room types with their details (type, description, capacity, cost). If any of the dates doesn't mention the year, use the {current_year} as the year."
        "Otherwise, you can answer the user's question based on the information available in the conversation history."
        "Always send the final response in decorated manner and not in plain text or markdown format."
    ).format(current_year=current_year)
    lc_messages.insert(0, SystemMessage(content=system_prompt))
    lc_messages.append(HumanMessage(content=message.message))

    # 5. Add chat history using correct message types
    for doc in retrieved_docs:
        sender = doc.metadata["sender"]
        if sender == "User":
            lc_messages.append(HumanMessage(content=doc.page_content))
        elif sender == "AI":
            lc_messages.append(AIMessage(content=doc.page_content))
        elif sender == "Tool":
            lc_messages.append(ToolMessage(content=doc.page_content, tool_call_id="tool_call_id"))

    # 6. Add the current user message
    lc_messages.append(
        HumanMessage(content=message.message)
    )

    # 7. Tool binding with LangChain
    tool_func = make_get_room_types_tool(db)
    tool_func2 = make_get_available_rooms_tool(db)
    tool_name_to_func = {
        "getRoomTypes": tool_func,
        "getRooms": tool_func2
    }
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2, model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools(list(tool_name_to_func.values()))

    tool_args_spec = {
        "getRooms": {"check_in", "check_out"},
        "getRoomTypes": set()
    }

    # 8. Tool call loop
    max_tool_loops = 2
    tool_loops = 0
    while True:
        response = llm_with_tools.invoke(lc_messages)
        logger.info("LLM response: %s", response)

        if isinstance(response, AIMessage) and response.tool_calls:
            lc_messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_func = tool_name_to_func.get(tool_name)

                if not tool_func:
                    ai_message_text = f"Sorry, I don't know how to handle the tool: {tool_name}"
                    break

                try:
                    args = tool_call.get("args", {})
                    args = args if isinstance(args, dict) else json.loads(args)
                except Exception as e:
                    logger.error(f"Error parsing tool args: {e}")
                    args = {}

                # Check for missing required arguments
                required_args = tool_args_spec.get(tool_name, set())
                missing_args = [arg for arg in required_args if arg not in args or not args[arg]]

                if missing_args:
                    ai_message_text = f"To proceed, I need the following information: {', '.join(missing_args)}. Could you please provide them?"
                    break  # Exit loop to return this message

                try:
                    logger.info(f"Invoking tool {tool_name} with args: {args}")
                    result = tool_func.invoke(args)
                except Exception as e:
                    result = f"Tool invocation error: {str(e)}"
                    logger.error(result)

                lc_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )
                tool_loops += 1

            continue
        else:
            ai_message_text = response.content if isinstance(response, AIMessage) else str(response)
            break

    ai_message_obj = MessageCreate(
        conversation_id=message.conversation_id,
        message=ai_message_text,
        sender="AI",
        toolsused=None
    )
    ai_message = crud.create_message(db, ai_message_obj)

    # 10. Embed and store AI message in Chroma
    ai_emb = embeddings.embed_documents([ai_message_text])[0]
    vectorstore.add_texts([ai_message_text], metadatas=[{"sender": "AI", "message_id": str(ai_message.id)}])

    # 11. Return AI message
    return MessageResponse(
        id=ai_message.id,
        conversation_id=ai_message.conversation_id,
        message=ai_message.message,
        sender=ai_message.sender,
        toolsused=ai_message.toolsused,
        created_at=ai_message.created_at
    )