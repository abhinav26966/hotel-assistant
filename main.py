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
from app.tools.tools import make_get_room_types_tool, make_get_available_rooms_tool, make_single_room_booking_tool, make_get_upcoming_bookings_tool
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
        db.commit()  # Commit changes if no exception occurred
    except:
        db.rollback()  # Roll back changes on exception
        raise
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
    try:
        # 1. Store user message in Postgres
        user_message = crud.create_message(db, message)

        # 2. Embed and store user message in Chroma
        vectorstore = get_vectorstore(str(message.conversation_id))
        vectorstore.add_texts([message.message], metadatas=[{"sender": message.sender, "message_id": str(user_message.id), "timestamp": str(user_message.created_at)}])

        # 3. Get conversation history from database (ordered by timestamp)
        conversation_messages = db.query(Message).filter(
            Message.conversation_id == message.conversation_id
        ).order_by(Message.created_at.asc()).limit(20).all()  # Get last 20 messages
        
        # 4. Build message history
        lc_messages = []
        current_year = datetime.now().year
        
        # 5. Add improved system prompt
        system_prompt = (
            "You are a hotel assistant specializing in room bookings. Follow these STRICT rules:\n\n"
            "BOOKING PROCESS:\n"
            "1. When user requests booking, collect: guests count, check-in date, check-out date, room preference\n"
            "2. Use getRooms tool with room_type parameter to filter results (e.g., getRooms(check_in='2025-06-08', check_out='2025-06-10', room_type='Deluxe'))\n"
            "3. When user wants specific room type, ALWAYS use room_type parameter in getRooms\n"
            "4. After showing rooms, ask which specific room number they want to book\n"
            "5. IMPORTANT: Use room_number NOT room_id when booking. For example: single_room_booking(email='user@example.com', room_type='Deluxe', check_in='2025-06-10', check_out='2025-06-12', room_number=201)\n"
            "6. Collect email address before booking if not provided\n"
            "7. NEVER ask for confirmation multiple times - confirm once then book\n\n"
            "RESPONSE RULES:\n"
            "- Remember all previously provided information in the conversation\n"
            "- ALWAYS filter room results using room_type parameter when user specifies preference\n"
            "- Show rooms in decorated format with clear pricing\n"
            "- Format dates as YYYY-MM-DD for tools\n"
            "- Assume current year {current_year} if year not specified\n"
            "- Display total cost and nights clearly\n\n"
            "COMPLETION RULES:\n"
            "- Complete bookings once the user confirms all the details after giving email and room selection\n"
            "- Ask the user to confirm the booking details once again just before and properly completing the booking\n"
            "- Display full booking confirmation with all details\n"
            "- Don't repeat information collection\n"
            "- If booking succeeds, congratulate and provide booking reference\n"
        ).format(current_year=current_year)
        
        lc_messages.append(SystemMessage(content=system_prompt))
        
        # 6. Add conversation history (excluding current message)
        for msg in conversation_messages[:-1]:  # Exclude the just-added current message
            if msg.sender == "User":
                lc_messages.append(HumanMessage(content=msg.message))
            elif msg.sender == "AI":
                lc_messages.append(AIMessage(content=msg.message))
        
        # 7. Add current user message
        lc_messages.append(HumanMessage(content=message.message))

        # 8. Tool setup
        tool_func = make_get_room_types_tool(db)
        tool_func2 = make_get_available_rooms_tool(db)
        tool_func3 = make_single_room_booking_tool(db)
        tool_func4 = make_get_upcoming_bookings_tool(db)
        tool_name_to_func = {
            "getRoomTypes": tool_func,
            "getRooms": tool_func2,
            "single_room_booking": tool_func3,
            "get_upcoming_bookings": tool_func4
        }

        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2, model="gpt-4o-mini")
        llm_with_tools = llm.bind_tools(list(tool_name_to_func.values()))

        # 9. Tool call loop with proper termination
        max_tool_loops = 3
        tool_loops = 0
        
        while tool_loops < max_tool_loops:
            try:
                response = llm_with_tools.invoke(lc_messages)
                
                if isinstance(response, AIMessage) and response.tool_calls:
                    lc_messages.append(response)
                    
                    # Process each tool call
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_func = tool_name_to_func.get(tool_name)
                        
                        if not tool_func:
                            ai_message_text = f"Sorry, I don't have access to the {tool_name} tool."
                            break
                        
                        try:
                            args = tool_call.get("args", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            
                            # Handle room_id vs room_number conversion
                            if tool_name == "single_room_booking" and "room_id" in args:
                                # If room_id is actually a number (room number), move it to room_number
                                try:
                                    room_number = int(args["room_id"])
                                    args["room_number"] = room_number
                                    del args["room_id"]
                                except (ValueError, TypeError):
                                    # Not a number, leave it as is
                                    pass
                                    
                        except Exception as e:
                            logger.error(f"Error parsing tool args: {e}")
                            ai_message_text = "I encountered an error processing your request. Please try again."
                            break
                        
                        try:
                            logger.info(f"Invoking tool {tool_name} with args: {args}")
                            
                            # Create new session for each tool call to avoid transaction issues
                            tool_db = SessionLocal()
                            try:
                                if tool_name == "getRoomTypes":
                                    tool_func_new = make_get_room_types_tool(tool_db)
                                    result = tool_func_new.invoke(args)
                                elif tool_name == "getRooms":
                                    tool_func_new = make_get_available_rooms_tool(tool_db)
                                    result = tool_func_new.invoke(args)
                                elif tool_name == "single_room_booking":
                                    tool_func_new = make_single_room_booking_tool(tool_db)
                                    result = tool_func_new.invoke(args)
                                elif tool_name == "get_upcoming_bookings":
                                    tool_func_new = make_get_upcoming_bookings_tool(tool_db)
                                    result = tool_func_new.invoke(args)
                                else:
                                    result = tool_func.invoke(args)
                                
                                lc_messages.append(
                                    ToolMessage(
                                        content=result,
                                        tool_call_id=tool_call["id"]
                                    )
                                )
                            except Exception as e:
                                error_msg = f"Tool execution error: {str(e)}"
                                logger.error(error_msg)
                                lc_messages.append(
                                    ToolMessage(
                                        content=json.dumps({"error": str(e)}),
                                        tool_call_id=tool_call["id"]
                                    )
                                )
                            finally:
                                tool_db.close()
                        except Exception as e:
                            error_msg = f"Tool execution error: {str(e)}"
                            logger.error(error_msg)
                            lc_messages.append(
                                ToolMessage(
                                    content=error_msg,
                                    tool_call_id=tool_call["id"]
                                )
                            )
                    
                    tool_loops += 1
                    
                    # If we've hit the tool limit, get final response
                    if tool_loops >= max_tool_loops:
                        response = llm_with_tools.invoke(lc_messages)
                        ai_message_text = response.content if isinstance(response, AIMessage) else str(response)
                        break
                        
                else:
                    # No tool calls, we have our final response
                    ai_message_text = response.content if isinstance(response, AIMessage) else str(response)
                    break
                    
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}")
                ai_message_text = "I encountered a technical issue. Could you please try again?"
                break
        
        # 10. Store AI response in a fresh db session to avoid transaction issues
        fresh_db = SessionLocal()
        try:
            ai_message_obj = MessageCreate(
                conversation_id=message.conversation_id,
                message=ai_message_text,
                sender="AI",
                toolsused=None
            )
            ai_message = crud.create_message(fresh_db, ai_message_obj)
            fresh_db.commit()

            # 11. Embed and store AI message in Chroma
            vectorstore.add_texts([ai_message_text], metadatas=[{"sender": "AI", "message_id": str(ai_message.id), "timestamp": str(ai_message.created_at)}])

            return MessageResponse(
                id=ai_message.id,
                conversation_id=ai_message.conversation_id,
                message=ai_message.message,
                sender=ai_message.sender,
                toolsused=ai_message.toolsused,
                created_at=ai_message.created_at
            )
        finally:
            fresh_db.close()
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        # Use a completely fresh session for error response
        error_db = SessionLocal()
        try:
            error_message = "I'm sorry, but I encountered a technical issue. Please try again."
            error_msg_obj = MessageCreate(
                conversation_id=message.conversation_id,
                message=error_message,
                sender="AI",
                toolsused=None
            )
            error_ai_message = crud.create_message(error_db, error_msg_obj)
            error_db.commit()
            
            return MessageResponse(
                id=error_ai_message.id,
                conversation_id=error_ai_message.conversation_id,
                message=error_ai_message.message,
                sender=error_ai_message.sender,
                toolsused=error_ai_message.toolsused,
                created_at=error_ai_message.created_at
            )
        except Exception as inner_e:
            logger.error(f"Failed to create error message: {str(inner_e)}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")
        finally:
            error_db.close()