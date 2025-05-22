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
from app.tools.tools import make_get_room_types_tool, make_get_available_rooms_tool, make_single_room_booking_tool, make_get_upcoming_bookings_tool, make_get_ongoing_bookings_tool, make_get_past_bookings_tool, make_update_booking_tool, make_cancel_booking_tool
from fastapi import BackgroundTasks
from app.utils.email_utils import send_booking_confirmation
from app.config.config import settings
from fastapi import File, UploadFile
from io import BytesIO
from deepgram import Deepgram
import aiofiles
import logging
import json
logger = logging.getLogger(__name__)
import dotenv
from fastapi.responses import JSONResponse

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()

deepgram_api_key=os.getenv("DEEPGRAM_API_KEY")

@app.post("/voice-chat")
async def voice_chat(
    conversation_id: UUID,
    user_id: UUID,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    try:
        # Load API key
        if not deepgram_api_key:
            raise HTTPException(status_code=500, detail="Deepgram API key not configured")
            
        deepgram = Deepgram(deepgram_api_key)

        # Read audio file
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio data received")
            
        mimetype = file.content_type or "audio/webm"

        # Send audio to Deepgram for transcription
        try:
            response = await deepgram.transcription.prerecorded(
                {
                    "buffer": audio_data,
                    "mimetype": mimetype
                },
                {
                    "punctuate": True,
                    "language": "en"
                }
            )
        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}")
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")

        # Extract transcript safely
        alternatives = response.get("results", {}).get("channels", [{}])[0].get("alternatives", [])
        if not alternatives or not alternatives[0].get("transcript"):
            raise HTTPException(status_code=400, detail="No speech detected")

        transcript = alternatives[0]["transcript"]
        # logger.info(f"Transcribed text: {transcript}")

        # Create user message
        user_msg = MessageCreate(
            conversation_id=conversation_id,
            message=transcript,
            sender="User",
            user_id=user_id,
        )
        
        # Create and save user message
        try:
            user_message = crud.create_message(db, user_msg)
            db.commit()
        except Exception as e:
            logger.error(f"Error saving user message: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to save user message")

        # Get AI response
        try:
            ai_response = await chat(user_msg, background_tasks, db)
            # logger.info(f"AI response: {ai_response.message}")
            return JSONResponse(content={
                "user_message": transcript,
                "ai_message": ai_response.message
            })
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            raise HTTPException(status_code=500, detail="Failed to get AI response")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    users = db.execute("SELECT * FROM hotelassistant.users").fetchall()
    return users

@app.post("/chat", response_model=MessageResponse)
async def chat(message: MessageCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    try:
        user_message = crud.create_message(db, message)
        vectorstore = get_vectorstore(str(message.conversation_id))
        vectorstore.add_texts([message.message], metadatas=[{"sender": message.sender, "message_id": str(user_message.id), "timestamp": str(user_message.created_at)}])

        conversation_messages = db.query(Message).filter(
            Message.conversation_id == message.conversation_id
        ).order_by(Message.created_at.asc()).limit(20).all()

        lc_messages = []
        current_year = datetime.now().year
        current_date = datetime.now().date()

        system_prompt = (
            "You are Vera, a hotel assistant specializing in room bookings. Follow these STRICT rules:\n\n"
            "BOOKING PROCESS:\n"
            "1. When user requests booking, collect: guests count, check-in date, check-out date, room preference\n"
            "2. Use getRooms tool with room_type parameter to filter results (e.g., getRooms(check_in='2025-06-08', check_out='2025-06-10', room_type='Deluxe'))\n"
            "3. When user wants specific room type, ALWAYS use room_type parameter in getRooms\n"
            "4. Do NOT allow more than the allowed guest limit per room type:\n"
                "- Deluxe: max 3 guests\n"
                "- Suite: max 4 guests\n"
                "- Standard: max 2 guests\n"
            "5. Inform the user and suggest booking multiple rooms one-by-one if needed.\n"
            "6. Book rooms using the lowest available `room_number`, not `room_id`.\n"
            "7. Collect email address before booking if not provided\n"
            "8. NEVER ask for confirmation multiple times - confirm once then book\n\n"
            "9. If the user asks to book a room which is before current date {current_date}, tell them that you can not book a room which is before current date {current_date}.\n"
            "MULTIPLE BOOKINGS:\n"
                "- You can only book ONE room at a time.\n"
                "- If the user wants multiple rooms, tell them to repeat the booking process one-by-one.\n"
                "- After each successful booking, remember the details and avoid repeating unless asked.\n\n"
            "CONTEXT & MEMORY:\n"
                "- Always remember all previously provided information in this conversation.\n"
                "- After each tool response, trust your own summaries and never repeat the same tool call unless the user asks again.\n"
                "- Look for phrases like 'I want to book another room' to start new bookings.\n\n"
            "RESPONSE RULES:\n"
            "- Remember all previously provided information in the conversation\n"
            "- ALWAYS filter room results using room_type parameter when user specifies preference\n"
            "- Show rooms in decorated format with clear pricing\n"
            "- Always send all the responses in a deccorated format. You are a hotel assistant and you can not send responses in a plain text format. If needed then use emojis and other formatting.\n"
            "- Format dates as YYYY-MM-DD for tools\n"
            "- Congratulate the user after a successful booking.\n"
            "- Assume current year {current_year} if year not specified\n"
            "- Provide booking reference and full summary after confirmation.\n"
            "- NEVER enter a loop. If unsure, ask the user for clarification.\n"
            "- Display total cost and nights clearly\n\n"
            "- DO NOT ask for confirmation again if booking is already marked as 'Booked'.\n"
            "- If a booking has already been completed, just respond with a friendly message confirming it again.\n"
            "- Always check context before repeating actions.\n"
            "- If the user tells you that they want to book multiple rooms at a time then you have to tell them that you can only book one room at a time. However, you can tell them that they can book more rooms one by one.\n"
            "- ONLY GIVE FOCUSED RESPONSES. YOU ARE A HOTEL ASSISTANT AND YOU CAN NOT ANSWER ANYTHING ELSE OTHER THAN GENERAL QUESTIONS.\n"
            "COMPLETION RULES:\n"
            "- Complete bookings once the user confirms all the details after giving email and room selection\n"
            "- Ask the user to confirm the booking details once again just before and properly completing the booking\n"
            "- Display full booking confirmation with all details\n"
            "- Don't repeat information collection\n"
            "- If booking succeeds, congratulate and provide booking reference\n"
        ).format(current_year=current_year, current_date=current_date)

        lc_messages.append(SystemMessage(content=system_prompt))

        for msg in conversation_messages[:-1]:
            if msg.sender == "User":
                lc_messages.append(HumanMessage(content=msg.message))
            elif msg.sender == "AI":
                lc_messages.append(AIMessage(content=msg.message))

        lc_messages.append(HumanMessage(content=message.message))
        # print(HumanMessage(content=message.message))

        tool_funcs = {
            "getRoomTypes": make_get_room_types_tool,
            "getRooms": make_get_available_rooms_tool,
            "single_room_booking": make_single_room_booking_tool,
            "get_upcoming_bookings": make_get_upcoming_bookings_tool,
            "get_ongoing_bookings": make_get_ongoing_bookings_tool,
            "get_past_bookings": make_get_past_bookings_tool,
            "update_booking": make_update_booking_tool,
            "cancel_booking": make_cancel_booking_tool
        }

        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2, model="gpt-4o-mini")
        llm_with_tools = llm.bind_tools([f(db) for f in tool_funcs.values()])

        def extract_summary_from_tool_result(result, tool_name):
            try:
                data = json.loads(result)
                if tool_name == "single_room_booking" and data.get("status") == "Booked":
                    return f"Booking confirmed for {data['room_type']} from {data['check_in']} to {data['check_out']} under the name {data['guest_name']}. Room Number: {data['room_number']}. Booking ID: {data['booking_id']}."
                # Handle other tools if needed
            except Exception:
                pass
            return f"Tool {tool_name} responded: {result}"
        

        max_tool_loops = 8
        tool_loops = 0

        while tool_loops < max_tool_loops:
            try:
                response = llm_with_tools.invoke(lc_messages)

                if isinstance(response, AIMessage) and response.tool_calls:
                    lc_messages.append(response)
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_func_constructor = tool_funcs.get(tool_name)
                        if not tool_func_constructor:
                            lc_messages.append(ToolMessage(content=json.dumps({"error": f"Tool {tool_name} not recognized."}), tool_call_id=tool_call["id"]))
                            continue

                        try:
                            args = tool_call.get("args", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            if tool_name == "single_room_booking" and "room_id" in args:
                                try:
                                    args["room_number"] = int(args.pop("room_id"))
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.error(f"Arg parsing error: {e}")
                            lc_messages.append(ToolMessage(content=json.dumps({"error": "Invalid arguments"}), tool_call_id=tool_call["id"]))
                            continue

                        tool_db = SessionLocal()
                        try:
                            tool_instance = tool_func_constructor(tool_db)
                            result = tool_instance.invoke(args)
                            lc_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
                            summary_text = extract_summary_from_tool_result(result, tool_name)
                            if tool_name == "single_room_booking":
                                # print("entering the email sending part")
                                try:
                                    booking_data = json.loads(result)
                                    # print(booking_data)

                                    confirmation = booking_data.get("booking_confirmation", {})
                                    guest_email = confirmation.get("guest_email")
                                    guest_name = guest_email.split("@")[0].capitalize() if guest_email else "Guest"

                                    email_body = (
                                        f"Dear {guest_name},\n\n"
                                        f"Your booking has been confirmed.\n"
                                        f"Room Type: {confirmation.get('room_type')}\n"
                                        f"Room Number: {confirmation.get('room_number')}\n"
                                        f"Check-in: {confirmation.get('check_in')}\n"
                                        f"Check-out: {confirmation.get('check_out')}\n"
                                        f"Booking ID: {confirmation.get('booking_id')}\n\n"
                                        f"Thank you for choosing our service!\n"
                                    )

                                    # print("Email body: ", email_body)
                                    await send_booking_confirmation(
                                        guest_email,
                                        "Your Hotel Booking Confirmation",
                                        email_body
                                    )
                                except Exception as e:
                                    logger.error(f"Email sending error: {e}")

                            if tool_name == "single_room_booking":
                                summary_text += "\n\n[NOTE FOR AI MEMORY]: This booking is completed. Do not attempt to book again unless the user clearly asks for a new booking."
                            lc_messages.append(AIMessage(content=summary_text))
                        except Exception as e:
                            logger.error(f"Tool error: {e}")
                            lc_messages.append(ToolMessage(content=json.dumps({"error": str(e)}), tool_call_id=tool_call["id"]))
                        finally:
                            tool_db.close()

                    tool_loops += 1
                    if tool_loops >= max_tool_loops:
                        response = llm_with_tools.invoke(lc_messages)
                        ai_message_text = response.content if isinstance(response, AIMessage) else str(response)
                        break
                else:
                    ai_message_text = response.content if isinstance(response, AIMessage) else str(response)
                    break
            except Exception as e:
                logger.error(f"Conversation loop error: {e}")
                ai_message_text = "I encountered a technical issue. Please try again."
                break

        fresh_db = SessionLocal()
        try:
            # logger.info(f"AI raw response: {response.content}")
            if not ai_message_text.strip():
                logger.warning("AI message is empty, providing default response")
                ai_message_text = "I'm here to help you with hotel bookings. How can I assist you today?"

            ai_message_obj = MessageCreate(
                conversation_id=message.conversation_id,
                message=ai_message_text,
                sender="AI",
                toolsused=None
            )
            ai_message = crud.create_message(fresh_db, ai_message_obj)
            fresh_db.commit()
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
        logger.error(f"Chat endpoint error: {str(e)}")
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

@app.options("/voice-chat")
async def options_voice_chat():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )

@app.get("/user/{user_id}/conversations")
def get_user_conversations(user_id: UUID, db: Session = Depends(get_db)):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).all()
    
    # Get the latest message for each conversation to show a preview
    result = []
    for conv in conversations:
        latest_message = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).order_by(Message.created_at.desc()).first()
        
        result.append({
            "id": str(conv.id),
            "user_id": str(conv.user_id),
            "created_at": conv.created_at if hasattr(conv, 'created_at') else None,
            "latest_message": latest_message.message if latest_message else None
        })
    
    return result

@app.get("/messages")
def get_messages(conversation_id: UUID, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()
    
    return [
        {
            "id": str(msg.id),
            "conversation_id": str(msg.conversation_id),
            "message": msg.message,
            "sender": msg.sender,
            "created_at": msg.created_at
        }
        for msg in messages
    ]