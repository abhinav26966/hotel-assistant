from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from uuid import UUID
from datetime import date
from app.models.models import SenderEnum, BookingStatus, RoomTypeEnum

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: UUID
    email: EmailStr

class MessageCreate(BaseModel):
    conversation_id: UUID
    message: str
    sender: SenderEnum  # 'User', 'AI', or 'Tool'
    toolsused: Optional[List[str]] = None

class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    message: str
    sender: SenderEnum
    toolsused: Optional[List[str]] = None
    created_at: date

class ConversationCreate(BaseModel):
    user_id: UUID

class ConversationResponse(BaseModel):
    id: UUID
    user_id: UUID

class BookingCreate(BaseModel):
    user_id: UUID
    rooms: List[UUID]
    check_in: date
    check_out: date
    status: Optional[BookingStatus] = BookingStatus.Booked

class RoomTypeResponse(BaseModel):
    id: UUID
    type: RoomTypeEnum
    description: Optional[str]
    capacity: Optional[int]
    cost: Optional[float]
