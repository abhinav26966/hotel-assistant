from sqlalchemy import Column, String, Integer, Enum, ForeignKey, Date, Numeric, ARRAY
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
import enum
import uuid
from sqlalchemy.sql import func

metadata = MetaData(schema="hotelassistant")
Base = declarative_base(metadata=metadata)

class SenderEnum(str, enum.Enum):
    User = "User"
    AI = "AI"
    Tool = "Tool"

class BookingStatus(str, enum.Enum):
    Booked = "Booked"
    Cancelled = "Cancelled"

class RoomTypeEnum(str, enum.Enum):
    Standard = "Standard"
    Deluxe = "Deluxe"
    Suite = "Suite"

class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    hashpass = Column(String, nullable=False)

class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('hotelassistant.users.id'), nullable=False)

class Message(Base):
    __tablename__ = 'messages'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('hotelassistant.conversations.id'), nullable=False)
    message = Column(String)
    sender = Column(Enum(SenderEnum))
    toolsused = Column(ARRAY(String))
    created_at = Column(Date, nullable=False, default=func.now())

class Booking(Base):
    __tablename__ = 'bookings'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('hotelassistant.users.id'), nullable=False)
    rooms = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    check_in = Column(Date, nullable=False)
    check_out = Column(Date, nullable=False)
    status = Column(Enum(BookingStatus), default=BookingStatus.Booked)

class Room(Base):
    __tablename__ = 'rooms'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    room_no = Column(Integer, unique=True, nullable=False)
    room_type_id = Column(UUID(as_uuid=True), ForeignKey('hotelassistant.room_type.id'), nullable=False)

class RoomType(Base):
    __tablename__ = 'room_type'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(Enum(RoomTypeEnum), nullable=False)
    description = Column(String)
    capacity = Column(Integer)
    cost = Column(Numeric)