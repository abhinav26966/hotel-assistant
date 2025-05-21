from langchain_core.tools import tool
from sqlalchemy import func
from app.models.models import RoomType
import json
import logging
from app.models.models import Room, Booking, BookingStatus, User, RoomTypeEnum
from datetime import date, datetime
from uuid import uuid4
logger = logging.getLogger(__name__)

def make_get_room_types_tool(db_session):
    @tool
    def getRoomTypes():
        """Get all different types of rooms provided by the hotel. Returns a list of room types with their details (type, description, capacity, cost)."""
        logger.info("getRoomTypes tool called")
        room_types = db_session.query(RoomType).all()
        return json.dumps([
            {
                "id": str(rt.id),
                "type": rt.type.value,
                "description": rt.description,
                "capacity": rt.capacity,
                "cost": float(rt.cost)
            }
            for rt in room_types
        ])
    return getRoomTypes

def parse_date(d):
    return datetime.strptime(d, "%Y-%m-%d").date() if isinstance(d, str) else d

def make_get_available_rooms_tool(db_session):
    @tool
    def getRooms(check_in: str, check_out: str, room_type: str = None):
        """Get available rooms between check-in and check-out dates. 
        Optional room_type parameter to filter by specific room type (Standard, Deluxe, Suite).
        Returns a list of available rooms with their details."""
        logger.info(f"getRooms tool called for dates: {check_in} to {check_out}, room_type: {room_type}")
        
        # Convert string dates to date objects
        try:
            check_in_date = parse_date(check_in)
            check_out_date = parse_date(check_out)
        except ValueError as e:
            return json.dumps({"error": f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"})

        # Validate dates
        if check_in_date >= check_out_date:
            return json.dumps({"error": "Check-in date must be before check-out date"})
        
        if check_in_date < date.today():
            return json.dumps({"error": "Check-in date cannot be in the past"})

        # Get booked rooms for the date range
        booked_rooms_subq = (
            db_session.query(func.unnest(Booking.rooms).label('room_id'))
            .where(
                Booking.status != BookingStatus.Cancelled,
                Booking.check_in < check_out_date,
                Booking.check_out > check_in_date,
            )
            .subquery()
        )

        # Base query for available rooms
        query = (
            db_session.query(Room, RoomType)
            .join(RoomType, Room.room_type_id == RoomType.id)
            .filter(~Room.id.in_(db_session.query(booked_rooms_subq.c.room_id)))
        )

        # Filter by room type if specified
        if room_type:
            try:
                room_type_enum = RoomTypeEnum(room_type)
                query = query.filter(RoomType.type == room_type_enum)
            except ValueError:
                return json.dumps({"error": f"Invalid room type '{room_type}'. Valid options are: {[e.value for e in RoomTypeEnum]}"})

        available_rooms = query.limit(10).all()  # Limit to prevent overwhelming responses

        if not available_rooms:
            room_type_msg = f" of type '{room_type}'" if room_type else ""
            return json.dumps({"error": f"No available rooms{room_type_msg} found for the specified dates"})

        nights = (check_out_date - check_in_date).days
        
        return json.dumps({
            "available_rooms": [
                {
                    "room_id": str(room.Room.id),
                    "room_no": room.Room.room_no,
                    "type": room.RoomType.type.value,
                    "description": room.RoomType.description,
                    "capacity": room.RoomType.capacity,
                    "cost_per_night": float(room.RoomType.cost),
                    "total_cost": float(room.RoomType.cost) * nights
                }
                for room in available_rooms
            ],
            "nights": nights,
            "check_in": check_in,
            "check_out": check_out
        })
    return getRooms


def make_single_room_booking_tool(db_session):
    @tool
    def single_room_booking(email: str, room_type: str, check_in: str, check_out: str, room_number: int = None):
        """Book a single room between check-in and check-out dates. 
        If room_number is provided, book that specific room number. Otherwise, book any available room of the specified type.
        Returns a confirmation message with booking details."""
        logger.info(f"single_room_booking tool called for {email}, {room_type}, {check_in} to {check_out}, room_number: {room_number}")

        # Convert string dates to date objects
        try:
            check_in_date = parse_date(check_in)
            check_out_date = parse_date(check_out)
        except ValueError as e:
            return json.dumps({"error": f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"})

        # Validate dates
        if check_in_date >= check_out_date:
            return json.dumps({"error": "Check-in date must be before check-out date"})
        
        if check_in_date < date.today():
            return json.dumps({"error": "Check-in date cannot be in the past"})

        # Find user
        find_user = db_session.query(User).filter(User.email == email).first()
        if not find_user:
            return json.dumps({"error": f"User with email {email} not found"})
        
        # Validate room type
        try:
            room_type_enum = RoomTypeEnum(room_type)
        except ValueError:
            return json.dumps({"error": f"Invalid room type '{room_type}'. Valid options are: {[e.value for e in RoomTypeEnum]}"})

        # Get booked rooms for the date range
        booked_rooms_subq = (
            db_session.query(func.unnest(Booking.rooms).label('room_id'))
            .where(
                Booking.status != BookingStatus.Cancelled,
                Booking.check_in < check_out_date,
                Booking.check_out > check_in_date,
            )
            .subquery()
        )

        # Find the specific room or any available room of the requested type
        if room_number:
            # Book specific room by room number (not UUID)
            available_room = (
                db_session.query(Room, RoomType)
                .join(RoomType, Room.room_type_id == RoomType.id)
                .filter(
                    Room.room_no == room_number,
                    ~Room.id.in_(db_session.query(booked_rooms_subq.c.room_id)),
                    RoomType.type == room_type_enum
                )
                .first()
            )
            
            if not available_room:
                return json.dumps({"error": f"Room {room_number} is not available for the specified dates"})
        else:
            # Book any available room of the requested type
            available_room = (
                db_session.query(Room, RoomType)
                .join(RoomType, Room.room_type_id == RoomType.id)
                .filter(
                    ~Room.id.in_(db_session.query(booked_rooms_subq.c.room_id)),
                    RoomType.type == room_type_enum
                )
                .first()
            )

        if not available_room:
            return json.dumps({"error": f"No available {room_type} rooms found for the specified dates"})

        # Calculate total cost
        nights = (check_out_date - check_in_date).days
        total_cost = float(available_room.RoomType.cost) * nights

        # Create booking
        booking = Booking(
            id=uuid4(),
            user_id=find_user.id,
            rooms=[available_room.Room.id],
            check_in=check_in_date,
            check_out=check_out_date,
            status=BookingStatus.Booked
        )

        try:
            db_session.add(booking)
            db_session.commit()

            return json.dumps({
                "success": True,
                "booking_confirmation": {
                    "booking_id": str(booking.id),
                    "guest_email": email,
                    "room_number": available_room.Room.room_no,
                    "room_type": room_type,
                    "check_in": check_in_date.isoformat(),
                    "check_out": check_out_date.isoformat(),
                    "nights": nights,
                    "cost_per_night": float(available_room.RoomType.cost),
                    "total_cost": total_cost,
                    "status": booking.status.value,
                    "booking_date": datetime.now().isoformat()
                }
            })
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error creating booking: {str(e)}")
            return json.dumps({"error": f"Failed to create booking: {str(e)}"})

    return single_room_booking