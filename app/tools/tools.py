from langchain_core.tools import tool
from sqlalchemy import func
from app.models.models import RoomType
import json
import logging
import json
from app.models.models import Room, Booking, BookingStatus
from datetime import date
logger = logging.getLogger(__name__)

def make_get_room_types_tool(db_session):
    @tool
    def getRoomTypes():
        """Get all different types of rooms provided by the hotel. Returns a list of room types with their details (type, description, capacity, cost)."""
        print("getRoomTypes tool called")
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

def make_get_available_rooms_tool(db_session):
    @tool
    def getRooms(check_in: date, check_out: date):
        """Get all available rooms between check in date and check out date. Returns a list of room types with their details (type, description, capacity, cost)."""
        print("getRooms tool called")
        logger.info("getRooms tool called")
        booked_rooms_subq = (
            db_session.query(func.unnest(Booking.rooms).label('room_id'))
            .where(
                Booking.status != BookingStatus.Cancelled,
                Booking.check_in < check_out,
                Booking.check_out > check_in,
            )
            .subquery()
        )
        available_rooms = (
            db_session.query(Room)
            .filter(Room.id.not_in(db_session.query(booked_rooms_subq.c.room_id)))
            .all()
        )
        return json.dumps([
            {
                "id": str(rt.id),
                "room_type_id": str(rt.room_type_id)
            }
            for rt in available_rooms
        ])
    return getRooms
    