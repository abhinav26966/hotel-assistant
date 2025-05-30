"""Initial migration

Revision ID: f40c781575d6
Revises: 
Create Date: 2025-05-20 07:58:23.815767

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f40c781575d6'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('room_type',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('type', sa.Enum('Standard', 'Deluxe', 'Suite', name='roomtypeenum'), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('capacity', sa.Integer(), nullable=True),
    sa.Column('cost', sa.Numeric(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    schema='hotelassistant'
    )
    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('email', sa.String(), nullable=False),
    sa.Column('hashpass', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    schema='hotelassistant'
    )
    op.create_table('bookings',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('check_in', sa.Date(), nullable=False),
    sa.Column('check_out', sa.Date(), nullable=False),
    sa.Column('status', sa.Enum('Booked', 'Cancelled', name='bookingstatus'), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['hotelassistant.users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    schema='hotelassistant'
    )
    op.create_table('conversations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['hotelassistant.users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    schema='hotelassistant'
    )
    op.create_table('rooms',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('room_no', sa.Integer(), nullable=False),
    sa.Column('room_type_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['room_type_id'], ['hotelassistant.room_type.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('room_no'),
    schema='hotelassistant'
    )
    op.create_table('booking_rooms',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('booking_id', sa.UUID(), nullable=False),
    sa.Column('room_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['booking_id'], ['hotelassistant.bookings.id'], ),
    sa.ForeignKeyConstraint(['room_id'], ['hotelassistant.rooms.id'], ),
    sa.PrimaryKeyConstraint('id'),
    schema='hotelassistant'
    )
    op.create_table('messages',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('conversation_id', sa.UUID(), nullable=False),
    sa.Column('message', sa.String(), nullable=True),
    sa.Column('sender', sa.Enum('User', 'AI', 'Tool', name='senderenum'), nullable=True),
    sa.Column('toolsused', postgresql.ARRAY(sa.String()), nullable=True),
    sa.ForeignKeyConstraint(['conversation_id'], ['hotelassistant.conversations.id'], ),
    sa.PrimaryKeyConstraint('id'),
    schema='hotelassistant'
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('messages', schema='hotelassistant')
    op.drop_table('booking_rooms', schema='hotelassistant')
    op.drop_table('rooms', schema='hotelassistant')
    op.drop_table('conversations', schema='hotelassistant')
    op.drop_table('bookings', schema='hotelassistant')
    op.drop_table('users', schema='hotelassistant')
    op.drop_table('room_type', schema='hotelassistant')
    # ### end Alembic commands ###
