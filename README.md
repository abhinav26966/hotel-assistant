# Hotel Assistant

A voice-enabled hotel booking assistant application. The assistant helps users book hotel rooms, check availability, and manage bookings through both text and voice interactions.

## Features

- User authentication (signup/login)
- Text chat with AI assistant for hotel bookings
- Voice input for natural conversation
- Room availability checking
- Room booking
- Booking management (view, update, cancel)
- Email confirmations for bookings

## Tech Stack

### Backend
- FastAPI
- SQLAlchemy ORM
- PostgreSQL database
- LangChain + OpenAI for AI conversation
- Deepgram for voice transcription
- Alembic for database migrations

### Frontend
- React
- MediaRecorder API for voice recording

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+
- PostgreSQL database
- OpenAI API key
- Deepgram API key
- SMTP server for email notifications

### Environment Variables
Create a `.env` file in the root directory with:

```
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
SMTP_HOST=your_smtp_host
SMTP_PORT=your_smtp_port
SMTP_USER=your_smtp_username
SMTP_PASSWORD=your_smtp_password
EMAIL_FROM=your_sender_email
```

### Backend Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run database migrations:
```
alembic upgrade head
```

3. Start the backend server:
```
uvicorn main:app --reload
```

### Frontend Setup

1. Navigate to the frontend directory:
```
cd frontend
```

2. Install dependencies:
```
npm install
```

3. Start the frontend development server:
```
npm start
```

## Usage

1. Open your browser to `http://localhost:3000`
2. Sign up for a new account or login
3. Create a new conversation
4. Start chatting with the hotel assistant using text or voice
   - For voice input, click the microphone button, speak your request, then click the stop button
   - The assistant will process your voice, transcribe it, and respond

## API Endpoints

- `/signup` - Create a new user account
- `/login` - Authenticate user
- `/conversations` - Create a new conversation
- `/chat` - Send and receive text messages
- `/voice-chat` - Send voice recordings for processing

## Architecture

The application follows a client-server architecture:

1. The React frontend captures user input (text or voice)
2. For voice, the MediaRecorder API records audio and sends it to the backend
3. The backend uses Deepgram to transcribe voice to text
4. The text is processed by the AI assistant using LangChain and OpenAI
5. The assistant uses tools to query and manipulate the database
6. Responses are sent back to the frontend
7. Email confirmations are sent for completed bookings 