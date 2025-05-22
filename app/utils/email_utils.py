# email_utils.py
import os
import aiosmtplib
from email.message import EmailMessage
import logging

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASSWORD")
FROM_EMAIL = os.getenv("EMAIL_FROM")

async def send_booking_confirmation(to_email: str, subject: str, body: str):
    try:
        message = EmailMessage()
        message["From"] = FROM_EMAIL
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(body)

        logger.info(f"Sending email to {to_email} from {FROM_EMAIL} via {SMTP_HOST}:{SMTP_PORT}")

        await aiosmtplib.send(
            message,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            username=SMTP_USER,
            password=SMTP_PASS,
            start_tls=True
        )

        logger.info(f"Email sent successfully to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")