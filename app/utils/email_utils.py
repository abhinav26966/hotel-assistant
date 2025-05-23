# email_utils.py
import os
import aiosmtplib
from email.message import EmailMessage
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASSWORD")
FROM_EMAIL = os.getenv("EMAIL_FROM")

async def send_booking_confirmation(to_email: str, subject: str, body: str):
    try:
        # Create a multipart message
        message = MIMEMultipart("alternative")
        message["From"] = FROM_EMAIL
        message["To"] = to_email
        message["Subject"] = subject
        
        # Create plain text version of the email (fallback)
        plain_text = "Your booking has been confirmed. Please enable HTML in your email client to view the full details."
        
        # Add plain text and HTML parts
        part1 = MIMEText(plain_text, "plain")
        part2 = MIMEText(body, "html")
        
        # Attach parts to the message
        message.attach(part1)
        message.attach(part2)  # HTML is the preferred format

        logger.info(f"Sending HTML email to {to_email} from {FROM_EMAIL} via {SMTP_HOST}:{SMTP_PORT}")

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