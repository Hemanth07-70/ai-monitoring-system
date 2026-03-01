import os
import time
import cv2
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

class AlertEngine:
    def __init__(self, sender_email=None, receiver_email=None, password=None):
        self.sender_email = sender_email
        self.receiver_email = receiver_email
        self.password = password

    def get_location(self):
        return "GPS: 12.9716N, 77.5946E (Live Location)"

    def trigger(self, frame, message):
        print(f"!!! TRIGGERED !!!: {message}")
        
        # 1. Local notification
        os.system(f'osascript -e \'display notification "{message}" with title "Distress Alert System"\'')
        
        # 2. Email alert
        if all([self.sender_email, self.receiver_email, self.password]):
            self.send_email(frame, message)
        else:
            print("Email alert skipped (not configured in config.py)")

    def send_email(self, frame, message):
        try:
            msg = MIMEMultipart()
            msg['Subject'] = "🆘 EMERGENCY: Distress Signal Detected"
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email

            body = f"{message}\nTime: {time.ctime()}\nLocation: {self.get_location()}"
            msg.attach(MIMEText(body, 'plain'))

            _, img_encoded = cv2.imencode('.jpg', frame)
            img_att = MIMEImage(img_encoded.tobytes(), name="snapshot.jpg")
            msg.attach(img_att)

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(self.sender_email, self.password)
                server.send_message(msg)
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {e}")
