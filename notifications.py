import smtplib
import json
from email.mime.text import MIMEText
import time

def get_credentials(file_path):
    with open(file_path, 'r') as file:
        credentials = json.load(file)
    return credentials


def send_email(message=None, test=False):
    credentials = get_credentials('credentials.json')
    email = credentials['email']
    password = credentials['password']
    sender = email
    receiver = email
    password = password  # App password you generated
    current_time = time.ctime(time.time())
    subject = "Training job notification"
    message = f"Training job completed at {current_time}.\n" + message
    if test:
        subject = "[TEST]" + subject
        message = "[TEST]" + message

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)

    print("Email sent successfully.")


if __name__ == "__main__":
    send_email(test=True)

