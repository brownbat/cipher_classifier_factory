import smtplib
import json
from email.mime.text import MIMEText
import time

FILE_PATH = 'credentials.json'

def get_credentials(f_path=None):
    if f_path is None:
        f_path = FILE_PATH
    with open(f_path, 'r') as file:
        credentials = json.load(file)
    return credentials


def send_email(message=None, test=False):
    current_time = time.ctime(time.time())
    credentials = get_credentials('credentials.json')
    email = credentials['email']
    password = credentials['password']
    sender = email
    receiver = email
    password = password  # App password you generated
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


def send_discord_notification(message):
    current_time = time.ctime(time.time())
    message = f"Training job completed at {current_time}.\n" + message
    
    data = {
        "content": message,
        "username": "Cipher Classifier"
    }

    credentials = get_credentials()
    webhook = credentials['webhook']
    result = requests.post(webhook, json=data)

    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    else:
        print("Payload delivered successfully, code {}.".format(result.status_code))

if __name__ == "__main__":
    # send_email(test=True)
    send_discord_notification("TEST")

