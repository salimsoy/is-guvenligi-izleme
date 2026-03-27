import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailSender:
    def __init__(self, sender_mail, sender_pass, to_mail):
        self.sender_mail = sender_mail
        self.sender_pass = sender_pass
        self.to_mail = to_mail

    def alert_gonder(self, subject, message_body, foto_path):
        try:
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.sender_mail
            msg['To'] = self.to_mail
            
            violation_msg = f"{message_body}\n ihlal fotoğrafının konumu:\n{foto_path}"
            msg.attach(MIMEText(violation_msg, "plain"))
        
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_mail, self.sender_pass)
            server.send_message(msg)
            server.quit()
            print(f"gönderildi: {subject}")
        except Exception as e:
            print(f" HATA: {e}")