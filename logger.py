from threading import Thread
import queue
from datetime import datetime, timedelta
import cv2
import json
import os
from config import MAX_LOG_WAIT_MINUTE, LOG_FİLE, FOTO_FOLDER

class Logger:
    def __init__(self, mail_class=None):
        # Gerekli değişkenleri tanımlar
        self.log_file = LOG_FİLE
        self.foto_folder = FOTO_FOLDER
        self.minute_threshold = MAX_LOG_WAIT_MINUTE
        self.mail_sender = mail_class
        
        # Fotoğrafın kaydedileceği klasör yoksa oluşturur
        if not os.path.exists(self.foto_folder):
            os.makedirs(self.foto_folder)
            
        # Log işlemlerinin ana döngüyü yavaşlatmaması için kuyruk ve thread başlatır
        self.log_queue = queue.Queue()
        self.log_tread = Thread(target=self.logger_action, daemon=True)
        self.log_tread.start()
        
    def chack_log(self, now_time, person_id, message, date_str):
        try:
            # Belirlenen bekleme süresine zaman kısmını hesaplar
            determined_threshold = now_time - timedelta(minutes=self.minute_threshold)
            border_clock = determined_threshold.strftime('%H:%M')
            
            if not os.path.exists(self.log_file):
                return True
            
            with open(self.log_file, "r", encoding='utf-8') as f:
                lines = f.readlines()
            
            # Log dosyasını sondan başa okuyarak yakın zamanda aynı ihlalin olup olmadığına bakar
            for line in reversed(lines):
                if line:
                    log_data = json.loads(line)
                    registration_person = log_data["person_id"]
                    registration_breach = log_data["ihlal_turu"]
                    registration_date = log_data["tarih"]
                    registration_time = log_data["saat"]
                    
                    if registration_person == person_id and registration_breach == message:
                        if border_clock <= registration_time and registration_date == date_str:
                            return False 
                        else:
                            break
                else:
                    break
            return True
                
        except Exception as e:
            print(f"HATA: {e}")
            return False
         
    def logger_action(self):
        # Kuyruktaki logları sırayla işler
        while True:
            camera_id, log_massage, frame = self.log_queue.get()
            try:
                person = log_massage["person"]
                person_id = int(person["ID"])
                message = log_massage["mesaj"]
                
                # Olayın gerçekleştiği zamanı değişkene kaydeder
                now_time = datetime.now()
                date_str = now_time.strftime('%Y-%m-%d')
                date_hour = now_time.strftime('%H:%M')
                date_hour_file = now_time.strftime('%H-%M')
                
                # Aynı ihlal için henüz süre dolmadıysa log atılmasını engeller
                log_check = self.chack_log(now_time, person_id, message, date_str)
                foto_path = f"{self.foto_folder}/id_{person_id}_{date_str}_{date_hour_file}.jpg"
                
                if log_check:
                    # İhlal anının fotoğrafını kaydeder
                    if frame is not None:
                        cv2.imwrite(foto_path, frame)
                    
                    if os.path.exists(foto_path):
                        curr_photo_path = foto_path
                    else:
                        curr_photo_path = None
                    # Log kaydını JSON formatında dosyaya ekler
                    entry = {
                        "person_id": person_id,
                        "tarih": date_str,
                        "saat": date_hour,
                        "kamera_id": str(camera_id),
                        "ihlal_turu": message,
                        "fotograf_yolu": curr_photo_path
                    }
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        json_record = json.dumps(entry, ensure_ascii=False)
                        f.write(json_record + "\n")
                
                    print("başarıyla log kaydedildi")
                
                    # Mail sınıfı verilmişse uyarı maili gönderir
                    if self.mail_sender:
                        subject = f"Guvenlik ihlali - kamera: {camera_id}"
                        message_body = f"{person_id} ID li personel guvenlik ihlal durumu: {message}"
                        self.mail_sender.alert_gonder(subject, message_body, foto_path)
                
            except Exception as e:
                print(f"HATA: {e}")
    
            finally:
                # sisteme işinin bittiğini söyler
                self.log_queue.task_done()
            
    def add_logger(self, camera_id, breach_list, frame):
        # Ana koddan gelen log bilgilerini kuyruğa atar
        self.log_queue.put((camera_id, breach_list, frame))