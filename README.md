# YOLO Tabanlı İş Güvenliği İhlal Tespit Sistemi
Bu proje, çalışma ortamlarındaki iş güvenliği kural ihlallerini (baretsiz veya yeleksiz çalışma) gerçek zamanlı olarak tespit etmek için geliştirilmiş yapay zeka tabanlı bir görüntü işleme sistemidir. 
YOLO modeli kullanılarak kişilerin güvenlik ekipmanı takıp takmadığı analiz edilir ihlal durumunda fotoğraf kaydedilir sisteme loglanır ve yetkili kişilere otomatik e-posta uyarısı gönderilir.

## YOLO
YOLO, 2015 yılında Joseph Redmond ve arkadaşları tarafından geliştirilmiştir. O
zamanlar Fast R-CNN de geliştirilen modellerden biriydi, ancak gerçek zamanlı olarak
kullanılamaması olumsuzlukları vardı. Bir görüntüyü tahmin etmek 2-3 saniye
sürüyordu. Buna karşılık YOLOda nihai tahminleri yapmak için ağ üzerinden sadece tek
bir ileri besleme (forward pass) geçişi yeterlidir.

**Çalışma Mantığı**
- Gerçek Zamanlı Nesne Tespiti: YOLO modeli ile kişi, baret ve yelek tespiti yapılır.

- Akıllı Eşleştirme (IoU): Kişi ile ekipman bounding-box'larının (sınırlayıcı kutular) kesişim oranlarını hesaplayarak doğru eşleştirme yapar. Hata payı oranları özelleştirilebilir.

- Kamera Akışı: Threading kullanılarak kamera veya video okuma işlemleri arka plana alınır, böylece FPS düşüşü ve donmalar önlenir.

- Loglama: Aynı kişi için art arda bildirim gönderilmesini engellemek için bekleme süresini kontrol eden bir mekanizma içerir.

- Otomatik Raporlama: İhlal tespit edildiğinde o anın fotoğrafını kaydeder, bir JSON dosyasına loglar ve SMTP üzerinden yetkiliye e-posta atar.

## Modüller

1. config.py

**Çalışma Mantığı:** İçerisinde fonksiyon veya sınıf barındırmaz; sadece sabit değişkenleri (kamera ID'si, YOLO modelinin yolu, hata tolerans oranları, e-posta adresleri ve şifreleri) tutar.

**Amacı:** Projeyi başka bir fabrikaya veya kameraya entegre etmek istediğinde, kodların içinde boğuşmadan sadece bu dosyayı açıp birkaç değeri değiştirerek sistemi yeni ortama uyarlamanı sağlar.

2. camera.py

**Çalışma Mantığı:** Threading kullanır. start() fonksiyonu çağrıldığında, arka planda sonsuz bir döngü başlatır ve kameradan sürekli frame yakalar.

**Amacı:** Ana program (main.py) YOLO ile nesne tespiti yaparken veya ekrana çizim yaparken kameranın donmasını engeller. Ana program Bana en son kareyi ver dediğinde (get_frame), bekleme yapmadan o anki en taze görüntüyü anında sunar. Bu sayede FPS düşüşü yaşanmaz.

3. warning.py

**Sınıflandırma:** YOLO dan gelen sonıuçları ayırır. Sınıf 3 kişi, sınıf 2 yeleksiz, sınıf 1 baretsiz olarak listelere böler

**Kesişim Hesaplanması:**  Bir kişi kutusu ile yeleksiz yada baretsiz kutusu ekranda üst üste geliyor mu diye bakar. Kutuların köşe koordinatlarını (x1, y1, x2, y2) karşılaştırarak matematiksel kesişim alanı hesaplar.

**Tolerans ve Hata Önleme:** Sadece kutular kesişti diye anında ihlal basmaz. ERROR_RATE (kesişim oranı) ve ERROR_THRESHOLD (hatalı kare sayısı) kontrollerinden geçirir. Yani yapay zeka bir anlığına yanlış tespit yapsa bile bu tolerans sayesinde sistemin sahte alarm vermesi engellenir.

4. logger.py

**Çalışma Mantığı:** Thread ve kuyruk yapısı kullanır. İhlal olduğunda ana döngüyü durdurup saniyelerce fotoğraf kaydetmek veya dosya yazmakla uğraşmaz. Görevi kuyruğa atar ve arka planda sırayla işler.

**Log Kontrolü** En önemli özelliği chack_log fonksiyonudur. Aynı kişi (aynı ID) ekranda belirtilen dakika boyunca herhangi bir ihlal durumuyla dolaşıyorsa, yetkiliye sürekli mail atmasını engeller.
Geçmiş log dosyasını okur eğer aynı kişi için MAX_LOG_WAIT_MINUTE süresi henüz dolmamışsa yeni kaydı yapmaz. Süre dolduysa fotoğrafı kaydeder ve JSON dosyasına yeni bir satır ekler ve mail atar.


5. email_sender.py

**Çalışma Mantığı** Pythonda bulunan smtplib ve email.mime kütüphanelerini kullanarak bir mail gönderici sınıftır.

**Amacı: logger.py tarafından bir ihlal kesinleştirildiğinde ve süresi dolduğunda tetiklenir. config.py içindeki gönderici bilgilerini kullanarak Gmail sunucularına bağlanır. 
İhlalin ne olduğunu, kimin yaptığını ve kaydedilen ihlal fotoğrafının bilgisayardaki tam yolunu metin olarak yetkiliye mail atar.

6. main.py

Tüm bu bağımsız modüllerin bir araya getirildiği ve senkronize bir şekilde çalıştırıldığı ana dosyadır.

**Çalışma Mantığı:** Çalıştığı anda Perception sınıfı üzerinden config ayarlarını çeker, YOLO modelini yükler ve camera.py üzerinden video akışını başlatır.

**Ana Döngü:** Sonsuz bir while True döngüsü başlatır. Sırasıyla şu adımları izler:

- Kameradan yada vidyodan yeni frame alır.

- YOLO modeline frame verir ve tespitleri alır.

- Tespit edilen nesne IDsi varsa bunları warning.py ye gönderip kural ihlali var mı diye kontrol eder.

- İhlal dönüyorsa logger.py ye Bunu kaydet ve mail at komutu verir.

- Ekrana kırmızı uyarı çerçevelerini ve YOLOnun çizimlerini yansıtır.

- q tuşuna basılana kadar bunu tekrarlar.

---

## 1. Sistem Ana Modülü (main.py)
```python
import cv2
from ultralytics import YOLO
from camera import VideoAkisi
from warning import HumanErrorDetection
from logger import Logger
from email_sender import EmailSender
from config import CAMERA_ID, MODEL_PATH, SENDER_MAIL, SENDER_PASSWORD, RECIPIENT_MAIL

class Perception:
    def __init__(self):
        # Gerekli değişkenleri tanımlar
        self.camera_path = CAMERA_ID
        self.modal_path = MODEL_PATH
        self.camera_id = str(CAMERA_ID)
        self.iscamera = None
            
    def main(self):
        # YOLO modelini yükler
        model = YOLO(self.modal_path)
        
        # Verilen dosya türüne göre kamera veya video dosyasını başlatır
        if isinstance(self.camera_path, str):
            cap = cv2.VideoCapture(self.camera_path)
            if not cap.isOpened():
                print("Hata: Video açılmadı")
                exit()
            self.iscamera = False
        elif isinstance(self.camera_path, int):
            cap = VideoAkisi(self.camera_path).start()
            self.iscamera = True
        else:
            exit()
            
        print("Çıkmak için q tuşuna basın")
        
        # Gerekli sınıfları çağırır
        human_err = HumanErrorDetection()
        mail_sender = EmailSender(SENDER_MAIL, SENDER_PASSWORD, RECIPIENT_MAIL)
        logger = Logger(mail_sender)
        
        while True:
        
            if not self.iscamera:
                ret, frame = cap.read()
            elif self.iscamera:
                ret, frame = cap.get_frame()
            else:
                break
            
            if not ret:
                break
            
            # frame üzerinde nesne tespiti ve takibi yapar
            resaults = model.track(frame, persist=True)
            drawn_square = resaults[0].plot()
            
            # framede takip edilen nesen var mı diye kontrol eder
            if resaults[0].boxes.id is not None:
                
                # kişileri ve ihlal durumlarını tespit eder
                persons, no_vests, no_helmets = human_err.separate_object(resaults)
                breachs = human_err.main(persons, no_vests, no_helmets, frame)
                
                # Tespit edilen ihlalleri kırmızı kutuya alır
                for breach in breachs:
                    breach_person = breach["person"]
                    x1, y1, x2, y2 = breach_person["kutu"]
                    px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)
                    
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.imshow('YOLO Is Guvenligi Sistemi', frame)
                    # İhlali loglar ve mail atar
                    logger.add_logger(self.camera_id, breach, frame)
                  
            # YOLOnun tespit ettiklerini görüntülememizi sağlar
            cv2.imshow('YOLO Is Guvenligi Sistemi', drawn_square)
            
            # q tuşuna basıldığında program kapanır
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # kamerayı serbest bırakır
        if not self.iscamera:
            cap.release()
        elif self.iscamera:
            cap.stop()
            
        cv2.destroyAllWindows()
        print("Sistem kapandı")
   
        
   
if __name__ == "__main__":
   
    proses = Perception()
    proses.main()
```

## 2. İhlal Tespiti (warning.py)
```python
from config import ERROR_RATE, ERROR_THRESHOLD

class HumanErrorDetection:
    
    def __init__(self):
        # gerekli değişkenleri tanımlar
        self.vests_error_proses = {}
        self.helmets_error_proses = {}
        self.error_threshold = ERROR_THRESHOLD
        self.error_rate = ERROR_RATE
        
    def separate_object(self, resaults):
        persons = []
        no_vests = []
        no_helmets = []
        
        # YOLO sonuçlarından gerekli bilgileri alır
        boxes = resaults[0].boxes.xyxy.cpu().numpy()
        ids = resaults[0].boxes.id.int().cpu().numpy()
        clss = resaults[0].boxes.cls.int().cpu().numpy()
        
        # Tespit edilen nesneleri sınıflarına göre ayırır
        for box, idss, cls_val in zip(boxes, ids, clss):
            if cls_val == 3:  # Kişi
                persons.append({"ID": idss, "kutu": box})
            elif cls_val == 2: # Yeleksiz
                no_vests.append({"kutu": box})
            elif cls_val == 1: # Baretsiz/Kasksız
                no_helmets.append({"kutu": box})
                
        return persons, no_vests, no_helmets

    def vest_common_area(self, person, articles):
        ix1, iy1, ix2, iy2 = person["kutu"]
        person_id = person["ID"]
        reset = True
        
        # kişiler ile yeleksiz olanların kesişimlerini hesaplar
        for article in articles:
            yx1, yy1, yx2, yy2 = article["kutu"]
            
            intersection_x1 = max(ix1, yx1)
            intersection_y1 = max(iy1, yy1)
            intersection_x2 = min(ix2, yx2)
            intersection_y2 = min(iy2, yy2)
            
            # Kesişim yoksa geçer
            if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
                continue
            
            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
            article_area = (yx2 - yx1) * (yy2 - yy1)
            
            if article_area <= 0:
                continue
            
            # Kesişim oranı eşik değerden büyükse sayacı günceller
            article_rate = intersection_area / article_area
            if article_rate > self.error_rate:
                reset = False
                
                # İhlal eşiği aşıldıysa durumu bildirir ve sayacı sıfırlar
                if person_id in self.vests_error_proses:
                    self.vests_error_proses[person_id] += 1
                    
                    if self.vests_error_proses[person_id] >= self.error_threshold:
                        self.vests_error_proses[person_id] = 0
                        return True
                    
                else:
                    self.vests_error_proses[person_id] = 1
                
                break
            
        # ihlal yoksa sayacı sıfırla
        if reset:
            self.vests_error_proses[person_id] = 0
        
        return False
    
    def helmets_common_area(self, person, articles):
        ix1, iy1, ix2, iy2 = person["kutu"]
        person_id = person["ID"]
        reset = True
        
        # kişiler ile baretsiz olanların kesişimlerini hesaplar gerisi yeleksiz ihlaliyle aynı
        for article in articles:
            yx1, yy1, yx2, yy2 = article["kutu"]
            
            intersection_x1 = max(ix1, yx1)
            intersection_y1 = max(iy1, yy1)
            intersection_x2 = min(ix2, yx2)
            intersection_y2 = min(iy2, yy2)
            
            if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
                continue
            
            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
            article_area = (yx2 - yx1) * (yy2 - yy1)
            
            if article_area <= 0:
                continue
            
            article_rate = intersection_area / article_area
            if article_rate > self.error_rate:
                reset = False
                
                if person_id in self.helmets_error_proses:
                    self.helmets_error_proses[person_id] += 1
                    
                    if self.helmets_error_proses[person_id] >= self.error_threshold:
                        self.helmets_error_proses[person_id] = 0
                        print("reset devrede")
                        return True
                    
                else:
                    self.helmets_error_proses[person_id] = 1
                
                break
            
        if reset:
            self.helmets_error_proses[person_id] = 0
        
        return False
    
    def massage_append(self, lists, person, txt):
        # İhlal listesine yeni kaydı ekler
        lists.append({"person": person, "mesaj": txt})
        
    def main(self, persons, no_vests, no_helmets, frame):
        breachs = []
        
        # Her bir kişi için ekipman durumlarını kontrol eder
        for person in persons:
            case_vest = self.vest_common_area(person, no_vests)
            case_helmets = self.helmets_common_area(person, no_helmets)
            
            if case_vest and not case_helmets:
                self.massage_append(breachs, person, "yeleksiz-kaskli")  
            elif not case_vest and case_helmets:
                self.massage_append(breachs, person, "yelekli-kasksiz")
            elif case_vest and case_helmets:
                self.massage_append(breachs, person, "yeleksiz-kasksiz")
                
        return breachs
```

## 3. Kayıt (logger.py)
```python
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
                
                    # Log kaydını JSON formatında dosyaya ekler
                    entry = {
                        "person_id": person_id,
                        "tarih": date_str,
                        "saat": date_hour,
                        "kamera_id": str(camera_id),
                        "ihlal_turu": message,
                        "fotograf_yolu": foto_path
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
```
## 4. Kamera (camera.py)
```python
import cv2
from threading import Thread

class VideoAkisi:
    def __init__(self, source=0):
        # Kamerayı başlatır ve ilk kareyi okur
        self.cap = cv2.VideoCapture(source)
        self.ret, self.frame = self.cap.read()
        self.stoped = False
    
    def start(self):
        # Kare okuma işlemini thread ile başlatır
        t = Thread(target=self.read_frame, args=())
        t.daemon = True
        t.start()
        return self
    
    def read_frame(self):
        # sürekli yeni kareyi alır ve kaydeder
        while True:
            if self.stoped:
                self.cap.release()
                return
            
            self.ret, self.frame = self.cap.read()
    
    def get_frame(self):
        # Ana döngünün kullanması için en son okunan kareyi verir
        return self.ret, self.frame
    
    def stop(self):
        # kameradan sürekli kare okuma döngüsünü durdurur
        self.stoped = True
```
## 5. Mail Gönderici (email_sender.py)
```python
import cv2
from threading import Thread

class VideoAkisi:
    def __init__(self, source=0):
        # Kamerayı başlatır ve ilk kareyi okur
        self.cap = cv2.VideoCapture(source)
        self.ret, self.frame = self.cap.read()
        self.stoped = False
    
    def start(self):
        # Kare okuma işlemini thread ile başlatır
        t = Thread(target=self.read_frame, args=())
        t.daemon = True
        t.start()
        return self
    
    def read_frame(self):
        # sürekli yeni kareyi alır ve kaydeder
        while True:
            if self.stoped:
                self.cap.release()
                return
            
            self.ret, self.frame = self.cap.read()
    
    def get_frame(self):
        # Ana döngünün kullanması için en son okunan kareyi verir
        return self.ret, self.frame
    
    def stop(self):
        # kameradan sürekli kare okuma döngüsünü durdurur
        self.stoped = True
```

## 6. config.py
```python
import os
from dotenv import load_dotenv

load_dotenv()

# .env dosyasında değişkenleri çeker
CAMERA_ID = os.getenv("CAMERA_ID")
MODEL_PATH = os.getenv("MODEL_PATH")
LOG_FİLE = os.getenv("LOG_FİLE")
FOTO_FOLDER = os.getenv("FOTO_FOLDER")
SENDER_MAIL = os.getenv("SENDER_MAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_MAIL = os.getenv("RECIPIENT_MAIL")
# Gerekli değişkenleri belirler
ERROR_THRESHOLD = 40
ERROR_RATE = 0.7
MAX_LOG_WAIT_MINUTE = 5
```

