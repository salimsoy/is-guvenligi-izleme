import cv2
from ultralytics import YOLO
from src.camera import VideoAkisi
from src.warning import HumanErrorDetection
from src.logger import Logger
from src.email_sender import EmailSender
from src.config import CAMERA_ID, MODEL_PATH, SENDER_MAIL, SENDER_PASSWORD, RECIPIENT_MAIL

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