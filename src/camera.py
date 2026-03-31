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