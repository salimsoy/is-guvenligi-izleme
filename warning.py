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