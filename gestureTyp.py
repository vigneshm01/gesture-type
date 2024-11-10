
import cv2
import mediapipe as mp
import time

import word_pred


class HandDetect():
    # Setter function (constructor)
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, modelComplexity=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    # Find hand
    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #         print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    # Find position of hand ( i.e., get landmarks )
    def find_position(self, img, hand_no=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 6, (0, 255, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox


def count_fingers(lmList):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    if len(lmList) != 0:
        if lmList[4][1] > lmList[0][1]:
            # right hand is raised
            if lmList[tip_ids[0]][1] > lmList[tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # print(fingers,'right')
        else:
            # left hand is raised
            # print(fingers,'left')
            if lmList[4][1] < lmList[4 - 1][1]:
                # thumb is open
                fingers.append(1)

            else:
                # thumb is close
                fingers.append(0)
            for id in range(1, 5):
                if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers
    return [-1]


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetect()

    st_time = time.time()
    tens_time=time.time()
    last_fingers = ['', '', '', '', '']

    tx = ""

    """lett = [
        ['a', 'b', 'c', 'd', 'e'],
        ['f', 'g', 'h', 'i', 'j'],
        ['k', 'l', 'm', 'n', 'o'],
        ['p', 'q', 'r', 's', 't'],
        ['u', 'v', 'w', 'x', 'y']]"""
    lett = [
        ['a', 'e', 'i', 'o', 'u'],
        ['t', 'n', 's', 'h', 'r'], 
        ['d', 'l', 'c', 'm', 'w'],
        ['f', 'g', 'y', 'p', 'b'], 
        ['v', 'k', 'j', 'x', 'q']
    ]
    i = 0
    sug = ''

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm, bbox = detector.find_position(img)
        r_fingers = count_fingers(lm)
        str_fingers = ''.join(str(e) for e in r_fingers)
        total_fingers = sum(r_fingers)

        if time.time()-st_time > 1:
            st_time = time.time()
            if ([last_fingers[4]] * 2) == last_fingers[3:]:
                if i == -1 and total_fingers>0:
                    if str_fingers != '10001':
                        temp_tx = tx[-20:]
                        temp_tx = temp_tx.split(' ')
                        ed_len = len(temp_tx[-1])
                        tx = tx[:-ed_len]
                        tx += sug[total_fingers-1]
                    i = 0
                elif str_fingers == '11000':
                    i += 1
                elif str_fingers == '10001' and len(sug)>1:
                    i = -1
                elif str_fingers == '01001':
                    tx += ' '
                    i = 0
                elif str_fingers == '11001':
                    tx = tx[:-1]
                    i = 0
                elif total_fingers == 1:
                    tx += lett[i][total_fingers - 1]
                    i = 0
                elif total_fingers == 2:
                    tx += lett[i][total_fingers - 1]
                    i = 0
                elif total_fingers == 3:
                    tx += lett[i][total_fingers - 1]
                    i = 0
                elif total_fingers == 4:
                    tx += lett[i][total_fingers - 1]
                    i = 0
                elif total_fingers == 5:
                    tx += lett[i][total_fingers - 1]
                    i = 0
                    
                if i>4:
                    i=0
                    
        elif time.time()-tens_time > 0.10:
            tens_time = time.time()
            last_fingers.pop(0)
            last_fingers.append(str_fingers)

        word = tx[-20:].split(" ")[-1:]
        word = ''.join(word[0])
        sug = ""
        sug_tx = ""
        if len(word)>2:
            sug = word_pred.predict(word)
            for e in sug:
                sug_tx += e+' '
        if i == -1:
            sug_tx = ">> "+sug_tx

        cv2.putText(img, str(tx), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        hint = '1:'+lett[i][0]+' | 2:'+lett[i][1]+' | 3:'+lett[i][2]+' | 4:'+lett[i][3]+' | 5:'+lett[i][4]
        cv2.putText(img, str(hint), (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        cv2.putText(img,str(sug_tx), (10, 170), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 0), 2)

        cv2.imshow("Image", img)

        k = cv2.waitKey(30) & 0xff
        # exit if Esc is pressed
        if k == 27:
            break


# Testing all function
if __name__ == "__main__":
    main()
