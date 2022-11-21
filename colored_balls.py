import cv2 as cv
import time
import numpy as np

def find_key(dict, value):
    res = ''

    for k, v in dict.items():
        if v == value:
            return k

    return None

cv.namedWindow("Camera", cv.WINDOW_GUI_NORMAL)

cam = cv.VideoCapture(0)

measures = []
hsv = []

lower_orange = np.array([0, 130, 200])
upper_orange = np.array([20, 230 , 255])
lower_red = np.array([0, 160, 110])
upper_red = np.array([25, 230 , 170])
lower_yellow = np.array([26, 120, 100])
upper_yellow = np.array([55, 230 , 190])
lower_blue = np.array([70, 190, 140])
upper_blue = np.array([140, 240 , 240])

random_values = np.random.permutation(3)
res_seq_dict = {'Red' : random_values[0], 'Yellow' : random_values[1], 'Blue' : random_values[2]}
res_seq = sorted(res_seq_dict.values())

is_win : bool = False

while cam.isOpened():
    ret, frame = cam.read()

    text_1 = f"{find_key(res_seq_dict, res_seq[0])}, {find_key(res_seq_dict, res_seq[1])}, {find_key(res_seq_dict, res_seq[2])}"    # Шарики справо налево (поменять имя переменной ниже (cv.putText))
    text_2 = f"{find_key(res_seq_dict, res_seq[2])}, {find_key(res_seq_dict, res_seq[1])}, {find_key(res_seq_dict, res_seq[0])}"     # Шарики слева направо (поменять имя переменной ниже (cv.putText))
    cv.putText(frame, f"SEQ = {text_2}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 127))

    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
    # mask_orange = cv.erode(mask_orange, None, iterations=2)
    # mask_orange = cv.dilate(mask_orange, None, iterations=2)
    mask_red = cv.inRange(hsv, lower_red, upper_red)
    mask_red = cv.erode(mask_red, None, iterations=2)
    mask_red = cv.dilate(mask_red, None, iterations=2)
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_yellow = cv.erode(mask_yellow, None, iterations=2)
    mask_yellow = cv.dilate(mask_yellow, None, iterations=2)
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    mask_blue = cv.erode(mask_blue, None, iterations=2)
    mask_blue = cv.dilate(mask_blue, None, iterations=2)

    # contours_orange, _ = cv.findContours(mask_orange, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv.findContours(mask_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    text_res = ""
    if len(contours_yellow) > 0 and len(contours_blue) > 0 and len(contours_red) > 0:
        x_y = [[0, 0],  # red
               [0, 0],  # yellow
               [0, 0]   # blue
               ]

        (x_y[0][0], x_y[0][1]), radius_0 = cv.minEnclosingCircle(max(contours_red, key=cv.contourArea))
        (x_y[1][0], x_y[1][1]), radius_1 = cv.minEnclosingCircle(max(contours_yellow, key=cv.contourArea))
        (x_y[2][0], x_y[2][1]), radius_2 = cv.minEnclosingCircle(max(contours_blue, key=cv.contourArea))

        cv.circle(frame, (int(x_y[0][0]), int(x_y[0][1])), int(radius_0), (0, 255, 2))
        cv.circle(frame, (int(x_y[1][0]), int(x_y[1][1])), int(radius_1), (0, 255, 2))
        cv.circle(frame, (int(x_y[2][0]), int(x_y[2][1])), int(radius_2), (0, 255, 2))

        res_dict = {'Red' : x_y[0][0], 'Yellow' : x_y[1][0], 'Blue' : x_y[2][0]}
        res = sorted(res_dict.values())
        
        text_res = f"{find_key(res_dict, res[0])}, {find_key(res_dict, res[1])}, {find_key(res_dict, res[2])}"
        cv.putText(frame, f"YOUR_SEQ = {text_res}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 127))

        if text_1 == text_res:
            cv.putText(frame, f"YOU_WIN", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 127))
            random_values = np.random.permutation(3)
            res_seq_dict = {'Red' : random_values[0], 'Yellow' : random_values[1], 'Blue' : random_values[2]}
            res_seq = sorted(res_seq_dict.values())
            is_win = True

    cv.imshow("Camera", frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

    if is_win:
        time.sleep(2)   # ЗАДЕРЖКА
        is_win = False

cam.release()
cv.destroyAllWindows()