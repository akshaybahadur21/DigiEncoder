from keras.models import load_model
import numpy as np
import cv2
from collections import deque

model_simple = load_model('Simple_Autoencoder.h5')
model_deep = load_model('Deep_Autoencoder.h5')
model_conv = load_model('Convolutional_Autoencoder.h5')


def keras_predict(image):
    processed = keras_process_image(image)
    decoded_image_simple = model_simple.predict(processed)
    decoded_image_deep = model_deep.predict(processed)
    decoded_image_conv = model_conv.predict(np.reshape(processed, (-1, 28, 28, 1)))
    return decoded_image_simple, decoded_image_deep, decoded_image_conv


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x * image_y * 1))
    img = img.astype('float32') / 255.
    return img


def adjust(decoded_image_simple, decoded_image_deep, decoded_image_conv):
    decoded_image_simple = decoded_image_simple.astype('float32') * 255
    decoded_image_deep = decoded_image_deep.astype('float32') * 255
    decoded_image_conv = decoded_image_conv.astype('float32') * 255
    decoded_image_simple = np.reshape(decoded_image_simple, (28, 28, 1))
    decoded_image_deep = np.reshape(decoded_image_deep, (28, 28, 1))
    decoded_image_conv = np.reshape(decoded_image_conv, (28, 28, 1))
    return decoded_image_simple, decoded_image_deep, decoded_image_conv


def main():
    cap = cv2.VideoCapture(0)
    Lower_green = np.array([110, 50, 50])
    Upper_green = np.array([130, 255, 255])
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    decoded_image_simple = np.zeros((28, 28, 1), dtype=np.uint8)
    decoded_image_deep = np.zeros((28, 28, 1), dtype=np.uint8)
    decoded_image_conv = np.zeros((28, 28, 1), dtype=np.uint8)

    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, Lower_green, Upper_green)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.bitwise_and(img, img, mask=mask)
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
        elif len(cnts) == 0:
            if len(pts) != []:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts) >= 1:
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = blackboard_gray[y:y + h, x:x + w]
                        decoded_image_simple, decoded_image_deep, decoded_image_conv = keras_predict(digit)
                        decoded_image_simple, decoded_image_deep, decoded_image_conv = adjust(decoded_image_simple,
                                                                                              decoded_image_deep,
                                                                                              decoded_image_conv)
            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imshow("Frame", img)
        cv2.imshow('Simple',
                   cv2.resize(np.array(decoded_image_simple, dtype=np.uint8), (200, 200), interpolation=cv2.INTER_AREA))
        cv2.imshow('Deep',
                   cv2.resize(np.array(decoded_image_deep, dtype=np.uint8), (200, 200), interpolation=cv2.INTER_AREA))
        cv2.imshow('Conv',
                   cv2.resize(np.array(decoded_image_conv, dtype=np.uint8), (200, 200), interpolation=cv2.INTER_AREA))
        k = cv2.waitKey(10)
        if k == 27:
            break


main()
