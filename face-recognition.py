import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
# webcam = cv2.flip(cv2.VideoCapture(0),1)

while True:
    read, frame = webcam.read()
    frame = cv2.flip(frame,1)
    gray_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('Face detector',frame)
    key = cv2.waitKey(1)
    if key==32:
        break

webcam.release()
cv2.destroyAllWindows

"""
gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(gray_scaled)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

cv2.imshow('Face detector',img)
cv2.waitKey()

print("hi")
"""