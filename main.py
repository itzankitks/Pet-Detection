import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')


# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1:
    # reads frames from a camera
    ret, img = cap.read()
    
    # convert to gray scale of each framme
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detects Faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # add CAT label around deteced cat
        cv2.putText(img, 'CAT', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    
    # Dispaly an image in a window
    cv2.namedWindow('Pet-Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    
    # wait for Esc key to be pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
