import cv2
import numpy as np
import functions as fnc



rc = 0.2
r_crc = (1-rc**2)**0.5
 
img = cv2.imread('./MW.jpg')
height, width, c = img.shape
img = cv2.resize(img, (width, 500))
 

 
i = 0
 
while True:
    i += 1
     

    l = img[:, :(i % width)]
    r = img[:, (i % width):]
 
    img1 = np.hstack((r, l))

    crop_img = img1[0:500, 0:500]

    lnsd_img = fnc.lens(crop_img, rc, 0, 3, 3)/255
    cv2.putText(img=lnsd_img, text='Press Esc to exit', org=(10, 480), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.75, color=(0, 255, 0),thickness=2)
     

    cv2.imshow('Movie', lnsd_img)
 
    if cv2.waitKey(1) == 27:
       
        cv2.destroyAllWindows()
        break