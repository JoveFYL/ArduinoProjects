from tkinter import W
import numpy as np, sys, cv2

img = cv2.imread("sample.jpg")
cv2.imshow('idPhotoImage', img)
print("Shape:", img.shape)       # (H, W, 3)
print("Data type:", img.dtype)   # usually uint8
print("Pixel value at (100, 100):", img[100, 100])  # [B, G, R]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # array of unsigned int, calculated using a formula from its RGB values
blur = cv2.GaussianBlur(img, (5, 5), 10) # array of unsigned int
cv2.imshow("Grayscale", gray)
cv2.imshow("Blurred", blur)

# Resizing
roi = img[100:200, 200:400]
cv2.imshow("roi", roi)

# Apply binary threshold: convert to pure black and white
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow("Thresholded", thresh)

# masking
mask = gray > 100 # array of boolean (T if > 100)
masked = np.zeros_like(img) # array of uint8
masked[mask] = img[mask] # copies pixel values from img into masked if mask = true

cv2.waitKey(0)
cv2.destroyAllWindows()
