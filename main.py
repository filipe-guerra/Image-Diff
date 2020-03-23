from skimage.measure import compare_ssim
import cv2
import numpy as np

# def image_diff(before, after):

before = cv2.imread('left.png')
after = cv2.imread('right.png')

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = compare_ssim(before_gray, after_gray, full=True)
print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1] 
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours[0:1]:
    area = cv2.contourArea(c)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
        

        # Pega o contorno para uma nova imagem
        crop_contour = before_gray[y:y+h, x:x+w]
        # faz uma análise do contorno na outra imagem
        res = cv2.matchTemplate(after_gray, crop_contour, cv2.TM_CCOEFF_NORMED)

        # com comparativo, verifica se existe resultado e desenha a ocorrencia do contorno na nova imagem
        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(after, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        cv2.imwrite('res.png',after)
        # cv2.imwrite('result/contourn.png', crop_contour)



cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff',diff)
cv2.imshow('mask',mask)
cv2.imshow('filled after',filled_after)

cv2.waitKey(0)

cv2.imwrite('result/before.png', before)
cv2.imwrite('result/after.png', after)
cv2.imwrite('result/diff.png',diff)
cv2.imwrite('result/mask.png',mask)
cv2.imwrite('result/filled after.png',filled_after)

# if __name__ == "__main__":
#     image_diff('left.png', 'right.png')