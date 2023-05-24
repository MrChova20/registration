import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Cargar la imagen de la matrícula
image = cv2.imread('matricula2.jpeg')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro Gaussiano para reducir el ruido
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Aplicar umbral adaptativo para resaltar los caracteres de la matrícula
_, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Realizar operaciones de dilatación y erosión para mejorar la calidad de la imagen binaria
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_image = cv2.dilate(threshold_image, kernel, iterations=3)
eroded_image = cv2.erode(dilated_image, kernel, iterations=2)

# Obtener los componentes conectados en la imagen binaria
contours, _ = cv2.findContours(eroded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar los contornos por área y forma
filtered_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    if aspect_ratio > 2 and area > 2000:
        filtered_contours.append(contour)

# Ordenar los contornos de izquierda a derecha
filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])

# Reconocer el texto de cada contorno y concatenarlo en una cadena
recognized_text = ''
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = eroded_image[y:y + h, x:x + w]
    text = pytesseract.image_to_string(roi, config='--psm 7')
    recognized_text += text.strip()

# Guardar la matrícula en un archivo de texto
with open('matricula.txt', 'w') as file:
    file.write(recognized_text)

print("Matrícula reconocida:", recognized_text)
