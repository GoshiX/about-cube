import cv2
import numpy as np
from PIL import Image

def solve(image_path, debug = False):
    output_path = "res.png"
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Изображение не загружено, проверьте путь!")
    
    # Преобразуем в градации серого и применяем размытие
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Детектируем углы (Shi-Tomasi)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=8, qualityLevel=0.01, minDistance=15)

    # Преобразуем координаты в целые числа
    corners = np.int32(corners)
    
    if debug:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        cv2.imwrite(output_path, image)
        print(f"Изображение с отмеченными вершинами сохранено в: {output_path}")
    
    vertices = [(int(x), int(y)) for corner in corners for x, y in corner.ravel().reshape(-1, 2)]
    return vertices

# input_image = "test.png"
# print(solve(input_image, True))