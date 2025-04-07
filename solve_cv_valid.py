import cv2
import numpy as np
from PIL import Image, ImageDraw
import math

from solve_valid import get_vec_amount, clustering, meaning

def solve(image_path, choose, debug = False):
    output_path = "res.png"
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Изображение не загружено, проверьте путь!")
    
    # Преобразуем в градации серого и применяем размытие
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Детектируем углы (Shi-Tomasi)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=choose, qualityLevel=0.01, minDistance=20)

    # Преобразуем координаты в целые числа
    corners = np.int32(corners)
    
    vertices = [(int(x), int(y)) for corner in corners for x, y in corner.ravel().reshape(-1, 2)]

    print(vertices)

    vertices = meaning(clustering(vertices))

    imagePIL = Image.open(image_path)
    gray_image = imagePIL.convert("L")
    best_p = 1
    best_p_res = -1
    for p in range(20, 4, -2):
        res = sum([1 if get_vec_amount(gray_image, i, 360, p/20) == 3 else 0 for i in vertices])
        if res > best_p_res:
            best_p = p / 20
            best_p_res = res
    vertices.sort(key=lambda x: abs(3 - get_vec_amount(gray_image, x, 360, best_p)))

    if debug:
        for corner in vertices:
            x, y = corner[0], corner[1]
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        for corner in vertices[:min(8, len(vertices))]:
            x, y = corner[0], corner[1]
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        cv2.imwrite(output_path, image)
        print(f"Изображение с отмеченными вершинами сохранено в: {output_path}")

    return vertices[:min(8, len(vertices))]

input_image = "test.png"
radius = 30
x, y = 331, 419
left = x - radius
top = y - radius
right = x + radius
bottom = y + radius

img = Image.open(input_image).convert("RGBA")
cropped_img = img.crop((left, top, right, bottom))
draw = ImageDraw.Draw(cropped_img)

# Convert to coordinates relative to the cropped image
rel_x = radius-5
rel_y = radius+3

angle_rad = math.radians(60)
dx = math.cos(angle_rad)
dy = -math.sin(angle_rad)

for i in range(25):
    # Only draw if point is within the cropped image
    if 0 <= rel_x < 2*radius and 0 <= rel_y < 2*radius:
        draw.point((int(rel_x), int(rel_y)), fill="red")
    
    rel_x += dx
    rel_y += dy

cropped_img.save("res.png")