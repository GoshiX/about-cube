from PIL import Image, ImageDraw
import random
import math

vec_len = 25

def count_black_pixels_along_vector(image, start_pixel, angle_degrees, max_steps=vec_len):
    """
    Функция для подсчета черных пикселей вдоль вектора, заданного углом.

    Параметры:
    - image: изображение в виде numpy массива (черно-белое, 0 - черный)
    - start_pixel: начальный пиксель (x, y)
    - angle_degrees: угол поворота вектора в градусах (0 - вправо, 90 - вверх)
    - max_steps: максимальное количество шагов (если None, идем до края изображения)

    Возвращает:
    - Количество черных пикселей вдоль вектора
    """
    # Преобразуем угол в радианы
    angle_rad = math.radians(angle_degrees)
    
    # Вычисляем направление вектора
    dx = math.cos(angle_rad)
    dy = -math.sin(angle_rad)  # отрицательное значение, так как ось Y направлена вниз
    
    x, y = start_pixel[0], start_pixel[1]
    width, height = image.size
    
    count = 0
    
    # Определяем максимальное количество шагов, если не задано
    if max_steps is None:
        # Вычисляем максимальное расстояние до границы изображения
        max_dist = max(width, height)
        max_steps = int(max_dist * 1.5)  # с запасом
        
    steps = 0
    
    while steps < max_steps:
        # Проверяем, вышли ли за границы изображения
        if x < 0 or y < 0 or x >= width or y >= height:
            break
        
        # Проверяем, является ли пиксель черным (значение 0)



        if image.getpixel((int(x), int(y))) < 110:
            count += 1
        
        # Перемещаемся вдоль вектора
        x += dx
        y += dy
        steps += 1
    
    return count

def get_vec_amount(image, start_pixel, steps, ver=1):

    dots = []

    for ang in range(steps):
        dots.append(count_black_pixels_along_vector(image, start_pixel, ang*360/steps))

    dots = [1 if dot > vec_len*ver else 0 for dot in dots]

    ans = 0
    if dots[0] == 1 and dots[-1] == 1:
        ans -= 1
    if dots[0] == 1:
        ans += 1
    for i in range(1, len(dots)):
        if dots[i] == 1 and dots[i-1] != 1:
            ans += 1
    return ans

def clustering(data, dist=13):
    res = []

    for coord in data:
        add = False
        for i in range(len(res)):
            if abs(coord[0] - res[i][0][0]) < 13 and abs(coord[1] - res[i][0][1]) < 13:
                res[i].append(coord)
                add = True
        if not add:
            res.append([coord])

    return res

def meaning(data):
    answer = []

    for i in range(len(data)):
        answer.append([0, 0])
        for j in range(len(data[i])):
            answer[-1][0] += data[i][j][0]
            answer[-1][1] += data[i][j][1]
        answer[-1] = (answer[-1][0] // len(data[i]), answer[-1][1] // len(data[i]))

    return answer

def solve(image_path, debug = False):
    imagePIL = Image.open(image_path)
    draw = ImageDraw.Draw(imagePIL)

    # Convert to grayscale
    gray_image = imagePIL.convert("L")

    # Find coordinates of all black pixels
    black_pixel_coords = [(x, y) for x in range(gray_image.width) for y in range(gray_image.height) if gray_image.getpixel((x, y)) < 110]

    if debug:
        for coord in black_pixel_coords:
            draw.point(coord, fill=(255, 0, 0))
        imagePIL.save("res.png")

    answer_pix = clustering([p for p in black_pixel_coords if get_vec_amount(gray_image, p, 100) == 3])

    answer = meaning(answer_pix)

    answer.sort(key=lambda x: abs(3 - get_vec_amount(gray_image, x, 360)))

    if debug:
        for coord in answer[:8]:
            draw.ellipse((coord[0] - 5, coord[1] - 5, coord[0] + 5, coord[1] + 5), fill=(0, 0, 255))
        imagePIL.save("res.png")

    return answer[:min(8, len(answer))]
