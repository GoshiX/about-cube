import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

dpi = 120

def add_noise_and_texture(image):
    """Добавляет шум, текстуру и эффект фотографии"""

    width, height = image.size
    gradient = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(gradient)
    
    for _ in range(random.randint(1, 3)):
        x, y = random.randint(0, width), random.randint(0, height)
        radius = random.randint(100, 500)
        for i in range(radius, 0, -10):
            alpha = int(125 * (1 - i/radius))
            draw.ellipse([(x-i, y-i), (x+i, y+i)], fill=alpha)
    
    darken = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    darken.putalpha(gradient)
    image = Image.alpha_composite(image, darken)

    texture = Image.effect_noise((width//4, height//4), 128).resize((width, height))
    
    # Наложение темных областей
    mask = texture.point(lambda p: p if p < 50 else 0)
    dark = Image.new('RGBA', (width, height), (0, 0, 0, 30))
    dark.putalpha(mask)
    image = Image.alpha_composite(image, dark)

    # Добавляем легкое размытие (имитация дефекта фокуса)
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))

    return image

def generate_transparent_cube():
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Define cube vertices
    vertices = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])

    # Define edges
    edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]]
    ]

    # Plot cube
    for edge in edges:
        ax.plot3D(*zip(*edge), color='black', linewidth=2)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_axis_off()

    ax.view_init(elev=random.uniform(-90, 90), azim=random.uniform(-180, 180))
    fig.tight_layout(pad=0)

    # Draw the plot
    fig.canvas.draw()

    # Get dimensions
    width, height = fig.canvas.get_width_height()

    # Get image as RGBA buffer and convert to PIL Image
    image = Image.frombuffer(
        'RGBA', (width, height),
        fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1
    )

    # Добавляем шумы и текстуры
    image = add_noise_and_texture(image)

    draw = ImageDraw.Draw(image)

    # Convert 3D vertices to 2D screen coordinates
    pixel_coords = []
    for vertex in vertices:
        x3, y3, _ = proj3d.proj_transform(*vertex, ax.get_proj())
        x2, y2 = ax.transData.transform((x3, y3))
        pixel_coords.append((int(x2), int(height - y2)))  # Flip y for PIL

    # Draw red dots on projected vertices
    # for x, y in pixel_coords:
    #     draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill='red', outline='red')

    final_path = "ideal-photo/" + ";".join(f"{coord[0]},{coord[1]}" for coord in pixel_coords) + ".png"
    # final_path = "test.png"
    image.save(final_path)

    plt.close(fig)

    print(f"Изображение с выделенными вершинами сохранено как '{final_path}'")
    return pixel_coords, final_path

for i in range(1000):
    generate_transparent_cube()
    print(i+1)