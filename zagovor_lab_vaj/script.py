import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
parent_dir = "/home/urban/Faks/Obdelava slik in videa/Vaje"
sys.path.append(parent_dir)
from OSV_lib import loadImage, displayImage, equalizeHistogram, computeHistogram, displayHistogram

# Naloga 1:
def equalize_image(iImage):
    # Izravnava histograma vsakega kanala posebej z opencv
    channels = cv2.split(iImage)
    eq_channels = [cv2.equalizeHist(channel) for channel in channels]
    equalized_image = cv2.merge(eq_channels)

    # Ce bi bila potrebna pretvorba v grayscale bi uporabil se to 
    # gray_image = cv2.cvtColor(iImage, cv2.COLOR_BGR2GRAY)
    
    return equalized_image


if __name__ == "__main__":
    image = loadImage("zagovor_lab_vaj/data/marilyn-monroe-484x699-08bit.raw", (484, 699), np.uint8)
    displayImage(image, "Originalna slika")
    # Originalna slika se ze prikaze v grayscale

    equalized_image = equalize_image(image)
    displayImage(equalized_image, "Originalna slika po izravnavi")
    hist, prob, CDF, levels = computeHistogram(image)
    displayHistogram(hist, levels, "Histogram originalne slike")
    hist, prob, CDF, levels = computeHistogram(equalized_image)
    displayHistogram(hist, levels, "Histogram izravnane slike")

    # NOTE: displayImage, loadImage, computeHistogram, displayHistogram so funkcije ki so bile uporabljene na lab vajah in niso nic spremenjene


# Naloga 2:
def draw_circle(canvas, center, radius, color):
    x_center, y_center = center
    for x in range(x_center - radius, x_center + radius + 1):
        for y in range(y_center - radius, y_center + radius + 1):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                canvas[y, x] = color
    return canvas


if __name__ == "__main__":
    canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255  
    center = (50, 50)
    radius = 25
    color = (0, 0, 0)  
    canvas = draw_circle(canvas, center, radius, color)
    plt.imshow(canvas)
    plt.show()


# Naloga 3:
def create_pop_art(iImage, max_dot_radius, background_color, dot_colors):
    width, height = len(iImage[0]), len(iImage)
    oImage = [[background_color for _ in range(width)] for _ in range(height)]
    
    for y in range(0, height, 2 * max_dot_radius):
        for x in range(0, width, 2 * max_dot_radius):
            brightness = iImage[y][x]
            dot_radius = int((1 - brightness / 255) * max_dot_radius)
            dot_color = dot_colors[0]  
            
            if dot_radius > 0:
                for dy in range(-dot_radius, dot_radius + 1):
                    for dx in range(-dot_radius, dot_radius + 1):
                        if dx * dx + dy * dy <= dot_radius * dot_radius:
                            if 0 <= x + dx < width and 0 <= y + dy < height:
                                oImage[y + dy][x + dx] = dot_color
    
    return np.array(oImage, dtype=np.uint8)


if __name__ == "__main__":
    max_dot_radius = 5
    background_color = (255, 255, 255)  
    dot_colors = [(0, 0, 0)]  

    pop_art_image = create_pop_art(equalized_image, max_dot_radius, background_color, dot_colors)
    displayImage(pop_art_image, "Pop art slika")


# Naloga 4:
def create_pop_art2(iImage, max_dot_radius, background_color, dot_colors):
    width, height = len(iImage[0]), len(iImage)
    oImage = [[background_color for _ in range(width)] for _ in range(height)]
    
    for y in range(0, height, 2 * max_dot_radius):
        for x in range(0, width, 2 * max_dot_radius):
            brightness = iImage[y][x]
            dot_radius = int((1 - brightness / 255) * max_dot_radius)
            
            if brightness < 85:
                dot_color = dot_colors[0]
            elif brightness < 170:
                dot_color = dot_colors[1]
            else:
                dot_color = dot_colors[2]
            
            if dot_radius > 0:
                for dy in range(-dot_radius, dot_radius + 1):
                    for dx in range(-dot_radius, dot_radius + 1):
                        if dx * dx + dy * dy <= dot_radius * dot_radius:
                            if 0 <= x + dx < width and 0 <= y + dy < height:
                                oImage[y + dy][x + dx] = dot_color
    
    return np.array(oImage, dtype=np.uint8)


if __name__ == "__main__":
    color_choices_rgb = [
    {"background": (255, 255, 255),
    "dots": [(0, 0, 0), (255, 0, 0), (0, 255, 0)]},
    {"background": (255, 255, 255),
    "dots": [(255 , 0, 0), (0, 0, 255), (255, 255, 0)]},
    {"background": (255, 255, 0),
    "dots": [(255 , 0, 0), (0, 0, 255), (255, 0, 255)]},
    {"background": (255, 192, 203),
    "dots": [(75, 0, 130), (238, 130, 238), (147, 112, 219) ]}]
    
    
    iImage = equalized_image
    max_dot_radius = 5
    
    pop_art_images = []
    for colors in color_choices_rgb:
        pop_art = create_pop_art(iImage, max_dot_radius, colors["background"], colors["dots"])
        pop_art_images.append(pop_art)
    
    combined_image = np.vstack([
        np.hstack([pop_art_images[0], pop_art_images[1]]),
        np.hstack([pop_art_images[2], pop_art_images[3]])
    ])
    
    displayImage(combined_image, "Pop art slike skupaj")