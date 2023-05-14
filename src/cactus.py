import numpy as np
import pygame as pg


class Cactus():

    cactus_type_width = {
        0: 30,
        1: 64,
        2: 88,
        3: 46,
        4: 86,
        5: 136
    }

    def __init__(self, screen_width):
        
        self.x_position = 1350
        self.type = np.random.randint(0,6)
        self.height = 66 if self.type < 3 else 96
        self.y_position = (screen_width // 2) + 20 if self.type < 3 else (screen_width // 2) - 10
        self.width = self.cactus_type_width[self.type]

    def update(self, speed):
        self.x_position -= speed

    def draw(self, screen):
        pg.draw.rect(screen,"green", self.rect)

    @property
    def rect(self):
        return pg.Rect(self.x_position, self.y_position, self.width, self.height)
