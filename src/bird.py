import numpy as np
import pygame as pg

class Bird():

    bird_type_y = {
        0: 5,
        1: 22,
        2: 95
    }

    def __init__(self, screen_width):
        
        self.x_position = 1350
        self.type = np.random.randint(0,3)
        self.height = 40
        self.y_position = (screen_width // 2) - self.bird_type_y[self.type]
        self.width = 84

    def update(self, speed):
        self.x_position -= speed

    def draw(self, screen):
        pg.draw.rect(screen,"blue", self.rect)

    @property
    def rect(self):
        return pg.Rect(self.x_position, self.y_position, self.width, self.height)

