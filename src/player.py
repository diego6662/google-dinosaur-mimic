import pygame as pg
import numpy as np
from brain import NN


class Dinosaur():


    def __init__(self, screen_width,):

        self.x_position = 50
        self.screen_width = screen_width
        self.y_position = self.screen_width // 2
        self.jumping = False
        self.crouching = False
        self.jump_state = 0.0
        self.height = 86
        self.width = 80
        self.alive = True
        self.brain = NN()
        self.color = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
        self.score = None

    def __repr__(self):
        return f"{self.score}"

    def update(self,):
        
        if self.jumping:
            self.y_position = (self.screen_width // 2) - (self.f(self.jump_state))
            self.jump_state += 0.03
            
            if self.jump_state > 1:
                self.jumping = False
                self.jump_state = 0
                self.y_position = self.screen_width // 2

    def f(self, x):
        return (-4 * x * (x - 1)) * 230

    def jump(self):
        self.jumping = True

    def stop_jump(self):
        self.jumping = False
        self.jump_state = 0
        self.y_position = self.screen_width // 2

    def crouch(self):
        self.crouching = True
        self.y_position = (self.screen_width // 2) + 34
        self.width = 110
        self.height = 52

    def stop_crouch(self):
        self.crouching = False
        self.y_position = self.screen_width // 2
        self.width = 80
        self.height = 86

    def is_jumping(self):
        return self.jumping
    
    def die(self, score):
        self.alive = False
        self.score = score

    @property
    def rect(self):
        return pg.Rect(self.x_position, self.y_position, self.width, self.height)

    def draw(self, screen):
        pg.draw.rect(screen,self.color, self.rect)

    def is_alive(self):
        return self.alive

    def compute(self, distance, speed, e_x, e_y, e_w, e_h):
        x_input = np.array([distance, e_x, e_y, e_w, e_h, self.y_position, speed])
        return np.argmax(self.brain.run(x_input))
