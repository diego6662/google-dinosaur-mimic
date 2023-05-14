from copy import deepcopy

import numpy as np
import pygame as pg

from bird import Bird
from cactus import Cactus
from player import Dinosaur
import argparse

pg.font.init()


class Environment:
    def __init__(self, population, n_top):
        self.height = 1280
        self.width = 720
        self.clock = pg.time.Clock()
        self.running = True
        self.screen = pg.display.set_mode((self.height, self.width))
        self.dinosaurs = [Dinosaur(self.width) for _ in range(population)]
        self.dead_dinosaurs = []
        self.cactus = []
        self.birds = []
        self.speed = 12
        self.score = 0
        self.font = pg.font.SysFont("Arial", 26)
        self.generation_number = 0
        self.population = population
        self.n_top = n_top
        self.last_mean_score = 0
        self.best_score = 0
        self.best_dino = None
        self.best_generation = 0

    def run(self):
        start = pg.time.get_ticks()
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
            self.screen.fill("black")
            self.info()
            now = pg.time.get_ticks()
            time = np.random.uniform(0.6, 1.2)

            if now - start > time * 1000:
                self.spawn_enemy()
                start = now

            pg.draw.line(
                self.screen,
                "white",
                (0, (self.width // 2) + 86),
                (self.height, (self.width // 2) + 86),
            )
            self.update()
            self.draw()
            self.check_collisions()
            for d in self.dinosaurs:
                action = self.brain_step(d)
                if action == 0:
                    d.jump()
                if action == 1:
                    if d.is_jumping():
                        d.stop_jump()

                    d.crouch()
                else:
                    d.stop_crouch()

            self.clean_passed_enemies()
            self.clean_dead_dinosaurs()
            pg.display.update()
            # limits FPS to 60
            self.clock.tick(60)
            self.speed += 0.005
            self.score += 1
            if len(self.dinosaurs) == 0:
                self.screen.fill("black")
                self.draw()
                self.info()
                pg.display.update()
                # input('continue?')
                self.next_gen()

    def info(self):
        text_surface = self.font.render(f"Score:{self.score}", True, "white")
        self.screen.blit(text_surface, (0, 0))
        text_surface = self.font.render(
            f"Generation:{self.generation_number}", True, "white"
        )
        self.screen.blit(text_surface, (0, 25))
        text_surface = self.font.render(f"Alive:{len(self.dinosaurs)}", True, "white")
        self.screen.blit(text_surface, (0, 50))
        text_surface = self.font.render(
            f"Last Mean Score:{self.last_mean_score}", True, "white"
        )
        self.screen.blit(text_surface, (400, 0))
        text_surface = self.font.render(f"Best Score:{self.best_score}", True, "white")
        self.screen.blit(text_surface, (400, 25))
        text_surface = self.font.render(
            f"Best Generation:{self.best_generation}", True, "white"
        )
        self.screen.blit(text_surface, (400, 50))

    def update(self):
        for d in self.dinosaurs:
            d.update()
        for e in self.cactus + self.birds:
            e.update(self.speed)

    def keyPressed(self, inputKey):
        keysPressed = pg.key.get_pressed()
        if keysPressed[inputKey]:
            return True
        else:
            return False

    def spawn_enemy(self):
        if np.random.randint(0, 10) == 0:
            self.birds.append(Bird(self.width))
        else:
            self.cactus.append(Cactus(self.width))

    def check_collisions(self):
        for d in self.dinosaurs:
            for e in self.cactus + self.birds:
                if pg.Rect.colliderect(d.rect, e.rect):
                    d.die(self.score)
                    break

    def draw(self):
        for e in self.dinosaurs + self.cactus + self.birds:
            e.draw(self.screen)

    def clean_passed_enemies(self):
        valid_cactus = filter(lambda c: c.x_position >= -100, self.cactus)
        valid_birds = filter(lambda b: b.x_position >= -100, self.birds)
        self.cactus = [*valid_cactus]
        self.birds = [*valid_birds]

    def clean_dead_dinosaurs(self):
        dead_dinosaurs = filter(lambda d: not d.is_alive(), self.dinosaurs)
        self.dead_dinosaurs += [*dead_dinosaurs]
        self.dinosaurs = [*filter(lambda d: d.is_alive(), self.dinosaurs)]

    def brain_step(self, dinosaur):
        enemies = self.cactus + self.birds
        if not enemies:
            return
        nearest_enemy_ind = self.close_enemy(enemies, dinosaur)
        near_enemy = enemies[nearest_enemy_ind]
        distance = np.abs(near_enemy.x_position - dinosaur.x_position)
        return dinosaur.compute(
            distance,
            self.speed,
            near_enemy.x_position,
            near_enemy.y_position,
            near_enemy.width,
            near_enemy.height,
        )

    def close_enemy(self, enemies, dinosaur):
        dist_enemies = map(
            lambda e: np.inf
            if e.x_position < 0
            else np.abs(e.x_position - dinosaur.x_position),
            enemies,
        )
        return np.argmin([*dist_enemies])

    def next_gen(self):
        new_best_dinosaurs = self.get_n_best()
        if new_best_dinosaurs[0].score > self.best_score:
            self.best_score = new_best_dinosaurs[0].score
            self.best_dino = new_best_dinosaurs[0]
            self.best_generation = self.generation_number

        new_gen = []
        for _ in range(self.population):
            if np.random.rand() <= 0.1:
                d = Dinosaur(self.width)
                d.brain.model.layers = deepcopy(self.best_dino.brain.model.layers)
                new_gen.append(d)
            elif np.random.rand() <= 0.5:
                d = Dinosaur(self.width)
                d.brain.model.layers = deepcopy(
                    new_best_dinosaurs[0].brain.model.layers
                )
                new_gen.append(d)
            else:
                d = Dinosaur(self.width)
                d.brain.model.layers = deepcopy(
                    new_best_dinosaurs[
                        np.random.randint(0, self.n_top)
                    ].brain.model.layers
                )
                new_gen.append(d)

        for d in new_gen:
            if np.random.rand() <= 0.4:
                d.brain.cross(self.best_dino)

            if np.random.rand() <= 0.8:
                d.brain.mutate()

            d.alive = True
        self.dinosaurs = new_gen[: self.population]
        self.last_mean_score = np.mean([*map(lambda d: d.score, self.dead_dinosaurs)])
        self.dead_dinosaurs = []
        self.birds = []
        self.cactus = []
        self.generation_number += 1
        self.alive = self.population
        self.speed = 12
        self.score = 0

    def get_n_best(self):
        best_dinosaurs = [*reversed(sorted(self.dead_dinosaurs, key=lambda d: d.score))]
        return best_dinosaurs[: self.n_top]



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--population', type=int, help='Population number')
parser.add_argument('-t', '--top', type=int, help='Number of top dinosaur to reproduce')

def main(population: int, n_top: int):
    env = Environment(population, n_top)
    env.run()

if __name__ == "__main__":
    args = parser.parse_args()
    n_top = args.top if args.top else 3
    population = args.population if args.population else 400
    main(population, n_top)
