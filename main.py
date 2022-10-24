from glob import glob
import os
import pygame
import numpy as np
import random

from brain import Network, sigmoid
from genome import Genetics

pygame.font.init()

WIDTH = 500
HEIGHT = 800

BIRD_IMGS = []
for i in range(1, 4):
    BIRD_IMGS.append(pygame.transform.scale2x(
        pygame.image.load(os.path.join('assests', f'bird{i}.png'))))

PIPE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join('assests', f'pipe.png')))
BASE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join('assests', f'base.png')))
BG_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join('assests', f'bg.png')))

STAT_FONT = pygame.font.SysFont('comicsans', 30)

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GA - Flappy")


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2

        if d >= 16:
            d = 16

        # Shitty physics
        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_img = pygame.transform.rotate(self.img, self.tilt)
        rect = rotated_img.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_img, rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x) -> None:
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird: Bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y) -> None:
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, gen, pop_size, alive):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render(str(score), 1, (0, 0, 0))
    win.blit(text, (WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (0, 0, 0))
    win.blit(text, (10, 10))

    text = STAT_FONT.render("Pop: " + str(pop_size), 1, (0, 0, 0))
    win.blit(text, (10, 50))

    text = STAT_FONT.render("Alive: " + str(alive), 1, (0, 0, 0))
    win.blit(text, (10, 90))

    for bird in birds:
        bird.draw(win)
    
    base.draw(win)
    pygame.display.update()


def fit(genomes, gen, pop_size):

    nets = []
    birds = []
    ge = []

    for g in genomes:
        nn = Network()
        nn.from_genome(g)
        nets.append(nn)
        birds.append(Bird(230, 350))
        ge.append(g)

    base = Base(730)
    clock = pygame.time.Clock()
    pipes = [Pipe(700)]
    score = 0

    run = True
    while run and len(birds) > 0:
        clock.tick(45)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        pipe_ind = 0
        if len(birds) > 0:
            # determine whether to use the first or second
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()
            output = nets[x].propagate(np.array([bird.y, abs(
                bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)]))
            if output[0] > 0.5:
                bird.jump()

        base.move()

        add_pipe = False
        rem = []
        for pipe in pipes:
            pipe.move()

            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 700 or bird.y < -50:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        draw_window(win, birds, pipes, base, score, gen, pop_size, len(birds))


def run(generations):
    config = [(3, 4, sigmoid), (4, 3, sigmoid), (3, 1, sigmoid)]
    pop_size = 100
    genomes = Genetics.generate_population(pop_size, config)
    for gen in range(generations):
        genomes = Genetics.evolve(fit, genomes, gen, pop_size)


if __name__ == "__main__":
    run(50)
