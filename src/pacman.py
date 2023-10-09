import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites
import random

MANA = 32
COOLDOWN = 5

class Pacman(Entity):
    def __init__(self, node, strength=0):
        Entity.__init__(self, node)
        self.name = PACMAN
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.dead = False
        self.sprites = PacmanSprites(self)
        self.frames = 0
        self.closest_pellet = None
        self.cell = None
        self.closest_cell = None
        self.changed_target = True
        self.strength = strength
        self.dangerDistance = 5
        self.mana = MANA
        self.mana_delay = 0

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()
        # pathfinding
        self.changed_target = True
        self.closest_cell = None
        self.cell = None
        self.closest_pellet = None

    def die(self):
        self.alive = False
        self.dead = True
        self.direction = STOP

    def update(self, dt, game):
        # if collided with closest pellet, eat it
        if self.closest_pellet is not None:
            if self.collideCheck(self.closest_pellet):
                self.closest_pellet = None
                self.changed_target = True

        if self.closest_cell is not None:
            if self.collideCheck(self.closest_cell):
                self.cell = self.closest_cell

        if self.frames > 3 and not self.dangerDistance > 5:  # every 3 seconds we force change of strategy
            self.closest_cell = None
            self.closest_pellet = None
            self.changed_target = True
            self.frames = 0

        if self.mana == 0:
            self.mana_delay += dt
            if self.mana_delay > COOLDOWN:
                self.mana = MANA
                self.mana_delay = 0

        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt*1
        self.frames += dt
        direction = self.getValidKey(game)

        if self.oppositeDirection(direction) and not self.changed_target and not self.dangerDistance > 5:
            # random direction
            directions = [UP, DOWN, LEFT, RIGHT]
            directions.remove(direction)
            direction = random.choice(directions)
        self.dangerDistance = 5
        self.changed_target = False
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else:
            if self.oppositeDirection(direction):
                self.reverseDirection()

    def getValidKey(self, game):
        if self.strength == 0:
            return self.getValidKeyWeak(game)
        elif self.strength == 1:
            return self.getValidKeyMedium(game)
        elif self.strength == 2:
            return self.getValidKeyBetter(game)
        elif self.strength == 3:
            return self.getValidKeyStrong(game)

    def getValidKeyMedium(self, game):
        # based on the location of the nearest pellet, return the direction to move
        # shuffle pellets
        if self.closest_pellet is None:
            # print("Finding closest pellet")
            pellets = game.pellets.pelletList
            random.shuffle(pellets)
            if len(pellets) > 0:
                closest_pellet = pellets[0]
            else:
                return STOP

            d = self.position - closest_pellet.position
            closest_distance = d.magnitudeSquared()
            for pellet in pellets:
                # We get the closest pellet to the pacman
                d = self.position - pellet.position
                dSquared = d.magnitudeSquared()
                if dSquared < closest_distance:
                    closest_distance = dSquared
                    closest_pellet = pellet

            x = closest_pellet.position.x / TILEWIDTH
            y = closest_pellet.position.y / TILEHEIGHT

            # We loop through all the neighbors of the pacman cell to get the closest one to the pellet
            closest_pellet.color = RED
            self.closest_pellet = closest_pellet

        # closest cell to pacman is the pacman cell
        closest_cell = None
        pacman_cell_distance = 100000000
        for cell in game.cells.cellDict.values():
            d = cell.position - self.position
            dSquared = d.magnitudeSquared()
            if dSquared < pacman_cell_distance:
                pacman_cell_distance = dSquared
                closest_cell = cell

        self.closest_cell = closest_cell

        if self.cell is None:
            self.cell = closest_cell

        closest_distance = 100000000

        for neighbor in self.cell.neighbors.values():
            if neighbor is not None:
                d = neighbor.position - self.closest_pellet.position
                dSquared = d.magnitudeSquared()
                if dSquared < closest_distance:
                    closest_distance = dSquared
                    t = neighbor

        # all other cells should be invisible
        for cell in game.cells.cellDict.values():
            cell.visible = False
        t.visible = True

        # Based on the position of the target, we return the direction
        # print(t.position.x, t.position.y,
        #       self.position.x, self.position.y)
        if abs(t.position.y - self.position.y) < abs(t.position.x - self.position.x):
            if t.position.x > self.position.x:
                dir = RIGHT
            elif t.position.x < self.position.x:
                dir = LEFT
            else:
                dir = STOP
        else:
            if t.position.y < self.position.y:
                dir = UP
            elif t.position.y > self.position.y:
                dir = DOWN
            else:
                dir = STOP

        return dir

    def getValidKeyWeak(self, game):
        # based on the location of the nearest pellet, return the direction to move
        # shuffle pellets
        if self.closest_pellet is None:
            # print("Finding closest pellet")
            pellets = game.pellets.pelletList

            random.shuffle(pellets)
            closest_pellet = pellets[0]
            # We get the furthest pellet from the ghosts average
            g1 = game.ghosts.pinky
            g2 = game.ghosts.blinky
            average = (g1.position + g2.position) / 2
            d = average - closest_pellet.position
            furthest_distance = d.magnitudeSquared()
            for pellet in pellets:
                # We get the furthest pellet from the ghosts average
                d = average - pellet.position
                dSquared = d.magnitudeSquared()
                if dSquared > furthest_distance:
                    furthest_distance = dSquared
                    closest_pellet = pellet

            x = closest_pellet.position.x / TILEWIDTH
            y = closest_pellet.position.y / TILEHEIGHT

            # We loop through all the neighbors of the pacman cell to get the closest one to the pellet
            closest_pellet.color = RED
            self.closest_pellet = closest_pellet

        # closest cell to pacman is the pacman cell
        closest_cell = None
        pacman_cell_distance = 100000000
        for cell in game.cells.cellDict.values():
            d = cell.position - self.position
            dSquared = d.magnitudeSquared()
            if dSquared < pacman_cell_distance:
                pacman_cell_distance = dSquared
                closest_cell = cell

        self.closest_cell = closest_cell

        if self.cell is None:
            self.cell = closest_cell

        closest_distance = 100000000

        for neighbor in self.cell.neighbors.values():
            if neighbor is not None:
                d = neighbor.position - self.closest_pellet.position
                dSquared = d.magnitudeSquared()
                if dSquared < closest_distance:
                    closest_distance = dSquared
                    t = neighbor

        # all other cells should be invisible
        for cell in game.cells.cellDict.values():
            cell.visible = False
        t.visible = True

        # Based on the position of the target, we return the direction
        # print(t.position.x, t.position.y,
        #       self.position.x, self.position.y)
        if abs(t.position.y - self.position.y) < abs(t.position.x - self.position.x):
            if t.position.x > self.position.x:
                dir = RIGHT
            elif t.position.x < self.position.x:
                dir = LEFT
            else:
                dir = STOP
        else:
            if t.position.y < self.position.y:
                dir = UP
            elif t.position.y > self.position.y:
                dir = DOWN
            else:
                dir = STOP

        return dir

    def getValidKeyBetter(self, game):
        # based on the location of the nearest pellet, return the direction to move
        # shuffle pellets

        # if the distance between us the average of the ghosts is less than 5, we run away
        g1 = game.ghosts.pinky
        g2 = game.ghosts.blinky
        g1_danger = False
        g2_danger = False

        if g1.cell and g2.cell:
            g1_danger = game.cells.getBFSDistance(
                g1.cell, self.cell) < self.dangerDistance and g1.mode.current != FREIGHT
            g2_danger = game.cells.getBFSDistance(
                g2.cell, self.cell) < self.dangerDistance and g1.mode.current != FREIGHT
        if (g1_danger or g2_danger) and self.mana > 0:
            self.mana -= 1
            self.dangerDistance = 10
            self.closest_cell = None
            return self.getValidKeyEscape(game)
        else:
            return self.getValidKeyMedium(game)

    def getValidKeyStrong(self, game):
        # based on the location of the nearest pellet, return the direction to move
        # shuffle pellets

        # if the distance between us the average of the ghosts is less than 5, we run away
        g1 = game.ghosts.pinky
        g2 = game.ghosts.blinky
        g1_danger = False
        g2_danger = False

        if g1.cell and g2.cell:
            g1_danger = game.cells.getBFSDistance(
                g1.cell, self.cell) < self.dangerDistance and g1.mode.current != FREIGHT
            g2_danger = game.cells.getBFSDistance(
                g2.cell, self.cell) < self.dangerDistance and g1.mode.current != FREIGHT
        if g1_danger or g2_danger:
            self.dangerDistance = 10
            self.closest_cell = None
            return self.getValidKeyEscape(game)
        else:
            return self.getValidKeyMedium(game)

    def getValidKeyEscape(self, game):
        # based on the average location of the ghosts, return the furthest direction to move
        g1 = game.ghosts.pinky
        g2 = game.ghosts.blinky
        g1_closer = (g1.position - self.position).magnitudeSquared() < (
            g2.position - self.position).magnitudeSquared()
        enemy_position = g1.position if g1_closer else g2.position

        # closest cell to pacman is the pacman cell
        closest_cell = None
        pacman_cell_distance = 100000000
        for cell in game.cells.cellDict.values():
            d = cell.position - self.position
            dSquared = d.magnitudeSquared()
            if dSquared < pacman_cell_distance:
                pacman_cell_distance = dSquared
                closest_cell = cell

        self.closest_cell = closest_cell
        closest_cell.color = GREEN

        maximum_distance = 0

        for neighbor in closest_cell.neighbors.values():
            if neighbor is not None:
                d = neighbor.position - enemy_position
                dSquared = d.magnitudeSquared()
                if dSquared > maximum_distance:
                    maximum_distance = dSquared
                    t = neighbor

        # all other cells should be invisible
        for cell in game.cells.cellDict.values():
            cell.visible = False
        t.visible = True

        # Based on the position of the target, we return the direction
        # print(t.position.x, t.position.y,
        #       self.position.x, self.position.y)
        if abs(t.position.y - self.position.y) < abs(t.position.x - self.position.x):
            if t.position.x > self.position.x:
                dir = RIGHT
            elif t.position.x < self.position.x:
                dir = LEFT
            else:
                dir = STOP
        else:
            if t.position.y < self.position.y:
                dir = UP
            elif t.position.y > self.position.y:
                dir = DOWN
            else:
                dir = STOP

        return dir

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None

    def collideGhost(self, ghost):
        return self.collideCheckCustom(ghost, 3)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False

    def collideCheckCustom(self, other, cModif):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius*cModif+other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
