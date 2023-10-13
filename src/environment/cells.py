# The cells module contains the Cell class, which is used to represent the cells
# in the maze. The Cell class contains the following methods:
# We need this because we need to know the neighbors of the closest pellet
# And having only nodes is not enough
import pygame
from environment.vector import Vector2
from constants import *
import numpy as np
import os


class Cell(object):
    def __init__(self, row, column):
        self.position = Vector2(column*TILEWIDTH, row*TILEHEIGHT)
        self.color = RED
        self.x = column
        self.y = row
        self.radius = int(15 * TILEWIDTH / 16)
        self.collideRadius = 1 * TILEWIDTH / 16
        self.neighbors = {UP: None, DOWN: None,
                          LEFT: None, RIGHT: None}
        self.visible = False

    def render(self, screen):
        if self.visible:
            adjust = Vector2(TILEWIDTH, TILEHEIGHT) / 2
            p = self.position + adjust
            pygame.draw.circle(screen, self.color, p.asInt(), self.radius, 1)


class CellGroup(object):
    def __init__(self, cellfile):
        # for performance reasons, we need to be able to access cells by their coordinates
        self.cellDict = {}
        self.createCellList(cellfile)

    def createCellList(self, pelletfile):
        pelletfile = os.path.join(os.path.dirname(__file__), pelletfile)
        data = self.readCellfile(pelletfile)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in ['.', '+', 'P', 'p', '-', '|', 'n']:
                    self.cellDict[(col, row)] = Cell(row, col)

        # set neighbors
        for cell in self.cellDict.values():
            cell.neighbors[UP] = self.getCell(cell.x, cell.y-1)
            cell.neighbors[DOWN] = self.getCell(cell.x, cell.y+1)
            cell.neighbors[LEFT] = self.getCell(cell.x-1, cell.y)
            cell.neighbors[RIGHT] = self.getCell(cell.x+1, cell.y)

        # set neighbors for tunnel cells
        for cell in self.cellDict.values():
            if cell.x == 0:
                cell.neighbors[LEFT] = self.getCell(NCOLS-1, cell.y)
            elif cell.x == NCOLS-1:
                cell.neighbors[RIGHT] = self.getCell(0, cell.y)

    def getCell(self, x, y):
        if (x, y) in self.cellDict:
            return self.cellDict[(x, y)]
        else:
            return None

    def readCellfile(self, textfile):
        return np.loadtxt(textfile, dtype='<U1')

    def isEmpty(self):
        if len(self.pelletList) == 0:
            return True
        return False

    def render(self, screen):
        for cell in self.cellDict.values():
            cell.render(screen)

    def getClosestCell(self, position):
        min_dist = 100000
        closest_cell = None
        for cell in self.cellDict.values():
            dist = (cell.position - position).magnitudeSquared()
            if dist < min_dist:
                min_dist = dist
                closest_cell = cell
        return closest_cell

    def getBFSDistance(self, cell1, cell2):
        # BFS
        visited = set()
        queue = []
        queue.append((cell1, 0))
        while queue:
            cell, distance = queue.pop(0)
            if cell == cell2:
                return distance
            if cell in visited:
                continue
            visited.add(cell)
            for neighbor in cell.neighbors.values():
                if neighbor:
                    queue.append((neighbor, distance + 1))
        # throw error
        return -1  # unreachable
