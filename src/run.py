import sys
import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from cells import CellGroup
from ghosts import GhostGroup
from walls import WallGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData
import time


# for plotting, we need to know the score at each time step
# we can use the score variable in the game controller to do this
# matplotlib is a library for plotting

DISPLAY = True  # Set to False for no graphics
FRAMERATE = 50
TRAINING = False
DEBUGGING = False
GRAPH = False
GHOSTS4 = False

N_GAMES = 100

if "-t" in sys.argv or "--train" in sys.argv:
    TRAINING = True
    FRAMERATE = 9999
    DISPLAY = False

if "-d" in sys.argv or "--debug" in sys.argv:
    DEBUGGING = True
    TRAINING = True
    FRAMERATE = 50

if "-s" in sys.argv or "--show" in sys.argv:
    FRAMERATE = 9999

if "-g" in sys.argv or "--graph" in sys.argv:
    FRAMERATE = 9999
    DISPLAY = False
    GRAPH = True

if "-4" in sys.argv or "--4ghosts" in sys.argv:
    GHOSTS4 = True

class GameController(object):
    def __init__(self):
        pygame.init()
        self.human = False
        if "-p" in sys.argv or "--play" in sys.argv:
            self.human = True
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(self.human)  # Set True for human player
        self.level = 0
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.gaming = self.human
        self.training = TRAINING
        self.debugging = DEBUGGING
        self.games = 0
        self.terminated = False
        self.pacman_strength = 2
        self.total_score = np.zeros(N_GAMES)
        self.mazedata = MazeData()
        self.frames = 0

    def getState(self):
        # the state is the position of the ghosts and the position of pacman, for a total of 6 values
        # the state is a 6x1 vector
        state = np.zeros(6)
        state[0] = self.pacman.position.x / TILEWIDTH
        state[1] = self.pacman.position.y / TILEHEIGHT
        state[2] = self.ghosts.blinky.position.x / TILEWIDTH
        state[3] = self.ghosts.blinky.position.y / TILEHEIGHT
        state[4] = self.ghosts.pinky.position.x / TILEWIDTH
        state[5] = self.ghosts.pinky.position.y / TILEHEIGHT

        # state[6] = self.pacman.direction
        # state[7] = self.ghosts.blinky.direction
        # state[8] = self.ghosts.pinky.direction

        # print("State: ", state)

        # normalize the state
        state[0] = state[0] / 28
        state[1] = state[1] / 36
        state[2] = state[2] / 28
        state[3] = state[3] / 36
        state[4] = state[4] / 28
        state[5] = state[5] / 36

        # state[6] = abs((state[6] - 2) / 4)
        # state[7] = abs((state[7] - 2) / 4)
        # state[8] = abs((state[8] - 2) / 4)

        return state
    
    def getState2(self):
        # the state is the position of the ghosts and the position of pacman, for a total of 6 values
        # the state is a 6x1 vector
        state = np.zeros(6)
        state[0] = self.pacman.position.x / TILEWIDTH
        state[1] = self.pacman.position.y / TILEHEIGHT
        state[2] = self.ghosts.inky.position.x / TILEWIDTH
        state[3] = self.ghosts.inky.position.y / TILEHEIGHT
        state[4] = self.ghosts.clyde.position.x / TILEWIDTH
        state[5] = self.ghosts.clyde.position.y / TILEHEIGHT

        # state[6] = self.pacman.direction
        # state[7] = self.ghosts.blinky.direction
        # state[8] = self.ghosts.pinky.direction

        # print("State: ", state)

        # normalize the state
        state[0] = state[0] / 28
        state[1] = state[1] / 36
        state[2] = state[2] / 28
        state[3] = state[3] / 36
        state[4] = state[4] / 28
        state[5] = state[5] / 36

        # state[6] = abs((state[6] - 2) / 4)
        # state[7] = abs((state[7] - 2) / 4)
        # state[8] = abs((state[8] - 2) / 4)

        return state

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(
            self.background_norm, self.level % 5)
        self.background_flash = self.mazesprites.constructBackground(
            self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):
        # we stop at N_GAMES games and plot the results, and save the plot as an image, and then change the strength of pacman to 2
        if self.games == N_GAMES and GRAPH:
            plt.plot(self.total_score)
            if self.pacman.strength == 0:
                print("Weak Pacman: ", self.total_score.mean())
            elif self.pacman.strength == 1:
                print("Medium Pacman: ", self.total_score.mean())
            elif self.pacman.strength == 2:
                print("Better Pacman: ", self.total_score.mean())
            else:
                plt.savefig('score.png')
                print("Stronk Pacman: ", self.total_score.mean())
                exit()
            self.pacman_strength += 1
            self.games = 0
            self.total_score = np.zeros(N_GAMES)
        self.games += 1
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(
            self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(
            *self.mazedata.obj.pacmanStart), self.pacman_strength)
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
        self.cells = CellGroup(self.mazedata.obj.name+".txt")
        self.walls = WallGroup(self.mazedata.obj.name+".txt")
        self.ghosts = GhostGroup(
            self.nodes.getStartTempNode(), self.pacman, training=self.training, ghosts4=GHOSTS4)

        self.ghosts.pinky.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))
        if GHOSTS4:
            self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))
            self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(
            *self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(
            self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        if GHOSTS4:
            self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
            self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)

        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

    def startGame_old(self):
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze_nodes.txt")
        self.nodes.setPortalPair((0, 17), (27, 17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12, 14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15, 14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(
            self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(
            self.nodes.getNodeFromTiles(2+11.5, 3+14))
        if GHOSTS4:
            self.ghosts.inky.setStartNode(
                self.nodes.getNodeFromTiles(2+11.5, 0+14))
            self.ghosts.clyde.setStartNode(
                self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        # self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        # self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)

    def update(self):
        if DISPLAY:
            self.clock.tick(FRAMERATE)
        dt = 0.02  # 50 fps
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt, self)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt, self)
        else:
            self.pacman.update(dt, self)

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        self.frames += 1
        if self.frames > 3333:  # 1.1 minutes of play, 3333 frames at 50 fps
            if self.training:
                print("SUPER DUPER FORCING REFRESH")
            if self.score > 10000 and GRAPH:
                self.lives -= 1
                if self.lives <= 0:
                    self.restartGame()
            else:
                print("Score", self.score)
            self.terminated = True
            self.frames = 0
            self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
            self.setBackground()
            self.pause.setPause(
                pauseTime=(0, 3)[self.human], func=self.resetLevel)


        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                # game.ghosts.saveModel()
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=self.human)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            # self.hideEntities()

    def checkPelletEvents(self, dt=0.02):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            # if there are 2/3 pellets left, multiply the score by 2, if there is 1/3 left, multiply the score by 3
            multiply = 1
            if self.pellets.numEaten > 1/3 * len(self.pellets.pelletList):
                multiply = 2
            elif self.pellets.numEaten > 2/3 * len(self.pellets.pelletList):
                multiply = 3

            self.updateScore(pellet.points * multiply)
            if GHOSTS4:
                if self.pellets.numEaten == 30:
                    self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
                if self.pellets.numEaten == 70:
                    self.ghosts.clyde.startNode.allowAccess(
                        LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.pacman.powered_up = True
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                if self.training:
                    self.ghosts.completeTraining(self, dt)
                    self.ghosts.action = None
                    self.pause.setPause(
                        pauseTime=(0, 3)[self.human], func=self.resetLevel)
                else:
                    if self.score > 25000 and GRAPH:
                        self.restartGame()
                    self.pause.setPause(pauseTime=(
                        0, 3)[self.human], func=self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(
                        str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(
                        pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.training:
                            # regen the pellets
                            self.pellets = PelletGroup(
                                self.mazedata.obj.name+".txt")

                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(
                                pauseTime=(0, 3)[self.human], func=self.restartGame)
                        else:
                            self.pause.setPause(
                                pauseTime=(0, 3)[self.human], func=self.resetLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(
                    self.nodes.getNodeFromTiles(9, 20), self.level)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(
                    self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        if not self.gaming:
            self.resetLevel()
            self.setBackground()
            self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
            return
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.frames = 0
        self.terminated = True
        self.lives = 5
        self.level = 0
        self.pause.paused = self.human
        self.fruit = None
        self.startGame()
        self.total_score[self.games - 1] = self.score
        print("Pacman strength: ", self.pacman.strength, "Game: ", self.games-1,
              "Score: ", self.score)
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []

    def resetLevel(self):
        if self.training:
            self.lives = 5
            # regen pellets
            self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
        self.frames = 0
        self.terminated = True
        self.flashBG = False
        self.pause.paused = self.human
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        # self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        self.cells.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        if DISPLAY:
            pygame.display.update()


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()
