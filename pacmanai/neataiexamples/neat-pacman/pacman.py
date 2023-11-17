import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites
import random

class Pacman(Entity):
    def __init__(self,node,renderMode,net):
        Entity.__init__(self, node )
        self.renderMode=renderMode
        self.name = PACMAN
        self.color = YELLOW
        self.alive = True
        self.agentDecision = self.neuralAgent#agent decision making function
        self.sprites = None
        self.net = net
        self.observation = [0 for i in range(22)]
        if renderMode:
            self.sprites = PacmanSprites(self)
        
    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.alive = True

    def die(self):
        self.alive = False
        self.direction = STOP

    def update(self, dt,observations):
        self.observation = observations
        if self.renderMode:
            self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt
        direction = self.agentDecision()#this is for direction
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                #self.target = self.getNewTarget(self.direction) #remove this to keep pacman going in sinular direction
                pass

            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()

    #These functions are for classifying agent decision making
    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP
    
    def randomAgent(self):
        if self.target is self.node:
            action = random.choice([UP,DOWN,LEFT,RIGHT])
            return action
        return STOP
    
    def neuralAgent(self):
        if self.target is self.node:
            output = self.net.activate(self.observation)
            action = output.index(max(output))
            action = [LEFT,RIGHT,UP,DOWN][action]
            return action
        return STOP
    
    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None
    
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False