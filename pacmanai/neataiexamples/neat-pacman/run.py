import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup,Node
from pellets import PelletGroup
from ghosts import GhostGroup,Ghost
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites,MazeSprites
from mazedata import MazeData
from collections import namedtuple
import os
import neat
import pickle

#https://pacmancode.com/
#https://ieeexplore.ieee.org/abstract/document/6615002

Observation = namedtuple("Observations",["PillsEaten", "PowerPillDuration",
                                         "PillDirLeft", "PillDirRight", "PillDirUp", "PillDirDown",
                                         "GhostDirLeft", "GhostDirRight","GhostDirUp","GhostDirDown",
                                         "ScaredGhostLeft","ScaredGhostRight","ScaredGhostUp","ScaredGhostDown",
                                         "SafeLeft","SafeRight","SafeUp","SafeDown",
                                         "MovingLeft","MovingRight","MovingUp","MovingDown"])


class GameController(object):
    def __init__(self,numGames,genomes,nets):
        self.games = dict()
        self.genomes = genomes
        self.nets = nets
        self.numGames = numGames
        self.background = None
        self.pause = Pause(False)
        self.level = 0
        self.lives = 1
        self.score = 0
        self.clock = None
        self.textgroup = None
        self.mazedata = MazeData()
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.clock = pygame.time.Clock()
        self.lifesprites = LifeSprites(self.lives)
        self.textgroup = TextGroup()
        self.dividend = 5 #value to mod by to keep game running semi-smooth
            

    def startGame(self):
        self.mazedata.loadMaze(self.level)
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazesprites = MazeSprites(self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")

        for i in range(self.numGames):
            self.games[i] = Game(self.mazedata,None,False,self.genomes[i],self.nets[i])
        self.games[0] = Game(self.mazedata,self.screen,True,self.genomes[0],self.nets[0])

        for key,game in self.games.items():
            game.startGame(self.nodes,self.mazesprites)


    def restartGame(self):
        for key,game in self.games.items():
            game.restartGame()

    def resetLevel(self):
        for key,game in self.games.items():
            game.resetLevel()

    def nextLevel(self):        
        for key,game in self.games.items():
            game.nextLevel()

    def update(self):
        dt = self.clock.tick(30) / 1000.0 #average delta time is .034

        removeList = []
        count = 0
        for key,game in self.games.items():
            count += 1
            gameover,gamewon = game.step(dt)
            if gameover or gamewon:
                removeList.append(key)
            if count>=self.dividend:
                break
        self.checkEvents()

        for i in removeList:
            #print(f"Number of games running: {len(self.games)-1}, Score: {self.games[i].score}")
            self.games.pop(i)
            
        if len(self.games) > 0 and 0 in self.games:
            self.games[0].render()
            return True
        elif len(self.games) == 0:
            return False
        else:
            return True

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    """
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()
                    """
                    print("spacebar pressed")


class Game(object):
    def __init__(self,mazeData,screen,renderMode,genome,net):

        self.pillTimer = 0
        self.pillActive = False
        self.pause = Pause(False)
        self.level = 0
        self.lives = 1
        self.score = 0
        self.mazedata = mazeData
        self.lifesprites = None
        self.textgroup = None
        self.gameOver = False
        self.gameWon = False
        self.renderMode = renderMode
        self.genome = genome
        self.net = net
        self.count = 0
        if self.renderMode:
            self.screen = screen
            self.textgroup = TextGroup()
            self.lifesprites = LifeSprites(self.lives)

    def startGame(self,nodes:NodeGroup,mazesprites:MazeSprites):
        
        self.nodes = nodes
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart),self.renderMode,self.net)
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman,self.renderMode)
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)
        if self.renderMode:
            self.mazesprites = mazesprites
            self.setBackground()
            self.background = self.mazesprites.constructBackground(self.background, self.level%5)

    def restartGame(self):
        self.lives = 1
        self.level = 0
        self.pause.paused = False
        self.startGame()
        self.score = 0
        self.gameOver = False
        self.gameWon = False
        if self.renderMode:
            self.textgroup.updateScore(self.score)
            self.textgroup.updateLevel(self.level)
            #self.textgroup.showText(READYTXT)
            self.lifesprites.resetLives(self.lives)
            self.textgroup.hideText()

    def resetLevel(self):
        self.pause.paused = False
        self.pacman.reset()
        self.ghosts.reset()

    def nextLevel(self):        
        self.level += 1
        self.pause.paused = False
        self.startGame()
        if self.renderMode:
            self.textgroup.updateLevel(self.level)
            self.showEntities()

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def step(self,dt:float) -> tuple[bool,bool]:
        """
        returns two bools first being gameover second gamewon
        """
        if self.gameOver or self.gameWon:
            return self.gameOver,self.gameWon
        
        if self.renderMode:
            self.textgroup.update(dt)

        self.count += 1
        #self.genome.fitness += -1
        observations = [0 for i in range(22)]
        if self.pacman.target is self.pacman.node:
            observations = self.makeObservation2()
        self.pellets.update(dt)
        if not self.pause.paused:
            self.pillTimer += dt
            if self.pillTimer >= 7:#check mode controller for timer
                self.pillActive = False
            self.pacman.update(dt,observations)
            #self.ghosts.update(dt)
            self.checkPelletEvents()
            #self.checkGhostEvents()
        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        if self.renderMode:
            self.render()
        if self.count>=500:
            self.gameOver = True
            #print("game over")
        return False,False

    def updateScore(self, points):
        self.score += points
        if self.renderMode:
            self.textgroup.updateScore(self.score)

    def makeObservation(self):
        pillsEaten = (self.pellets.totalPellets - len(self.pellets.pelletList))/self.pellets.totalPellets
        powerpillLeft = 0
        if self.pillActive:
            powerpillLeft = (7-self.pillTimer)/7
        else:
            powerpillLeft = 0
        #max path length using shortest route is 54 counted?
        MAXPATHLENGTH = 54
        pillDistLeft = 0
        if self.pacman.node.neighbors[LEFT] is not None:
            dist = self.nearestPelletDistance(self.pacman.node.neighbors[LEFT])
            pillDistLeft = (MAXPATHLENGTH-dist)/MAXPATHLENGTH
        pillDistRight= 0
        if self.pacman.node.neighbors[RIGHT] is not None:
            dist = self.nearestPelletDistance(self.pacman.node.neighbors[RIGHT])
            pillDistRight = (MAXPATHLENGTH-dist)/MAXPATHLENGTH
        pillDistUp = 0
        if self.pacman.node.neighbors[UP] is not None:
            dist = self.nearestPelletDistance(self.pacman.node.neighbors[UP])
            pillDistUp = (MAXPATHLENGTH-dist)/MAXPATHLENGTH
        pillDistDown = 0
        if self.pacman.node.neighbors[DOWN] is not None:
            dist = self.nearestPelletDistance(self.pacman.node.neighbors[DOWN])
            pillDistDown = (MAXPATHLENGTH-dist)/MAXPATHLENGTH

        ghostSpeed = 1#relative to pacman
        ghostInputLeft = 0
        if self.pacman.node.neighbors[LEFT] is not None:
            intersectionDist, IntersectionNode = self.nearestIntersection(self.pacman.node.neighbors[LEFT])
            ghostDistToIntersection = self.nearestGhostDistToNode(IntersectionNode)
            ghostInputLeft = (MAXPATHLENGTH+intersectionDist*ghostSpeed-ghostDistToIntersection)/MAXPATHLENGTH
        ghostInputRight = 0
        if self.pacman.node.neighbors[RIGHT] is not None:
            intersectionDist, IntersectionNode = self.nearestIntersection(self.pacman.node.neighbors[RIGHT])
            ghostDistToIntersection = self.nearestGhostDistToNode(IntersectionNode)
            ghostInputRight = (MAXPATHLENGTH+intersectionDist*ghostSpeed-ghostDistToIntersection)/MAXPATHLENGTH
        ghostInputUp = 0
        if self.pacman.node.neighbors[UP] is not None:
            intersectionDist, IntersectionNode = self.nearestIntersection(self.pacman.node.neighbors[UP])
            ghostDistToIntersection = self.nearestGhostDistToNode(IntersectionNode)
            ghostInputUp = (MAXPATHLENGTH+intersectionDist*ghostSpeed-ghostDistToIntersection)/MAXPATHLENGTH
        ghostInputDown = 0
        if self.pacman.node.neighbors[DOWN] is not None:
            intersectionDist, IntersectionNode = self.nearestIntersection(self.pacman.node.neighbors[DOWN])
            ghostDistToIntersection = self.nearestGhostDistToNode(IntersectionNode)
            ghostInputDown = (MAXPATHLENGTH+intersectionDist*ghostSpeed-ghostDistToIntersection)/MAXPATHLENGTH
        
        scaredGhostDistLeft = 0
        if self.pacman.node.neighbors[LEFT] is not None:
            scaredGhostDistLeft = self.nearestScaredGhost(self.pacman.node.neighbors[LEFT])
            scaredGhostDistLeft = (MAXPATHLENGTH-scaredGhostDistLeft)/MAXPATHLENGTH
        scaredGhostDistRight = 0
        if self.pacman.node.neighbors[RIGHT] is not None:
            scaredGhostDistRight = self.nearestScaredGhost(self.pacman.node.neighbors[RIGHT])
            scaredGhostDistRight = (MAXPATHLENGTH-scaredGhostDistRight)/MAXPATHLENGTH
        scaredGhostDistUp = 0
        if self.pacman.node.neighbors[UP] is not None:
            scaredGhostDistUp = self.nearestScaredGhost(self.pacman.node.neighbors[UP])
            scaredGhostDistUp = (MAXPATHLENGTH-scaredGhostDistUp)/MAXPATHLENGTH
        scaredGhostDistDown = 0
        if self.pacman.node.neighbors[DOWN] is not None:
            scaredGhostDistDown = self.nearestScaredGhost(self.pacman.node.neighbors[DOWN])
            scaredGhostDistDown = (MAXPATHLENGTH-scaredGhostDistDown)/MAXPATHLENGTH

        #entrapment
        pacmanSafeRoutes,routesAvailable = self.getIntersectionsSafe()
        leftSafeRoutes = 0
        if self.pacman.node.neighbors[LEFT] is not None and routesAvailable:
            leftSafeRoutes = self.nearestScaredGhost(self.pacman.node.neighbors[LEFT])
            leftSafeRoutes = (pacmanSafeRoutes-leftSafeRoutes)/pacmanSafeRoutes
        rightSafeRoutes = 0
        if self.pacman.node.neighbors[RIGHT] is not None and routesAvailable:
            rightSafeRoutes = self.nearestScaredGhost(self.pacman.node.neighbors[RIGHT])
            rightSafeRoutes = (pacmanSafeRoutes-rightSafeRoutes)/pacmanSafeRoutes
        upSafeRoutes = 0
        if self.pacman.node.neighbors[UP] is not None and routesAvailable:
            upSafeRoutes = self.nearestScaredGhost(self.pacman.node.neighbors[UP])
            upSafeRoutes = (pacmanSafeRoutes-upSafeRoutes)/pacmanSafeRoutes
        downSafeRoutes = 0
        if self.pacman.node.neighbors[DOWN] is not None and routesAvailable:
            downSafeRoutes = self.nearestScaredGhost(self.pacman.node.neighbors[DOWN])
            downSafeRoutes = (pacmanSafeRoutes-downSafeRoutes)/pacmanSafeRoutes

        movingLeft = int(self.pacman.direction == LEFT)
        movingRight = int(self.pacman.direction == RIGHT)
        movingUp = int(self.pacman.direction == UP)
        movingDown = int(self.pacman.direction == DOWN)

        return [pillsEaten,powerpillLeft,
                pillDistLeft,pillDistRight,pillDistUp,pillDistDown,
                ghostInputLeft,ghostInputRight,ghostInputUp,ghostInputDown,
                scaredGhostDistLeft,scaredGhostDistRight,scaredGhostDistUp,scaredGhostDistDown,
                leftSafeRoutes,rightSafeRoutes,upSafeRoutes,downSafeRoutes,
                movingLeft,movingRight,movingUp,movingDown]
    
    def makeObservation2(self):
        pillactive = int(self.pillActive)
        ghostleftdist = self.GhostDistance(LEFT)
        ghostrightdist = self.GhostDistance(RIGHT)
        ghostupdist = self.GhostDistance(UP)
        ghostdowndist = self.GhostDistance(DOWN)
        wallLeft = self.isWallNext(LEFT)
        wallRight = self.isWallNext(RIGHT)
        wallUp = self.isWallNext(UP)
        wallDown = self.isWallNext(DOWN)
        dotDistLeft = self.dotDistance(LEFT)
        dotDistRight = self.dotDistance(RIGHT)
        dotDistUp = self.dotDistance(UP)
        dotDistDown = self.dotDistance(DOWN)
        return [pillactive,ghostleftdist,ghostrightdist,ghostupdist,ghostdowndist,wallLeft,wallRight,wallUp,wallDown,dotDistLeft,dotDistRight,dotDistUp,dotDistDown]


    def GhostDistance(self,direction):
        if self.pacman.node.neighbors[direction] is None:
            return 0
        current = self.pacman.node
        count = 0
        while current is not None:
            count += 1
            current = current.neighbors[direction]
            if current and self.ghosts.containNode(current):
                break
        return 1/count
    
    def isWallNext(self,direction):
        return int(self.pacman.node.neighbors[direction] is None)

    def dotDistance(self,direction):
        if self.pacman.node.neighbors[direction] is None:
            return 0
        current = self.pacman.node
        count = 0
        while current is not None:
            count += 1
            current = current.neighbors[direction]
            if current and current.position in self.pellets.pelletCoords:
                break
        return 1/count


    def getIntersectionsSafe(self):
        nodesCheck = set()
        queue:list[Node] = []
        intersectionNodeList = []
        pacmanDist = []
        queue.append(self.pacman.node)
        nodesCheck.add(self.pacman.node.position)
        self.pacman.node.dist = 0
        count = 0#default amount
        while len(queue)>0:
            temp:Node = queue.pop(0)
            if temp.isIntersection:
                intersectionNodeList.append(temp)
                pacmanDist.append(temp.dist)
            for key,value in temp.neighbors.items():
                if value and value.position not in nodesCheck:
                    queue.append(value)
                    nodesCheck.add(value.position)
                    value.dist = temp.dist+1
                    count = max(count,value.dist)
            if count >= 10:#break loop after searching for 10 distances away from pacman position
                queue = []
        finalList = []
        for i,node in enumerate(intersectionNodeList):#check ghost distances
            node.dist = 0
            nodesCheck = set()
            queue:list[Node] = []
            queue.append(node)
            nodesCheck.add(node.position)
            while len(queue)>0:
                temp:Node = queue.pop(0)
                if self.ghosts.containNode(temp):
                    if temp.dist > pacmanDist[i]:
                        finalList.append(node)
                    break
                for key,value in temp.neighbors.items():
                    if value and value.position not in nodesCheck:
                        queue.append(value)
                        nodesCheck.add(value.position)
                        value.dist = temp.dist+1
        pacmanSafeIntersections = len(finalList)
        return pacmanSafeIntersections, (pacmanSafeIntersections != 0)

    def getIntersectionNodeSafe(self,start:Node):
        nodesCheck = set()
        queue:list[Node] = []
        intersectionNodeList = []
        pacmanDist = []
        queue.append(start)
        nodesCheck.add(start.position)
        start.dist = 0
        count = 0#default amount
        while len(queue)>0:
            temp:Node = queue.pop(0)
            if temp.isIntersection:
                intersectionNodeList.append(temp)
                pacmanDist.append(temp.dist)
            for key,value in temp.neighbors.items():
                if value and value.position not in nodesCheck:
                    queue.append(value)
                    nodesCheck.add(value.position)
                    value.dist = temp.dist+1
                    count = max(count,value.dist)
            if count >= 10:#break loop after searching for 10 distances away from pacman position
                queue = []
        finalList = []
        for i,node in enumerate(intersectionNodeList):#check ghost distances
            node.dist = 0
            nodesCheck = set()
            queue:list[Node] = []
            queue.append(node)
            nodesCheck.add(node.position)
            while len(queue)>0:
                temp:Node = queue.pop(0)
                if self.ghosts.containNode(temp):
                    if temp.dist > pacmanDist[i]:
                        finalList.append(node)
                    break
                for key,value in temp.neighbors.items():
                    if value and value.position not in nodesCheck:
                        queue.append(value)
                        nodesCheck.add(value.position)
                        value.dist = temp.dist+1
        safeIntersections = len(finalList)
        return safeIntersections


    def nearestScaredGhost(self,node:Node) -> int:
        for ghost in self.ghosts:
            if node.position == ghost.position:
                return 0
        nodesCheck = set()
        queue:list[Node] = []
        count = 0
        queue.append(node)
        nodesCheck.add(node.position)
        node.dist = 0
        while len(queue)>0:
            temp:Node = queue.pop(0)
            ghost:Ghost = None
            for ghost in self.ghosts:
                if ghost.node == temp and ghost.mode.current == FREIGHT:
                    return temp.dist
            for key,value in temp.neighbors.items():
                if value and value.position not in nodesCheck:
                    queue.append(value)
                    nodesCheck.add(value.position)
                    value.dist = temp.dist+1
                    count = max(count,value.dist)
        return count


    def nearestIntersection(self,node:Node) -> tuple[int,Node]:
        if node.isIntersection:
            return 0,node
        
        nodesCheck = set()
        queue:list[Node] = []
        count = 0
        queue.append(node)
        nodesCheck.add(node.position)
        node.dist = 0
        while len(queue)>0:
            temp:Node = queue.pop(0)
            if temp.isIntersection:
                return temp.dist,temp
            for key,value in temp.neighbors.items():
                if value and value.position not in nodesCheck:
                    queue.append(value)
                    nodesCheck.add(value.position)
                    value.dist = temp.dist+1
                    count = max(count,value.dist)
        return count,None#shouldn't ever reach here
        
    def nearestGhostDistToNode(self,node:Node) -> int:
        for ghost in self.ghosts:
            if ghost.node == node:
                return 0
            
        nodesCheck = set()
        queue:list[Node] = []
        count = 0
        queue.append(node)
        nodesCheck.add(node.position)
        node.dist = 0
        while len(queue)>0:
            temp:Node = queue.pop(0)
            ghost:Ghost = None
            for ghost in self.ghosts:
                if ghost.node == temp and ghost.mode.current != FREIGHT:
                    return temp.dist       
            for key,value in temp.neighbors.items():
                if value and value.position not in nodesCheck:
                    queue.append(value)
                    nodesCheck.add(value.position)
                    value.dist = temp.dist+1
                    count = max(count,value.dist)
        return count
        

    def nearestPelletDistance(self,node:Node):
        if not node:
            return 0
        if node.position in self.pellets.pelletCoords:
            return 0
        nodesCheck = set()
        queue:list[Node] = []
        queue.append(node)
        nodesCheck.add(node.position)
        node.dist = 0
        count = 0#default amount
        while len(queue)>0:
            temp:Node = queue.pop(0)
            if temp.position in self.pellets.pelletCoords:
                return temp.dist
            for key,value in temp.neighbors.items():
                if value and value.position not in nodesCheck:
                    queue.append(value)
                    nodesCheck.add(value.position)
                    value.dist = temp.dist+1
                    count = max(count,value.dist)
        return count


    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.genome.fitness += 2
            #self.count -= 10
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.removePellet(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
                self.pillTimer = 0
                self.pillActive = True
                self.genome.fitness += 3
            if self.pellets.isEmpty():
                self.hideEntities()
                self.gameWon = True
                self.genome.fitness += 50
                #self.nextLevel()


    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    #self.pacman.visible = False
                    #ghost.visible = False
                    self.updateScore(ghost.points)
                    if self.renderMode:
                        self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)  
                    #self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                    self.genome.fitness += 5
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.genome.fitness += -20
                        if self.renderMode:
                            self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()                         
                        if self.lives <= 0:
                            if self.renderMode:
                                self.textgroup.showText(GAMEOVERTXT)
                            self.gameOver = True
                            #self.resetLevel()
                        else:
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)

    
    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def render(self):
        self.screen.blit(self.background, (0, 0))

        #self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))
        pygame.display.update()
    
def eval_genomes(genomes,config):
    nets = []
    ge = []

    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)

    game = GameController(numGames=len(genomes),genomes=ge,nets=nets)
    game.startGame()
    run = True
    while run:
        run = game.update()
    

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint("./models/neat-checkpoint99") #use checkpoint 99 for simple neat algorithm observation
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5,300,'./models/neat-checkpoint'))

    winner = p.run(eval_genomes,500)
    print('\nBest genome:\n{!s}'.format(winner))
    with open("./best.pkl","wb") as f:
        pickle.dump(winner,f)
        f.close()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-feedforward.txt")
    run(config_path)