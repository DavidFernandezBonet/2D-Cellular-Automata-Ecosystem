"""
Python-Ecosystem


"""

# run.py takes care of creating the world and animating it

import numpy as np

from random import randint, random
from math import sqrt, inf
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#--------------------------------------------------------------------------
def distance(agent1, agent2):
    """
    Measures the distance between two agents
    :param agent1, agent2: an animal, bunny or fox
    :type agent1, agent2: Object
    :return: distance
    :rtype: Float
    """
    return max(sqrt((agent1.x - agent2.x)**2 + (agent1.y - agent2.y)**2), 0.1)


def unitVector(agent1, agent2):
    """
    Returns the unit vector from agent1 to agent2
    :param agent1, agent2: an animal, bunny or fox
    :type agent1, agent2: Object
    :return: unit vector (x, y)
    :rtype: Tuple
    """
    d = distance(agent1, agent2)
    return ((agent2.x - agent1.x)/d, (agent2.y - agent1.y)/d)


def legalMove(move, state):
    """
    Checks if the move is possible and is not out of bounds
    :param move: next potential position for an agent (x, y)
    :type move: Tuple
    :param state: state, 2D array of size h*w with 0 if the spot is empty or the id of an agent if an agent is in the spot
    :type state: Array
    :return: True if the move is legal, False elsewise
    :rtype: Bool
    """
    yMax = len(state)
    xMax = len(state[0])
    if move[0] < 0 or move[0] >= xMax:
        return False
    if move[1] < 0 or move[1] >= yMax:
        return False
    return True


def moveTowards(agent, agentT, state, direction):
    """
    Move agent towards agentT. If the move is illegal, move randomly
    :param agent, agentT: an animal, fox or bunny
    :type agent, agentT: Object
    :param state: state, 2D array of size h*w with 0 if the spot is empty or the id of an agent if an agent is in the spot
    :type state: Array
    :param direction: 1 if agent wants to move towards agentT, -1 if agent wants to run away from agentT
    :type direction: int
    """
    u = unitVector(agent, agentT)
    xU = u[0]
    yU = u[1]
    if abs(xU) >= abs(yU):
        if xU > 0:
            xU = 1*direction
        else:
            xU = -1*direction
        move = (agent.x + xU, agent.y)
        if legalMove(move, state):
            (agent.x, agent.y) = move
        else:
            randomMovement(agent, state)
    else:
        if yU > 0:
            yU = 1*direction
        else:
            yU = -1*direction
        move = (agent.x, agent.y + yU)
        if legalMove(move, state):
            (agent.x, agent.y) = move
        else:
            randomMovement(agent, state)


def randomMovement(agent, state):
    """
    Move randomly where it is legal to move
    :param agent: an animal, fox or bunny
    :type agent: Object
    :param state: state, 2D array of size h*w with 0 if the spot is empty or the id of an agent if an agent is in the spot
    :type state: Array
    """
    r = randint(0, 3)
    x = agent.x
    y = agent.y
    moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    move = moves[r]
    if legalMove(move, state):
        (agent.x, agent.y) = move
    else:
        randomMovement(agent, state)


def detectPrey(agent, liveAgents, animal):
    """
    Detects if agent can see an instance of type animal in his visibility range
    :param agent: an animal, fox or bunny
    :type agent: Object
    :param: liveAgents, a dictionary with key=id_of_agent and value=agent
    :type liveAgents: Dict
    :param: animal: fox or bunny class
    :type: animal: Class
    """
    minPrey = None
    minDist = inf
    minKey = None
    for key in liveAgents:
        prey = liveAgents[key]
        if prey != agent:
            if isinstance(prey, animal):
                dist = distance(agent, prey)
                if dist <= agent.visibility and dist < minDist:
                    minPrey = prey
                    minDist = dist
                    minKey = key
    return minPrey, minKey

# Bunny class, its variables are explained in run.py


class Bunny:
    def __init__(self, x, y, speed, visibility, gestChance, gestStatus, gestNumber, age):
        self.x = x
        self.y = y
        self.speed = speed
        self.visibility = visibility
        self.gestChance = gestChance
        self.gestStatus = gestStatus
        self.gestNumber = gestNumber
        self.age = age

    # act controls the behavior of the agent at every step of the simulation
    def act(self, t, state, liveAgents, age_bunny):
        self.age -= 1  # decrease the age (if age reaches 0, the agent dies)
        if self.age == 0:  # kill the agent if age reaches O
            for key in liveAgents:
                if liveAgents[key] == self:
                    liveAgents.pop(key, None)
                    break
        # the agent can only act on some values of t (time), the frequency of these values are defined by speed
        if t % self.speed == 0:
            # check for foxes in the area
            minFox, minFKey = detectPrey(self, liveAgents, Fox)
            if minFox != None:  # if there is a fox, run away
                moveTowards(self, minFox, state, -1)
            elif self.gestStatus == 0:  # if there is no fox and the agent doesn't want to reproduce, move randomly
                # random chance to want to reproduce next turn
                self.gestStatus = int(random() < self.gestChance)
                randomMovement(self, state)
            else:
                # if the agent wants to reproduce, find another bunny
                minPrey, minKey = detectPrey(self, liveAgents, Bunny)
                if minPrey != None:
                    moveTowards(self, minPrey, state, 1)
                    if self.x == minPrey.x and self.y == minPrey.y:  # if a bunny has been found, reproduce
                        self.gestStatus = 0
                        maxKey = 0
                        for key in liveAgents:  # find an unassigned key in liveAgents for the newborns
                            if key > maxKey:
                                maxKey = key
                        for i in range(self.gestNumber):
                            # the newborns are a copy of the parent
                            liveAgents[maxKey + i + 1] = deepcopy(self)
                            # reset the age of the newborns
                            liveAgents[maxKey + i + 1].age = age_bunny

                else:  # if no partner found, move randomly
                    randomMovement(self, state)

# Fox class, its variables are explained in run.py


class Fox:
    def __init__(self, x, y, speed, visibility, age, huntStatus, hunger, hungerThresMin, hungerThresMax, hungerReward, maxHunger,
                 gestChance, gestStatus, gestNumber):
        self.x = x
        self.y = y
        self.speed = speed
        self.visibility = visibility
        self.age = age
        self.huntStatus = huntStatus
        self.hunger = hunger
        self.hungerThresMin = hungerThresMin
        self.hungerThresMax = hungerThresMax
        self.hungerReward = hungerReward
        self.maxHunger = maxHunger
        self.gestChance = gestChance
        self.gestStatus = gestStatus
        self.gestNumber = gestNumber

    # act controls the behavior of the agent at every step of the simulation
    def act(self, t, state, liveAgents, age_fox):
        self.age -= 1  # decrease age (if age reaches O, the agent dies)
        # decrease hunger (if hunger reaches O, the agent dies)
        self.hunger -= 1
        # hunger can't go over maxHunger
        self.hunger = min(self.maxHunger, self.hunger)
        if self.age == 0 or self.hunger == 0:  # kill the agent in case of starvation or aging
            for key in liveAgents:
                if liveAgents[key] == self:
                    liveAgents.pop(key, None)
                    break
        # the agent can only act on some values of t (time), the frequency of these values are defined by speed
        if t % self.speed == 0:
            if self.huntStatus == 0:  # if not hunting
                if self.hunger <= self.hungerThresMin:  # if hunger goes under thresholdMin, go hunting
                    self.huntStatus = 1
                if self.gestStatus == 1:  # if the agent wants to reproduce, find another fox
                    minPrey, minKey = detectPrey(self, liveAgents, Fox)
                    if minPrey != None:
                        moveTowards(self, minPrey, state, 1)
                        if self.x == minPrey.x and self.y == minPrey.y:  # if another fox is found, reproduce
                            self.gestStatus = 0
                            maxKey = 0
                            for key in liveAgents:  # find an unassigned key for the newborns
                                if key > maxKey:
                                    maxKey = key
                            for i in range(self.gestNumber):
                                # the newborns are copies of the parent
                                liveAgents[maxKey + i + 1] = deepcopy(self)
                                # reset the age of the newborns
                                liveAgents[maxKey + i + 1].age = age_fox
                else:
                    if self.gestChance > random():  # random chance to want to reproduce
                        self.gestStatus = 1
            else:  # if the agent wants to hunt
                if self.hunger >= self.hungerThresMax:  # if hunger goes over thresholdMax, stop hunting
                    self.huntStatus = 0
                minPrey, minKey = detectPrey(
                    self, liveAgents, Bunny)  # find a prey
                if minPrey != None:
                    moveTowards(self, minPrey, state, 1)
                    if self.x == minPrey.x and self.y == minPrey.y:  # if the agent is on the prey, kill the prey
                        liveAgents.pop(minKey, None)
                        self.hunger += self.hungerReward




class Wolf:
    def __init__(self, x, y, speed, visibility, age, huntStatus, hunger, hungerThresMin, hungerThresMax, hungerReward, maxHunger,
                 gestChance, gestStatus, gestNumber):
        self.x = x
        self.y = y
        self.speed = speed
        self.visibility = visibility
        self.age = age
        self.huntStatus = huntStatus
        self.hunger = hunger
        self.hungerThresMin = hungerThresMin
        self.hungerThresMax = hungerThresMax
        self.hungerReward = hungerReward
        self.maxHunger = maxHunger
        self.gestChance = gestChance
        self.gestStatus = gestStatus
        self.gestNumber = gestNumber

    # act controls the behavior of the agent at every step of the simulation
    def act(self, t, state, liveAgents, age_Wolf):
        self.age -= 1  # decrease age (if age reaches O, the agent dies)
        # decrease hunger (if hunger reaches O, the agent dies)
        self.hunger -= 1
        # hunger can't go over maxHunger
        self.hunger = min(self.maxHunger, self.hunger)
        if self.age == 0 or self.hunger == 0:  # kill the agent in case of starvation or aging
            for key in liveAgents:
                if liveAgents[key] == self:
                    liveAgents.pop(key, None)
                    break
        # the agent can only act on some values of t (time), the frequency of these values are defined by speed
        if t % self.speed == 0:
            if self.huntStatus == 0:  # if not hunting
                if self.hunger <= self.hungerThresMin:  # if hunger goes under thresholdMin, go hunting
                    self.huntStatus = 1
                if self.gestStatus == 1:  # if the agent wants to reproduce, find another wolf
                    minPrey, minKey = detectPrey(self, liveAgents, Wolf)
                    if minPrey != None:
                        moveTowards(self, minPrey, state, 1)
                        if self.x == minPrey.x and self.y == minPrey.y:  # if another fox is found, reproduce
                            self.gestStatus = 0
                            maxKey = 0
                            for key in liveAgents:  # find an unassigned key for the newborns
                                if key > maxKey:
                                    maxKey = key
                            for i in range(self.gestNumber):
                                # the newborns are copies of the parent
                                liveAgents[maxKey + i + 1] = deepcopy(self)
                                # reset the age of the newborns
                                liveAgents[maxKey + i + 1].age = age_Wolf
                else:
                    if self.gestChance > random():  # random chance to want to reproduce
                        self.gestStatus = 1
            else:  # if the agent wants to hunt
                if self.hunger >= self.hungerThresMax:  # if hunger goes over thresholdMax, stop hunting
                    self.huntStatus = 0

                #I include the fox as a prey as well
                minPrey, minKey = detectPrey(
                    self, liveAgents, Fox)  # find a prey

                if minPrey != None:
                    moveTowards(self, minPrey, state, 1)
                    if self.x == minPrey.x and self.y == minPrey.y:  # if the agent is on the prey, kill the prey
                        liveAgents.pop(minKey, None)
                        self.hunger += self.hungerReward

#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------

def createWorld(h, w, n_bunnies, speed_bunny_min, speed_bunny_max, visibility_bunny, gestChance_bunny,
                gestStatus_bunny, gestNumber_bunny, age_bunny, n_foxes, speed_fox, visibility_fox, huntStatus_fox, age_fox,
                hunger_fox, hungerThresMin_fox, hungerThresMax_fox, hungerReward_fox, maxHunger_fox, gestChance_fox,
                gestStatus_fox, gestNumber_fox,n_wolves, speed_wolf, visibility_wolf, huntStatus_wolf, age_wolf,
                hunger_wolf, hungerThresMin_wolf, hungerThresMax_wolf, hungerReward_wolf, maxHunger_wolf, gestChance_wolf,
                gestStatus_wolf, gestNumber_wolf):
    """
    Creates an initial world by generating agents with their initial parameters on a h*w 2D grid
    :param h, w: size of the world (height, width)
    :type h, w: Int
    :param parameters of the agents: explained down there
    :type parameters of the agents: Int or Float
    :return: state, 2D array of size h*w with 0 if the spot is empty or the id of an agent if an agent is in the spot
    :rtype: Array
    :return: liveAgents, a dictionary with key=id_of_agent and value=agent
    :rtype: Dict
    """
    state = np.zeros((h, w))
    liveAgents = {}
    for i in range(1, n_bunnies + 1):
        x = randint(0, w - 1)
        y = randint(0, h - 1)
        state[y][x] = i
        liveAgents[i] = Bunny(
            x, y, randint(speed_bunny_min, speed_bunny_max), visibility_bunny, gestChance_bunny, gestStatus_bunny, gestNumber_bunny, age_bunny)

    for j in range(n_bunnies + 1, n_bunnies + 1 + n_foxes):
        x = randint(0, w - 1)
        y = randint(0, h - 1)
        state[y][x] = j
        liveAgents[j] = Fox(x, y, speed_fox, visibility_fox, age_fox, huntStatus_fox,
                            hunger_fox, hungerThresMin_fox, hungerThresMax_fox, hungerReward_fox, maxHunger_fox, gestChance_fox, gestStatus_fox, gestNumber_fox)

    for k in range( n_bunnies + 1 + n_foxes,  n_bunnies + 1 + n_foxes + n_wolves):
        x = randint(0, w - 1)
        y = randint(0, h - 1)
        state[y][x] = k
        liveAgents[k] = Wolf(x, y, speed_wolf, visibility_wolf, age_wolf, huntStatus_wolf,
                            hunger_wolf, hungerThresMin_wolf, hungerThresMax_wolf, hungerReward_wolf, maxHunger_wolf, gestChance_wolf, gestStatus_wolf, gestNumber_wolf)

    return state, liveAgents


def updateState(state, liveAgents):
    """
    updates state according to liveAgents
    :param state: state, 2D array of size h*w with 0 if the spot is empty or the id of an agent if an agent is in the spot
    :type state: Array
    :param: liveAgents, a dictionary with key=id_of_agent and value=agent
    :type liveAgents: Dict
    :return: state, 2D array of size h*w with 0 if the spot is empty or the id of an agent if an agent is in the spot
    :rtype: Array
    """
    state = np.zeros((len(state), len(state[0])))
    for key in liveAgents:
        agent = liveAgents[key]
        x = agent.x
        y = agent.y
        state[y][x] = key
    return state


def step(t, state, liveAgents):
    """
    Asks every agent to act according to their act function
    :param t: time
    :type t: Int
    :param state: state, 2D array of size h*w with 0 if the spot is empty or the id of an agent if an agent is in the spot
    :type state: Array
    :param: liveAgents, a dictionary with key=id_of_agent and value=agent
    :type liveAgents: Dict
    :return: state, 2D array of size h*w with 0 if the spot is empty or the id of an agent if an agent is in the spot
    :rtype: Array
    """
    for key in liveAgents.copy():
        if key in liveAgents:
            agent = liveAgents[key]
            agent.act(t, state, liveAgents, age_fox)
    state = updateState(state, liveAgents)
    return state


def export(liveAgents):
    """
    Exports the coordinates of the agents in lists readable by matplotlib
    :param: liveAgents, a dictionary with key=id_of_agent and value=agent
    :type liveAgents: Dict
    :return: XBunnies, YBunnies, XFoxes, YFoxes, list of the coordinates of the agents
    :rtype: List
    """
    XBunnies = []
    YBunnies = []
    XFoxes = []
    YFoxes = []
    XWolves= []
    YWolves=[]
    for key in liveAgents:
        agent = liveAgents[key]
        if isinstance(agent, Bunny):
            XBunnies.append(agent.x)
            YBunnies.append(agent.y)
        elif isinstance(agent, Fox):
            XFoxes.append(agent.x)
            YFoxes.append(agent.y)
        elif isinstance(agent, Wolf):
            XWolves.append(agent.x)
            YWolves.append(agent.y)
    return XBunnies, YBunnies, XFoxes, YFoxes, XWolves, YWolves


def count(liveAgents):
    """
    counts living bunnies and foxes and the average bunny speed (for natural selection )
    :param: liveAgents, a dictionary with key=id_of_agent and value=agent
    :type liveAgents: Dict
    :return: liveBunnies, liveFoxes, avgSpeed
    :rtype: Int or Float
    """
    liveBunnies = 0
    liveFoxes = 0
    liveWolves= 0
    speed = 0
    for key in liveAgents:
        agent = liveAgents[key]
        if isinstance(agent, Bunny):
            liveBunnies += 1
            speed += agent.speed
        elif isinstance(agent, Fox):
            liveFoxes += 1
        elif isinstance(agent, Wolf):
            liveWolves += 1
    return liveBunnies, liveFoxes, liveWolves, speed/max(liveBunnies, 0.1)


# Initialization of the variables
w = 50  # width of world
h = 50  # height of world
#Things I normally change:
n_bunnies=80
n_foxes=25
n_wolves=3
max_t=5000

#Bunnies--------

speed_bunny_max = 7 # maximum bunny speed (for natural selection study)
speed_bunny_min = 2  # minimum bunny speed (for natural selection study)
visibility_bunny = 200  # vision range of bunnies
gestChance_bunny = 0.25# chance to want to reproduce for bunnies (initial value 0.0008)
# reproduction status (0 for don't want to reproduce, 1 elsewise)
gestStatus_bunny = 0
gestNumber_bunny = 3  # bunnies created per reproduction
# bunny age (age decreases over time. If age reaches 0, the agent dies)
age_bunny = 10000

#FOXES--------

speed_fox = 3  # fox speed
visibility_fox = 400 # vision range of foxes
age_fox = 1000000 # fox age (my take, 1300)
huntStatus_fox = 0  # hunting status (0 not on the hunt, 1 elsewise)
# hunger of foxes (hunger decreases over time, If hunger reaches 0, the agent dies)
hunger_fox = 900
# if hunger goes under this threshold, the agent starts hunting
hungerThresMin_fox = 350
hungerThresMax_fox = 450  # if hunger goes over this threshold, the agent stops hunting
hungerReward_fox = 200  # hunger reward per bunny kill (original 150)
maxHunger_fox = 500  # hunger max limit (original 500)
gestChance_fox = 0.001  # chance to want to reproduce for foxes (original 0.0004)
# reproduction status for bunnies (0 for don't want to reproduce, 1 elsewise)
gestStatus_fox = 0
gestNumber_fox = 2  # foxes created per reproduction

#Wolves-------

speed_wolf = 1#  speed
visibility_wolf = 50  # vision range
age_wolf = 100000000  # age
huntStatus_wolf = 0  # hunting status (0 not on the hunt, 1 elsewise)
hunger_wolf = 250
# if hunger goes under this threshold, the agent starts hunting
hungerThresMin_wolf = 350
hungerThresMax_wolf = 450  # if hunger goes over this threshold, the agent stops hunting
hungerReward_wolf = 300  # hunger reward per FOX kill
maxHunger_wolf = 500 # hunger max limit
gestChance_wolf = 0.0004  # chance to want to reproduce
# reproduction status for bunnies (0 for don't want to reproduce, 1 elsewise)
gestStatus_wolf = 0
gestNumber_wolf = 1  # wolves created per reproduction

# Change the font size for matplotlib
size = 8
small_size = 6
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=small_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=small_size)

# Setting up the plots
fig = plt.figure()
ax1 = plt.subplot(221, title="Ecosystem (blue=bunny; red=fox; green=wolf)",
                  xlabel="x (-)", ylabel="y (-)")
plt.xlim(0, w)
plt.ylim(0, h)
bunnies, = ax1.plot([], [], 'bo', ms=4)
foxes, = ax1.plot([], [], 'ro', ms=4)
wolves, = ax1.plot([], [], 'go', ms=4)


# Plot to study the evolution of average speed of bunnies over time, for natural selection study
ax2 = plt.subplot(224, title="Average speed of bunnies over time (red=fox speed)",
                  xlabel="time (-)", ylabel="speed (less is faster) (-)")
plt.xlim(0, 5000)
plt.ylim(7, 2)
plt.plot([0, 5000], [speed_fox, speed_fox], color='r')
speedData, = ax2.plot([], [])


ax3 = plt.subplot(222, title="Population over time",
                  xlabel="time (-)", ylabel="population (-)")
plt.xlim(0, 5000)
plt.ylim(0, 200)
popBunnyData, = ax3.plot([], [])
popFoxData, = ax3.plot([], [], color='r')
popWolfData, = ax3.plot([], [], color='g')

fig.tight_layout(pad=1.5)


def init():
    """initialize animation"""
    bunnies.set_data([], [])
    foxes.set_data([], [])
    wolves.set_data([], [])
    popBunnyData.set_data([], [])
    popFoxData.set_data([], [])
    popWolfData.set_data([], [])
    speedData.set_data([], [])
    return bunnies, foxes, wolves, popBunnyData, popFoxData, popWolfData, speedData,


# Create a new world
(state, liveAgents) = createWorld(w, h, n_bunnies,
                                  speed_bunny_min, speed_bunny_max, visibility_bunny, gestChance_bunny, gestStatus_bunny,
                                  gestNumber_bunny, age_bunny, n_foxes, speed_fox, visibility_fox, age_fox, huntStatus_fox,
                                  hunger_fox, hungerThresMin_fox, hungerThresMax_fox, hungerReward_fox, maxHunger_fox,
                                  gestChance_fox, gestStatus_fox, gestNumber_fox,n_wolves, speed_wolf, visibility_wolf, huntStatus_wolf, age_wolf,
                                  hunger_wolf, hungerThresMin_wolf, hungerThresMax_wolf, hungerReward_wolf, maxHunger_wolf, gestChance_wolf,
                                  gestStatus_wolf, gestNumber_wolf)
t = 0  # time
T = []
popBunnyList = []
popFoxList = []
popWolfList = []
speedList = []

# Animation function


def animate(i):
    global t, state, liveAgents
    state = step(t, state, liveAgents)  # execute a step
    t += 1  # increment time
    T.append(t)  # time list for matplotlib

    totalCount = count(liveAgents)
    popBunnyList.append(totalCount[0])  # update the number of live bunnies
    popFoxList.append(totalCount[1])  # update the number of live foxes
    speedList.append(totalCount[3])  # update the average speed of bunnies
    popWolfList.append(totalCount[2])  # update the number of live foxes
    # export the positions of the agents for matplotlib
    (Xbunnies, Ybunnies, XFoxes, YFoxes, XWolves, YWolves) = export(liveAgents)

    # Set data for animation
    bunnies.set_data(Xbunnies, Ybunnies)
    foxes.set_data(XFoxes, YFoxes)
    wolves.set_data(XWolves,YWolves)
    popBunnyData.set_data(T, popBunnyList)
    popFoxData.set_data(T, popFoxList)
    popWolfData.set_data(T, popWolfList)
    speedData.set_data(T, speedList)

    return bunnies, foxes, wolves, popBunnyData, popFoxData, popWolfData, speedData,


# Animation
ani = animation.FuncAnimation(fig, animate, frames=400,
                              interval=5, blit=True, init_func=init)

plt.show()

print(liveAgents)
print(popBunnyList)
print(popWolfData)
print(popWolfList)