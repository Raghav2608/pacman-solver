# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    
    def __init__(self):
        print("Starting up MDPAgent!")
        name = "Pacman"

        self.prob = 0.8

        self.reward_map = {
            "ghosts":-50,
            "food":10,
            "finish":100,
            "danger":-20,
            "init":0
        }
        self.ACTIONS = {
        "North": (0, 1),
        "South": (0, -1),
        "East":  (1, 0),
        "West":  (-1, 0),
        }

        self.PERPENDICULAR = {
        "North": ("West", "East"),
        "South": ("East", "West"),
        "East":  ("North", "South"),
        "West":  ("South", "North"),
        }
        self.gamma = 0.9
        self.danger_distance = 2
        self.value = {}
        self.policy = {}
        self.reward = {}

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        corners = api.corners(state)
        self.height = self.getLayoutHeight(corners)
        self.width = self.getLayoutWidth(corners)
        self.food = api.food(state)
        self.max_food = len(self.food)
        self.ghosts = api.ghosts(state)
        self.walls = api.walls(state)
        for x in range(self.width):
            for y in range(self.height):
                s = (x,y)
                if s in self.walls:
                    continue
                self.policy[s] = random.choice(list(self.ACTIONS.keys())) # selecting a random policy 
                self.value[s] = 0

        # This is what gets run in between multiple games

        
    def getLayoutHeight(self, corners):
        yVals = []
        for i in range(len(corners)):
            yVals.append(corners[i][1])
        return max(yVals) + 1

    def getLayoutWidth(self, corners):
        xVals = []
        for i in range(len(corners)):
            xVals.append(corners[i][0])
        return max(xVals) + 1

    def check_pos_bounds(self,pos):
        return pos[0] < self.width and pos[0] >= 0 and pos[1] < self.height and pos[1] >= 0

    def final(self, state):
        print("Looks like the game just ended!")

    def update_position(self,pos,action):
        x,y = pos
        x += self.ACTIONS[action][0]
        y += self.ACTIONS[action][1]
        return x,y

    def valid_pos(self,pos):
        return pos not in self.walls and self.check_pos_bounds(pos)    

    def getNextStates(self,pos,action):
        """gets next state with state and action"""
        left, right = self.PERPENDICULAR[action]
        possible_outcomes = [
            (self.update_position(pos,action),self.prob),
            (self.update_position(pos,left),(1 - self.prob)/2),
            (self.update_position(pos,right),(1 - self.prob)/2)
        ]

        return [(pos,prob) for pos, prob in possible_outcomes if self.valid_pos(pos)]

    def manhattanDistance(self, xy1, xy2 ):
        "Returns the Manhattan distance between points xy1 and xy2"
        return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

    def danger_zone(self,state):
        danger_zone = []
        for s in self.value:
            if s in self.ghosts:
                continue
            for ghosts in api.ghosts(state) :
                if self.manhattanDistance(ghosts,s) <= self.danger_distance:
                    danger_zone.append(s)
        return danger_zone

    def update_reward(self, state):
        self.reward = {s: self.reward_map["init"] for s in self.value}
        numerator = self.max_food - len(api.food(state))
        self.reward_map["food"] = 10 + int(30*numerator /self.max_food)
        
        for f in api.food(state):
            self.reward[f] += self.reward_map["food"]
        
        for d in self.danger_zone(state):
            if d in self.reward:
                self.reward[d] += self.reward_map["danger"]
        
        for g in api.ghosts(state):
            g = (int(g[0]), int(g[1]))
            if g in self.reward:
                self.reward[g] += self.reward_map["ghosts"]
        
    def q_value(self, state, action):
        """Expected utility of taking `action` in `state`."""
        return sum(p * (self.reward[state] + self.gamma * self.value[s2])
                for s2, p in self.getNextStates(state, action))


    def max_utility(self, pos):
        """Returns (best_value, best_action) for a state."""
        actions = self.ACTIONS.keys()
        actions = [action for action in actions if self.valid_pos(self.update_position(pos,action))]
        if not actions:                      # terminal state
            return self.reward[pos], None

        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        if not actions:                      # terminal state
            return self.reward[pos], None
        
        best_value, best_action = float("-inf"), None
        for action in actions:
            q = self.q_value(pos, action)
            if q > best_value:
                best_value, best_action = q, action
        return best_value, best_action


    def policy_evaluation(self,state, theta, max_iter=200):
        self.update_reward(state)
        for _ in range(max_iter):
            diff = 0
            new_value = {}
            for s in self.value:
                new_value[s] = self.q_value(s, self.policy[s])
                diff = max(diff, abs(self.value[s] - new_value[s]))
            self.value = new_value
            if diff < theta:
                break
    
    def policy_improvement(self,state):
        stable = True
        for s in self.value:
            _, best_action = self.max_utility(pos=s)
            if best_action != self.policy[s]:
                self.policy[s] = best_action
                stable = False
        return stable

    def policy_iteration(self,state,max_rounds=30):
        stable = False
        for _ in range(max_rounds):
            self.policy_evaluation(state=state,theta=0.01)
            if self.policy_improvement(state):
                break

    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        self.policy_iteration(state)
        action = self.policy[api.whereAmI(state)]
        # self.policy_iteration(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Random choice between the legal options.
        return action if action in legal else random.choice(legal)

# • A finite set of states S;
# • A finite set of actions A;
# • A state-transition function P(s′|s, a);
# • A reward function R;
# • A discount factor γ ∈ [0, 1];

# state [positions of ghosts, fruits near me, my position]

# deterministic
# Reward Function : 

# print(len(api.food(state))) locations of all available food states
# deletes 