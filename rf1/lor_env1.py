import random
import numpy as np
from gym.spaces import Discrete,Tuple, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv

class LOREnv1(MultiAgentEnv):
    """Two-player environment for league of rookie setup1
    The game happens in a 5x5 2D space. Two players are put into two spots. 
    In each turn, the play can take one of the following actions
    - Move 1 step (one of the 4 directions)
    - Attack the opponent
    The attack action is evaluated after any move action
    
    The observation has the followings.
        - 2D position of self
        - health of self
        - 2D position of the opponent
        - health of the opponent
    """

    # all the actions
    MOVEUP = 0
    MOVEDOWN = 1
    MOVELEFT = 2
    MOVERIGHT = 3
    ATTACK = 4
    
    # max heath to start with
    max_health = 3
    
    # space is of size n x n 
    # (0, 0) is at the top left corner
    # x represents the vertical direction
    # y represents the horizontal direction
    space_size_n = 3
    
    # miss rate on any one attack
    attack_miss_rate = 0.1
    
    # each attack takes some health
    attak_power = 1
    
    # reward of win a game
    game_award = 100
    
    
    def generate_init_pos(self):
        player1_init_pos = [random.randrange(LOREnv1.space_size_n), random.randrange(LOREnv1.space_size_n)]
        player2_init_pos = [random.randrange(LOREnv1.space_size_n), random.randrange(LOREnv1.space_size_n)]
        
        while player1_init_pos == player2_init_pos:
          player2_init_pos = [random.randrange(LOREnv1.space_size_n), random.randrange(LOREnv1.space_size_n)]
        
        return player1_init_pos, player2_init_pos

    def __init__(self, config):
        self.action_space = Discrete(5)
        
        # the observation is a tuple: [self_pos_x, self_pos_y, self.health, pos_x, pos_y, health]
        # start with a discrete space
        self.observation_space = Tuple(
            [
                # self position in x/y
                Box(low = 0, high = LOREnv1.space_size_n - 1, shape=(2, ), dtype=np.int16),
                # opponent position in x/y
                Box(low = 0, high = LOREnv1.space_size_n - 1, shape=(2, ), dtype=np.int16),
                # self health and opponent health
                Box(low = 0, high = LOREnv1.max_health, shape=(2, ), dtype=np.int16),
                
            ]
        )
        
        self.player1 = "player1"
        self.player2 = "player2"
        
        # set init position
        self.player1_init_pos, self.player2_init_pos = self.generate_init_pos()
        
        self.position = {
                self.player1: self.player1_init_pos,
                self.player2: self.player2_init_pos
        }
        
        self.health = {
            self.player1: LOREnv1.max_health,
            self.player2: LOREnv1.max_health
        }
        
        # For test-case inspections (compare both players' scores).
        self.player1_score = self.player2_score = 0

    # reset the env
    # return the initial observation
    def reset(self):
        self.player1_init_pos, self.player2_init_pos = self.generate_init_pos()
        
        self.position = {
                self.player1: self.player1_init_pos,
                self.player2: self.player2_init_pos
        }
        
        self.health = {
            self.player1: LOREnv1.max_health,
            self.player2: LOREnv1.max_health
        }
        
        return {
            self.player1: tuple(
                [
                    np.array([self.position[self.player1][0], self.position[self.player1][1]]),
                    np.array([self.position[self.player2][0], self.position[self.player2][1]]),
                    np.array([self.health[self.player1], self.health[self.player2]])
                ]
            ),
            self.player2: tuple(
                [
                    np.array([self.position[self.player2][0], self.position[self.player2][1]]),
                    np.array([self.position[self.player1][0], self.position[self.player1][1]]),
                    np.array([self.health[self.player2], self.health[self.player1]])
                ]
            )
        }
    
    def move_agent(self, player, opponent, action):
        if self.health[player] <= 0:  # no health no action
            return
        
        if action == LOREnv1.MOVEUP or action == LOREnv1.MOVEDOWN:
            new_x = self.position[player][0] + (1 if action == LOREnv1.MOVEDOWN else -1)
            if new_x < 0 or new_x >= LOREnv1.space_size_n \
            or (self.position[opponent][0] == new_x and self.position[opponent][1] == self.position[player][1]):
                return # invalid move
            else:
                self.position[player][0] = new_x
                
        if action == LOREnv1.MOVELEFT or action == LOREnv1.MOVERIGHT:
            new_y = self.position[player][1] + (1 if action == LOREnv1.MOVERIGHT else -1)
            if new_y < 0 or new_y >= LOREnv1.space_size_n \
            or (self.position[opponent][1] == new_y and self.position[opponent][0] == self.position[player][0]):
                return # invalid move
            else:
                self.position[player][1] = new_y
        
        return # not a move action

    def attack_agent(self, player, opponent, action):
        if action != LOREnv1.ATTACK or self.health[player] <= 0:
            return 0 # 0 attack gain
        
        
        hit =  0 if random.random() < LOREnv1.attack_miss_rate else 1
        # attack is only valid if the two agents are adjacent (including diagonal)
        if abs(self.position[player][0] - self.position[opponent][0]) <= 1 \
            and abs(self.position[player][1] - self.position[opponent][1]) <= 1:
            self.health[opponent] = self.health[opponent] - hit * LOREnv1.attak_power
            
            return hit * LOREnv1.attak_power
        else:
            return 0 
    
    def get_reward(self, player, opponent, attack_gain):
        if self.health[player] <=0 and self.health[opponent] > 0:
            return -1 * LOREnv1.game_award
        
        if self.health[player] > 0 and self.health[opponent] <= 0:
            return LOREnv1.game_award
        
        if self.health[player] == 0 and self.health[opponent] == 0:
            return 0
        
        return attack_gain
    
    def get_reward2(self, attack_gain_player1, attack_gain_player2):
        if self.health[self.player1] <=0 and self.health[self.player2] > 0:
            return [-1 * LOREnv1.max_health, LOREnv1.max_health]
        
        if self.health[self.player1] > 0 and self.health[self.player2] <= 0:
            return [LOREnv1.max_health, -1 * LOREnv1.max_health]
        
        if self.health[self.player1] == 0 and self.health[self.player2] == 0:
            return [-1 * LOREnv1.max_health, -1 * LOREnv1.max_health]
        
        return [attack_gain_player1, attack_gain_player2]
        
    
    
    # update state and observation based on the 2 actions
    def step(self, action_dict):        
        # update position     
        # randomly pick who to move first (if both decide to move)
        who_moves_first = self.player1 if random.random() < 0.5 else self.player2
        
        if who_moves_first == self.player1: 
            self.move_agent(self.player1, self.player2, action_dict[self.player1])
            self.move_agent(self.player2, self.player1, action_dict[self.player2])
        else:
            self.move_agent(self.player2, self.player1, action_dict[self.player2])
            self.move_agent(self.player1, self.player2, action_dict[self.player1])
        
        # update attack 
        # randomly pick who to attack first (if both decide to attach)
        who_attacks_first = self.player1 if random.random() < 0.5 else self.player2
        
        if who_attacks_first == self.player1:
            attack_gain_player1 = self.attack_agent(self.player1, self.player2, action_dict[self.player1])
            attack_gain_player2 = self.attack_agent(self.player2, self.player1, action_dict[self.player2])
        else:
            attack_gain_player2 = self.attack_agent(self.player2, self.player1, action_dict[self.player2])
            attack_gain_player1 = self.attack_agent(self.player1, self.player2, action_dict[self.player1])
            
        # get the new obs
        obs = {
            self.player1: tuple(
                [
                    np.array([self.position[self.player1][0], self.position[self.player1][1]]),
                    np.array([self.position[self.player2][0], self.position[self.player2][1]]),
                    np.array([self.health[self.player1], self.health[self.player2]])
                ]
            ),
            self.player2: tuple(
                [
                    np.array([self.position[self.player2][0], self.position[self.player2][1]]),
                    np.array([self.position[self.player1][0], self.position[self.player1][1]]),
                    np.array([self.health[self.player2], self.health[self.player1]])
                ]
            )
        }
        
        # get the reward
        rew = {
            self.player1: self.get_reward(self.player1, self.player2, attack_gain_player1),
            self.player2: self.get_reward(self.player2, self.player1, attack_gain_player2),
        }
        
        done = {
            "__all__": self.health[self.player1] == 0 or self.health[self.player2] == 0,
        }

        if rew["player1"] == LOREnv1.game_award:
            self.player1_score += 1
        elif rew["player2"] == LOREnv1.game_award:
            self.player2_score += 1

        return obs, rew, done, {}