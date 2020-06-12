from ray.rllib.policy.policy import Policy

class LORHeuristic(Policy):
    """
    Heuristic policy
    if self.health >= opponent.health and self.health > 1:
        if self and opponent is adjacent:
            attack
        else:
            move torwards the opponent
    else:
        if self and opponent is adjacent:
            move away from the opponent 
        else:
            attack   
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()
    
    @staticmethod
    def take_action(obv):
        # each ob is np array (self.x, self.y, oponent.x, oppoennt.y, self.health, opponent.health)
        self_x = obv[0]
        self_y = obv[1]
        op_x = obv[2]
        op_y = obv[3]
        self_h = obv[4]
        op_h = obv[4]
        
        if self_h >= op_h and self_h > 1:
            if (self_x == op_x and abs(self_y - op_y) <= 1) or (self_y == op_y and abs(self_x - op_x) <= 1):
                return LOREnv1.ATTACK
            else:
                if self_x != op_x:
                    return LOREnv1.MOVEUP if self_x > op_x else LOREnv1.MOVEDOWN
                else:
                    return LOREnv1.MOVELEFT if self_y > op_y else LOREnv1.MOVERIGHT
        else:
            if (self_x == op_x and abs(self_y - op_y) <= 1) or (self_y == op_y and abs(self_x - op_x) <= 1):
                if self_x == op_x:
                    return LOREnv1.MOVEUP if self_x == LOREnv1.space_size_n -1  else LOREnv1.MOVEDOWN
                else:
                    return LOREnv1.MOVELEFT if self_y == LOREnv1.space_size_n -1  else LOREnv1.MOVERIGHT
            else:
                return LOREnv1.ATTACK

                

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        return [LORHeuristic.take_action(x) for x in obs_batch], [], {}
    
    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass