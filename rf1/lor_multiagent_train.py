import ray
from ray.rllib.agents.dqn import DQNTrainer

ray.shutdown()
ray.init()

def select_policy(agent_id):
    if agent_id == "player1":
        return "learned"
    else:
        return "LORHeuristic"

env = LOREnv1({})
    
config = {
    "env": LOREnv1,
    "gamma": 0.9,
    "num_workers": 0,
    "num_envs_per_worker": 4,
    "rollout_fragment_length": 10,
    "train_batch_size": 500,
    "multiagent": {
        "policies_to_train": ["learned"],
        "policies": {
            "LORHeuristic": (LORHeuristic, env.observation_space, env.action_space, {}),
            "learned": (None, env.observation_space, env.action_space, {
                "model": {
                        "use_lstm": True
                },
            }),
        },
        "policy_mapping_fn": select_policy,
    },
}

trainer_obj = DQNTrainer(config=config)
env = trainer_obj.workers.local_worker().env
for _ in range(100):
    results = trainer_obj.train()
    #print(results)
    
    #if _ % 100 == 0:
    print(env.player1_score, env.player2_score)