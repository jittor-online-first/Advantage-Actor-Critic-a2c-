import os
# os.environ["debug"] = "1"
# os.environ["trace_py_var"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import jittor as jt
import numpy as np
import gym
import sys
import time
from jittor.optim import Adam

class ActorCritic(jt.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        self.value = jt.nn.Sequential(
            jt.nn.Linear(input_size, hidden_size),
            jt.nn.ReLU(),
            jt.nn.Linear(hidden_size, 1)
        )
        self.policy = jt.nn.Sequential(
            jt.nn.Linear(input_size, hidden_size),
            jt.nn.ReLU(),
            jt.nn.Linear(hidden_size, num_actions),
        )

    def execute(self, x):
        value = self.value(x)
        policy = jt.nn.softmax(self.policy(x), 1)
        return value, policy

### Update per update_step
learning_rate = 3e-4
num_steps = 200
max_episodes = 1660
update_step = 200
GAMMA = 0.99

env = gym.make('CartPole-v0')
hidden_size = 256
num_actions = env.action_space.n

all_rewards = []
all_lengths = []
average_lengths = []
graphs = []
jt.clean()

model = ActorCritic(4, 256, 2)
model.eval()
model.train()
opt = Adam(model.parameters(), learning_rate)

init = lambda o,i: (np.random.rand(o,i) / np.sqrt(o)).astype("float32")

seed = 1
np.random.seed(seed)
# model.load_parameters({
#     "value.0.weight": init(256,4,), 
#     "value.0.bias":np.zeros(256,).astype("float32"), 
#     "value.2.weight": init(1,256,), 
#     "value.2.bias":np.zeros(1,).astype("float32"), 
#     "policy.0.weight": init(256,4,), 
#     "policy.0.bias":np.zeros(256,).astype("float32"), 
#     "policy.2.weight": init(2,256,), 
#     "policy.2.bias":np.zeros(2,).astype("float32"), 
# })


# print(model.parameters())
# vs = []
# for v in jt.find_vars():
#     vs.append(v().flatten())
# vs = np.concatenate(vs)
# pl.plot(vs)

env.seed(seed)
np.random.seed(seed)
jt.seed(seed)

prev_time = time.time()
start_time = prev_time

for episode in range(max_episodes):
    # print(np.random.rand(3))
    state = env.reset()
    done = False

    tot_reward = 0
    length = 0
    entropy_term = jt.float32(0).stop_fuse()
    num_step = num_steps // update_step
    for steps in range(num_step):

        rewards = []
        values = []
        log_probs = []
        
        for s in range(update_step):
            state = jt.float32(state).reshape([1,-1])
            value, policy_dist = model(state)
            value = value[0,0]
            # print("value: ", value())
            action = np.random.choice(num_actions, p=np.squeeze(policy_dist.data))
            # action = 0
            # print(action)
            log_prob = jt.log(policy_dist)[0,action]

            entropy = -jt.sum(jt.mean(policy_dist) * jt.log(policy_dist))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            state = new_state
            entropy_term += entropy
            tot_reward += reward
            length += 1
            
            if done:
                break
        
        Qvals = np.zeros(len(values))
        for t in reversed(range(len(rewards)-1)):
            Qvals[t] = rewards[t] + GAMMA * Qvals[t+1]
        
        actor_loss_sum = jt.float32(0).stop_fuse()
        critic_loss_sum = jt.float32(0).stop_fuse()
        for value, Qval, log_prob in zip(values, Qvals, log_probs):
            advantage = Qval - value
            actor_loss_sum += (-log_prob * advantage)
            critic_loss_sum += advantage.pow(2)
            
        ac_loss = (actor_loss_sum + 0.5 * critic_loss_sum)/len(values) + 0.001 * entropy_term
        
        
        opt.step(ac_loss)
        # print(ac_loss())
        # jt.nn.SGD('ActorCritic', ac_loss, learning_rate)
        
        # print(f"loss={ac_loss()}, {episode}, {jt.liveness_info()}", flush=True)
        
        if done or (steps == num_step - 1 and steps > 0):
            all_rewards.append(tot_reward)
            all_lengths.append(length)
            average_lengths.append(np.mean(all_lengths[-10:]))
            if episode % 10 == 0:
                # print(jt.liveness_info())
                nt = time.time()
                print('Episode: {}, time:{:.3f} {:.3f} {:.3f}ms, reward: {}, total length: {}, average_length: {}'.format(
                    episode, 
                    nt-prev_time, nt-start_time, (nt-prev_time)/average_lengths[-1]*1000,
                    tot_reward, length, average_lengths[-1]), flush=True)
                prev_time = nt
            break
