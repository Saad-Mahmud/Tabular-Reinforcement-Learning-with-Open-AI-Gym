import numpy as np
import gym
from collections import defaultdict
from tensorboardX import SummaryWriter

class PI_Agent(object):

    def __init__(self, env, maxItr=10000, pItr = 10, gamma=0.95):
        self.env = env
        self.maxItr = maxItr
        self.pItr = pItr
        self.gamma = gamma
        self.stateValue = defaultdict(lambda: 0)
        self.policy = defaultdict(lambda: 0)

    def Policy_Evaluation(self):
        for itr in range(self.pItr):
            delta = 0
            for state in range(self.env.nS): # for all state S
                state_value = 0
                for i in range(len(self.env.P[state][self.policy[state]])):
                    prob, next_state, reward, done = self.env.P[state][self.policy[state]][i]
                    state_value += (prob * (reward + self.gamma*self.stateValue[next_state]))
                delta = max(delta,abs(self.stateValue[state]-state_value))
                self.stateValue[state] = state_value  #update the value of the state
            if delta < 1e-04:
                break

    def Policy_Improvement(self):
        Done = True
        for state in range(self.env.nS):
            action_values = []
            for action in range(self.env.nA):
                action_value = 0
                for i in range(len(self.env.P[state][action])):
                    prob, next_state, r, _ = self.env.P[state][action][i]
                    action_value += prob * (r + self.gamma * self.stateValue[next_state])
                action_values.append(action_value)
            best_action = np.argmax(np.asarray(action_values))
            if(self.policy[state] != best_action):
                Done = False
            self.policy[state] = best_action
        return Done

    def learn(self, eval = 1, maxStep = 1000):
        self.stateValue = defaultdict(lambda: 0)
        self.policy = defaultdict(lambda: 0)
        Done = False
        writer = SummaryWriter(comment="DPPI")
        self.eval(writer,0, episodes=1000,maxStep = maxStep)
        for itr in range(self.maxItr):
            self.Policy_Evaluation()
            Done = self.Policy_Improvement()
            if itr%eval == 0:
                self.eval(writer,itr, episodes=1000,maxStep = maxStep)
            if Done:
                break
        self.eval(writer,self.maxItr, episodes=1000,maxStep = maxStep)

    
    def eval(self, writer, itr, episodes=1000, maxStep = 1000):
        score = 0
        steps_list = []
        for episode in range(episodes):
            observation = self.env.reset()
            steps=0
            while True:
                action = self.policy[observation]
                observation, reward, done, _ = self.env.step(action)
                steps+=1
                score+=reward
                if done:
                    steps_list.append(steps)
                    break
                if steps>maxStep:
                    steps_list.append(steps)
                    break
        print('----------------------------------------------')
        print('You took an average of {:.0f} steps'.format(np.mean(steps_list)))
        print('Average reward {:.2f}'.format((score/episodes)))
        print('----------------------------------------------')
        writer.add_scalar("Episode Length",np.mean(steps_list),itr)
        writer.add_scalar("Reward",score/episodes,itr)


if __name__ == "__main__":
    env = gym.make("Taxi-v3").env
    agent = PI_Agent(env)
    agent.learn()

     
