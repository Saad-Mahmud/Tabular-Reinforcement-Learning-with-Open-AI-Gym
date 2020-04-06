import numpy as np
import gym
from collections import defaultdict
from tensorboardX import SummaryWriter

class VI_Agent(object):

    def __init__(self, env, maxItr=10000, gamma=0.95):
        self.env = env
        self.maxItr = maxItr
        self.gamma = gamma
        self.stateValue = defaultdict(lambda: 0)
        self.policy = defaultdict(lambda: 0)

    def learn(self,eval = 2, maxStep = 1000):
        self.stateValue = defaultdict(lambda: 0)
        writer = SummaryWriter(comment="DPVI")
        self.cal_policy() 
        self.eval(writer,0, episodes=1000,maxStep = maxStep)
        for itr in range(1,self.maxItr+1):
            delta = 0
            for state in range(self.env.nS): # for all state S
                action_values = []      
                for action in range(self.env.nA): #maximize over action 
                    state_value = 0
                    for i in range(len(self.env.P[state][action])):
                        prob, next_state, reward, done = self.env.P[state][action][i]
                        state_value += prob * (reward + self.gamma*self.stateValue[next_state])
                    action_values.append(state_value)      #the value of each action
                best_action = np.argmax(np.asarray(action_values))   # choose the action which gives the maximum value
                delta = max(delta,abs(action_values[best_action]-self.stateValue[state]))
                self.stateValue[state] = action_values[best_action]  #update the value of the state
            #print({i:stateValue[i] for i in range(env.nS)})
            if itr%eval == 0:
                self.cal_policy()
                self.eval(writer,itr, episodes=1000,maxStep = maxStep)
            if delta < 1e-04:   # if there is negligible difference break the loop
                break
        self.cal_policy() 
        self.eval(writer,self.maxItr, episodes=1000,maxStep = maxStep)

    def cal_policy(self):
        self.policy = defaultdict(lambda: 0)
        for state in range(self.env.nS):
            action_values = []
            for action in range(self.env.nA):
                action_value = 0
                for i in range(len(self.env.P[state][action])):
                    prob, next_state, r, _ = self.env.P[state][action][i]
                    action_value += prob * (r + self.gamma * self.stateValue[next_state])
                action_values.append(action_value)
            best_action = np.argmax(np.asarray(action_values))
            self.policy[state] = best_action
        
    
    def eval(self,writer,itr, episodes=1000, maxStep = 1000):
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
    agent = VI_Agent(env)
    agent.learn()