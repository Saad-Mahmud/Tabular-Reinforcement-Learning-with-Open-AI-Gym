import numpy as np
import gym
from collections import defaultdict
from tensorboardX import SummaryWriter

class onMC_Agent(object):

    def __init__(self, env, maxEpi=100000, gamma = .95, epsilon = 0.1, alpha = 0.01):
        self.env = env
        self.maxEpi = maxEpi
        self.gamma = gamma
        self.alpha = alpha
        self.Q1 = defaultdict(lambda: 0)
        self.Q2 = defaultdict(lambda: 0)
        self.epsilon = epsilon
        self.action_no = env.action_space.n

    def e_soft(self,S):
        r = np.random.uniform()
        if r < self.epsilon:
            return np.random.randint(0,self.action_no)
        else:
            return np.argmax(np.array([self.Q1[(S,i)]+self.Q2[(S,i)] for i in range(self.action_no)])) 
    def greedy(self,S,Q):
        return np.argmax(np.array([Q[(S,i)] for i in range(self.action_no)]))   

    def learn(self, eval = 1000, maxStep = 5000):
        self.Q1 = defaultdict(lambda: 0) # Q1 value.
        self.Q2 = defaultdict(lambda: 0) # Q2 value.
        writer = SummaryWriter(comment="TDDQ")
        self.eval(writer,0, episodes=1000,maxStep = maxStep)
        for epi in range(1,self.maxEpi+1):
            if(epi%eval==0):
                self.eval(writer,epi, episodes=1000,maxStep = maxStep)
            S = self.env.reset()
            done = False
            while not done:
                A = self.e_soft(S)
                Sd, r, done, _ = self.env.step(A)
                if np.random.uniform() < 0.5: 
                    q = self.Q1[(S,A)]
                    self.Q1[(S,A)] = q + self.alpha * (r + self.gamma * self.Q2[(Sd,self.greedy(Sd,self.Q1))] - q)
                else:
                    q = self.Q2[(S,A)]
                    self.Q2[(S,A)] = q + self.alpha * (r + self.gamma * self.Q1[(Sd,self.greedy(Sd,self.Q2))] - q)
                S = Sd   
        self.eval(writer,self.maxEpi, episodes=1000,maxStep = maxStep)
    
    def eval(self, writer, itr,  episodes=1000, maxStep = 100000):
        score = 0
        steps_list = []
        for episode in range(episodes):
            observation = self.env.reset()
            steps=0
            while True:
                action = self.greedy(observation,self.Q1)
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
        print("{}/{}".format(itr,self.maxEpi))
        print('You took an average of {:.0f} steps'.format(np.mean(steps_list)))
        print('Average reward {:.2f}'.format((score/episodes)))
        print('----------------------------------------------')
        if writer is not None:
            writer.add_scalar("Episode Length",np.mean(steps_list),itr)
            writer.add_scalar("Reward",score/episodes,itr)
        
        

if __name__ == "__main__":
    env = gym.make("FrozenLake-v0").env
    agent = onMC_Agent(env)
    agent.learn()