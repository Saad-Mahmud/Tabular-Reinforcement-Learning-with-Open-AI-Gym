import numpy as np
import gym
from collections import defaultdict
from tensorboardX import SummaryWriter

class offMCW_Agent(object):

    def __init__(self, env, maxEpi=100000, gamma = 1, epsilon = 0.1):
        self.env = env
        self.maxEpi = maxEpi
        self.gamma = gamma
        self.Q = defaultdict(lambda: 0)
        self.P = defaultdict(lambda: 0)
        self.epsilon = epsilon
        self.action_no = env.action_space.n

    def e_soft(self,observation):
        r = np.random.uniform()
        if r < self.epsilon:
            return np.random.randint(0,self.action_no)
        else:
            return self.P[observation]
            
    def Generate_Episode(self):
        observation = self.env.reset()
        qs = []
        rewards = []
        isFirst = defaultdict(lambda: 0)
        updates = []
        for t in range(100):
            action = self.e_soft(observation)
            qs.append((observation,action))
            if isFirst[(observation,action)] == 0:
                isFirst[(observation,action)] = 1
                updates.append(1)
            else:
                updates.append(0)
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        return (qs,rewards,updates)

    def learn(self, eval = 1000, maxStep = 100000):
        self.Q = defaultdict(lambda: 0) # Q value.
        C = defaultdict(lambda: 0) # Number of time a Q is visited.
        self.P = defaultdict(lambda: 0) # Greedy Policy with respect to Q.
        writer = SummaryWriter(comment="MCoffO")
        self.eval(writer,0, episodes=1000,maxStep = maxStep)
        for epi in range(1,self.maxEpi+1):
            if(epi%eval==0):
                self.eval(writer,epi, episodes=1000,maxStep = maxStep)
            qs, rewards, updates = self.Generate_Episode()
            G = 0
            W = 1
            for i in range(len(updates)-1,-1,-1):
                G = self.gamma*G+rewards[i]
                if updates[i]==1:
                    Cn = C[qs[i]]+1
                    Qn = self.Q[qs[i]]
                    C[qs[i]] = Cn
                    self.Q[qs[i]] = Qn + W/Cn * (G-Qn)  
                    best_action = np.argmax(np.asarray([self.Q[(qs[i][0],j)] for j in range(self.action_no)]))
                    self.P[qs[i][0]] = best_action
                    if qs[i][1] != best_action:
                        break
                    W = W*1/(1-self.epsilon+self.epsilon/self.action_no)
        self.eval(writer,self.maxEpi, episodes=1000,maxStep = maxStep)
    
    def eval(self, writer, itr,  episodes=1000, maxStep = 100000):
        score = 0
        steps_list = []
        for episode in range(episodes):
            observation = self.env.reset()
            steps=0
            while True:
                action = self.P[observation]
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
    agent = offMCW_Agent(env)
    agent.learn()
