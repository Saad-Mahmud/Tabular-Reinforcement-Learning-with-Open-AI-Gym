import numpy as np
from tensorboardX import SummaryWriter
from K_Armed_Testbed import K_Armed

class EG_Agent(object):
    
    def __init__(self, env, maxItr = 10000, epsilon = 0.10, decay = 0.9995):
        self.env = env 
        self.maxItr = maxItr
        self.epsilon = epsilon 
        self.decay = decay
        self.avgReward = []
        self.bestAction = []
    
    def learn(self):
        Q = [0 for i in range(env.K)]
        N = [0 for i in range(env.K)]
        self.avgReward = []
        self.rightAction = []
        avgReward = 0
        bestAction = 0
        writer = SummaryWriter(comment="KABEG_{}_{}".format(self.epsilon,self.decay))
        epsilon = self.epsilon
        for epi in range(1,self.maxItr+1):
            r = np.random.uniform()
            arm = 0
            if r < 1-epsilon:
                arm = np.random.choice(np.argwhere(Q == np.amax(Q)).flatten())
            else:
                arm = np.random.randint(0, env.K)
            reward,isBest = self.env.step(arm)
            N[arm] = N[arm]+1
            Q[arm] = Q[arm]+1.0/N[arm]*(reward-Q[arm])
            avgReward =  avgReward + 1.0/epi*(reward-avgReward)
            if isBest:
                bestAction = bestAction+1 
            epsilon = epsilon*self.decay
            self.avgReward.append(avgReward)
            self.bestAction.append(bestAction/epi*100)
            if epi%int(self.maxItr/10.0) == 0:
                print("%d/%d: average_reward=%.3f, best_action=%.1f" % (\
                    epi,self.maxItr, self.avgReward[-1], self.bestAction[-1]))
            writer.add_scalar("average_reward", self.avgReward[-1],epi)
            writer.add_scalar("best_action", self.bestAction[-1],epi)
        print("Done: average_reward=%.3f, best_action=%.1f" % (\
             self.avgReward[-1], self.bestAction[-1]))
            
if __name__ == "__main__":
    env = K_Armed(10)
    es = [0.5,0.2,0.1,0.01]
    for i in es:
        agent = EG_Agent(env,epsilon=i)
        agent.learn()