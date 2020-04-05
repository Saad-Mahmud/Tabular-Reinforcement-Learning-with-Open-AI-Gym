import numpy as np
from tensorboardX import SummaryWriter
from K_Armed_Testbed import K_Armed

    
class GB_Agent(object):
    
    def __init__(self, env, maxItr = 10000, alpha = 0.1):
        self.env = env 
        self.maxItr = maxItr
        self.alpha = alpha
        self.avgReward = []
        self.bestAction = []
    
    def learn(self):
        H = [0 for i in range(self.env.K)]
        self.avgReward = []
        self.rightAction = []
        avgReward = 0
        bestAction = 0
        writer = SummaryWriter(comment="KABGB_{}".format(self.alpha))
        for epi in range(1,self.maxItr+1):
            policy = np.array([np.e**i for i in H])
            policy = policy/np.sum(policy)
            arm = np.random.choice(self.env.K,p=policy)
            reward,isBest = self.env.step(arm)
            avgReward =  avgReward + 1.0/epi*(reward-avgReward)
            if isBest:
                bestAction = bestAction+1 
            for i in range(self.env.K):
                if i == arm:
                    H[i] = H[i] + self.alpha * (reward - avgReward) * (1 - policy[i])
                else:
                    H[i] = H[i] - self.alpha * (reward - avgReward) * policy[i]
            
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
    es = [0.1,0.05,0.01]
    for i in es:
        agent = GB_Agent(env,alpha=i)
        agent.learn()
    