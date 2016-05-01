from __future__ import division
import math
import random
import copy
from numpy.random import beta
import numpy as np
import matplotlib.pyplot as plt

class Arms():
    def __init__(self, mus):
        self.mus = mus
        self.n_arms = len(mus)
        self.best = max(mus)
        assert all(0 <= mu <= 1 for mu in mus)

    def __str__(self):
        return str(self.mus)

    def pull(self, idx):
        # Bernoulli reward
        return 1 if random.random() < self.mus[idx] else 0


def experiment(arms, policy, T, N=1):
    ''' Run experiment N times, each with timespan T
        and return average total regret '''
    best_mu = arms.best
    n_arms = arms.n_arms
    total_regret = 0
    policy_backup = copy.deepcopy(policy)

    for n in range(N):
        policy = copy.deepcopy(policy_backup)
        history = [[0, 0] for _ in range(n_arms)]
        for t in range(T):
            picked = policy.pick(n_arms, history)
            reward = arms.pull(picked)
            history[picked][0] += reward
            history[picked][1] += 1
            total_regret += best_mu - arms.mus[picked]
    return total_regret / N

def experiment_range(arms, policy, T, draw_points, N=1):
    ''' Run experiment N times, each with timespan T
        and return average total regret '''
    best_mu = arms.best
    n_arms = arms.n_arms
    total_regret = 0
    policy_backup = copy.deepcopy(policy)

    plots = np.zeros_like(draw_points)
    for n in range(N):
        if n % 10 == 0:
            print n,'of', N,'...'
        this_regret = 0
        policy = copy.deepcopy(policy_backup)
        history = [[0, 0] for _ in range(n_arms)]
        for t in range(T):
            picked = policy.pick(n_arms, history)
            reward = arms.pull(picked)
            history[picked][0] += reward
            history[picked][1] += 1
            this_regret += best_mu - arms.mus[picked]
            if t in draw_points:
                plots[draw_points.index(t)] += this_regret
        total_regret += this_regret 
    plots = plots / N
    print total_regret / N
    return plots

def argmax(s):
    ''' return the first index corresponding to the max element '''
    return s.index(max(s))


class Policy():
    def __init__(self):
        pass

    def pick(self, n_arms, history, to_pick=[]):
        ''' to_pick stores the future picks '''
        pass


class RandomPick(Policy):
    def pick(self, n_arms, history):
        return random.choice(range(n_arms))

    
class BatchRandomPick(Policy):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def pick(self, n_arms, history, to_pick=[]):
        if not to_pick:
            to_pick += [random.choice(range(n_arms))] * self.batch_size
        return to_pick.pop()


class EpsGreedy(Policy):
    def __init__(self, eps):
        self.eps = eps
    def pick(self, n_arms, history):
        if random.random() < self.eps:
            return random.choice(range(n_arms))
        for i, [_, n] in enumerate(history):
            if n == 0:
                return i
        return argmax([r / n for r, n in history])


class UCB(Policy):
    def __init__(self, delta):
        self.delta = delta
    def pick(self, n_arms, history):
        for i, [_, n] in enumerate(history):
            if n == 0:
                return i
        t = sum(n for _, n in history)
        ucb = [r / n + math.sqrt(self.delta * math.log(t) / n) for r, n in history]
        return argmax(ucb)

    
class BatchUCB(Policy):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
    def pick(self, n_arms, history, to_pick=[]):
        if to_pick:
            return to_pick.pop()
        for i, [_, n] in enumerate(history):
            if n == 0:
                return i
        t = sum(n for _, n in history)
        ucb = [r / n + math.sqrt(math.log(t) / n) for r, n in history]
        to_pick += [argmax(ucb)] * self.batch_size
        return to_pick.pop()
    

class Thompson(Policy):
    def pick(self, n_arms, history):
        # list of (# success, # failure)
        S_F = [(arm_record[0], arm_record[1] - arm_record[0]) for arm_record in history]
        probs = [beta(s + 1,f + 1) for s, f in S_F]
        return argmax(probs)

# http://stackoverflow.com/questions/15204070/

from scipy.stats import norm, zscore
def sample_power_probtest(p1, p2, power=0.9, sig=0.05):
    
    z = norm.isf([sig / 2]) # two-sided t test
    zp = -norm.isf([power]) 
    d = p1 - p2
    s = 2 * ((p1 + p2) / 2) * (1 - (p1 + p2) / 2)
    n = s * ((zp + z) ** 2) / (d ** 2)
    return int(round(n[0]))

class ABTesting(Policy):

    def __init__(self, power=0.8, sig=0.05):
        from scipy.stats import norm
        self.power = power
        self.sig = sig
        self.best = None
        self.z_need = norm.isf(sig / 2) # 2-tail test
        self.eliminated = []
        self.to_pick = None

    def test_significance(self, history1, history2):
        [r1, n1] = history1
        [r2, n2] = history2
        p1 = r1 / n1
        p2 = r2 / n2
        try:
            z = (p1 - p2) / math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        except ZeroDivisionError:
            return 0
        if z > self.z_need:
            # first hand is better
            return 1
        if z < -self.z_need:
            # second hand is better
            return -1
        # cannot tell which one is better
        return 0

    def pick(self, n_arms, history):
        
        if self.to_pick is None:
            self.to_pick = list(range(n_arms))

        # if we have the best choice, pick it
        if self.best is not None:
            return self.best

        # pick the arm from to_pick if not eliminated
        while self.to_pick:
            pop = self.to_pick.pop()
            if pop in self.eliminated:
                continue
            else:
                return pop

        # to_pick is empty
        survived = [a for a in range(n_arms) if a not in self.eliminated]
        for a1 in survived:
            if a1 in self.eliminated:
                continue
            for a2 in survived:
                if a1 == a2 or a2 in self.eliminated:
                    continue
                test = self.test_significance(history[a1], history[a2])
                if test == 1:
                    self.eliminated.append(a2)
                    print 'Eliminated at',  sum([n for _, n in history])
                elif test == -1:
                    self.eliminated.append(a1)
                    print 'Eliminated at',  sum([n for _, n in history])

        survived = [a for a in range(n_arms) if a not in self.eliminated]
        if len(survived) == 1:
            self.best = survived[0]

        if self.best is not None:
            return self.best

        # one more round
        self.to_pick += survived
        return self.to_pick.pop()

def main():
    probs = [
        [0.2, 0.25, 0.3, 0.35, 0.4],
        [0.2, 0.2, 0.2, 0.2, 0.3],
        [0.2, 0.2, 0.2, 0.2, 0.21],
        [0.2] * 49 + [0.3],
    ]
    T = 10**5
    N = 1000
    for prob in probs:
        print 'T:', T, 'N:', N, 'probs:', prob
        a = Arms(prob)
        plt.figure()
        plt.title(('T: %d, N: %d, mu: ' % (T, N) + str(prob))[:60])
        draws = [i for i in range(0, T, 1000)]

        ucb_y = experiment_range(a, UCB(0.25), T, draws, N)
        ucb_line, = plt.plot(draws, ucb_y, lw=1, label='UCB', color='g')
        #ucb_y = experiment_range(a, UCB(1), T, draws, N)
        #ucb_line1, = plt.plot(draws, ucb_y, lw=1, label='UCB', color='g')
        t_y = experiment_range(a, Thompson(), T, draws, N)
        thompson_line, = plt.plot(draws, t_y, lw=1, label='Thompson', color='r')
        ab01_y = experiment_range(a, ABTesting(sig=0.01), T, draws, N)
        ab01_line, = plt.plot(draws, ab01_y, lw=1, label='A/B 0.01', color='black')
        plt.legend(handles=[ucb_line, thompson_line, ab01_line])
        fn = ('figures/N%d_' % N + str(prob)).replace('.', '').replace(' ', '').replace(',', '')[:25]
        plt.savefig(fn)
        print 'Saved', fn

    #return ucb_y, t_y, ab01_y

if __name__ == '__main__':
    main()
    '''
    plt.title('T: %d, N: %d, mu: (%.2f, %.2f)' % (T, N, probs[0], probs[1]))
    ucb_line, = plt.plot(draws, ucb_y, lw=1, label='UCB', color='g')
    thompson_line, = plt.plot(draws, t_y, lw=1, label='Thompson', color='r')
    ab01_line, = plt.plot(draws, ab01_y, lw=1, label='A/B 0.01', color='black')
    plt.legend(handles=[ucb_line, thompson_line, ab01_line])
    plt.show()
    '''
