import math
import random

class Arms():
    def __init__(self, mus):
        self.mus = mus
        self.n_arms = len(mus)
        self.best = max(mus)
        for mu in mus:
            assert 0 <= mu <= 1

    def __str__(self):
        return str(self.mus)

    def pull(self, idx):
        try:
            if random.random() < self.mus[idx]:
                return 1
            else:
                return 0
        except IndexError:
            raise


class Policy():
    def __init__(self):
        pass

    def pick(self, n_arms, history):
        pass


def experiment(arms, policy, T):
    ''' return simulated history and toal regret '''
    best_mu = arms.best
    n_arms = arms.n_arms
    history = [[0, 0] for _ in range(n_arms)]
    total_regret = 0

    for t in range(T):
        picked = policy.pick(n_arms, history)
        reward = arms.pull(picked)
        history[picked][0] += reward
        history[picked][1] += 1
        total_regret += best_mu - arms.mus[picked]

    return history, total_regret


def argmax(s):
    return s.index(max(s))


class RandomPick(Policy):
    def pick(self, n_arms, history):
        return random.choice(range(n_arms))


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
    def pick(self, n_arms, history):
        for i, [_, n] in enumerate(history):
            if n == 0:
                return i
        t = sum(n for _, n in history)
        ucb = [r / n + (math.log(t) / n) ** 0.5 for r, n in history]
        return argmax(ucb)


a = Arms([0.4, 0.6])
T = 100000
experiment(a, RandomPick(), T)
experiment(a, EpsGreedy(0.1), T)
experiment(a, UCB(), T)
