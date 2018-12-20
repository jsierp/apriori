import csv
import numpy as np

class Rule():
    def __init__(self, A, C, supp, conf, lift, leverage):
        self.A = A
        self.C = C
        self.supp = supp
        self.conf = conf
        self.lift = lift
        self.leverage = leverage

    def __str__(self):
        return str(list(self.A)) + '->' + str(list(self.C)) + ' - support: %f, confidence: %f, lift: %f, leverage: %f' % (self.supp, self.conf, self.lift, self.leverage)

def apriori(T, min_supp=0.01, min_conf=0.1):
    """
    Implementation of apriori algorithm
    Description of the algorithm can be found here:
    http://www.ii.uni.wroc.pl/~lipinski/ED2018/lecture04.pdf
    """
    N = len(T)
    supp_count = min_supp*N
    d = max(x for t in T for x in t)
    L = dict({frozenset({}): 1})

    def supp(A):
        return L[A]

    def conf(A, C):
        return supp(A|C)/supp(A)

    def lift(A, C):
        return conf(A,C)/supp(C)

    def leverage(A, C):
        return supp(A|C)-supp(A)*supp(C)

    # Count one-element sets occurances
    count = [0] * d
    for t in T:
        for x in t:
            count[x-1] += 1
    l = dict((frozenset({x}), n/N) for x, n in zip(range(1, d+1), count) if n >= supp_count)

    while len(l):
        L.update(l)
        C = dict()
        for x1 in l:
            for x2 in l:
                if len(x1-x2) == len(x2-x1) == 1:
                    C[x1 | x2] = 0
        for t in T:
            for c in C:
                if c <= t:
                    C[c] += 1
        l = dict((c, C[c]/N) for c in C if C[c] >= supp_count)

    # L - sets with minimal acceptable support
    # L - dict - A: supp(A)

    s = [Rule(l, frozenset({}), supp(l), 1., 1., 0.) for l in L]
    S = []
    while len(s):
        S += s
        s_next = []
        
        for rule in s:
            for x in rule.A:
                # Check confidence of a rule A \ {x} -> C + {x}, using:
                # conf(A -> C) = supp(A + C)/supp(A)
                A = rule.A - {x}
                C = rule.C | {x}
                if conf(A,C) >= min_conf:
                    s_next.append(Rule(A, C, supp(A|C), conf(A,C), lift(A,C), leverage(A,C)))
        s = s_next
    return S

def load(path='retail.dat'):
    """
    Load file from given path.
    File is read as csv with a space as delimiter.
    """
    T = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            T.append({int(x) for x in row if x})
    return T

"""
Example usage:
T = load('retail.dat')
rules = apriori(T)
"""
