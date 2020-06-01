def recollection(q, q_noise, W, thete, N):
    import numpy as np
    import copy
    S = np.zeros(100)
    q_rec = copy.copy(q_noise)
    for n in range(N): 
        s = 0
        for p in range(25):
            if q[p] == q_rec[p]:
                s += 1
        s = s / 25
        S[n] = s
        i = np.random.randint(25)
        if np.dot(W[i], q_rec) - thete >= 0:
            q_rec[i] = 1
        else:
            q_rec[i] = -1

    return S, q_rec