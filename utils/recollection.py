def recollection(q, q_noise, W, thete, N):
    import numpy as np
    import copy
    S = np.zeros(100)
    q_rec = copy.copy(q_noise)
    for n in range(N): 
        S[n] = np.dot(q_rec, q) / 25
        i = np.random.randint(25)
        if np.dot(W[i], q_rec) - thete >= 0:
            q_rec[i] = 1
        else:
            q_rec[i] = -1

    return S, q_rec