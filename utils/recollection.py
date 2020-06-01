def recollection(q, q_noise, W, thete, N):
    import numpy as np
    S = np.zeros(100)
    for n in range(N): 
        S[n] = np.dot(q_noise, q) / 25
        i = np.random.randint(25)
        if np.dot(W[i], q_noise) - thete >= 0:
            q_noise[i] = 1
        else:
            q_noise[i] = -1

    return S, q_noise