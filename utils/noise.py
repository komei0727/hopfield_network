def make_noise(q, noise):
    import numpy as np
    q_noise = np.zeros((1,25), dtype = int)
    for i in range(25):
        rand = np.random.randint(100)
        if rand < noise:
            if q[0][i] == 1:
                q_noise[0][i] = -1
            else:
                q_noise[0][i] = 1
        else:
            q_noise[0][i] = q[0][i]
    return q_noise