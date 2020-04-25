import numpy as np
from utils.noise import make_noise

#ノイズと閾値の設定
noise = 20
thete = 0

#画像の読み込み
Q_1 = np.loadtxt("picture/picture_1.csv", delimiter = ",", dtype = int)
q_1 = np.array([Q_1.flatten()])

#重み行列の作成
W = np.dot(q_1.T,q_1) 

#Wの対角成分をゼロに
W_diag = np.diag(W)
W_diag.flags.writeable = True
np.putmask(W_diag, W_diag > 0, 0)

#元画像にノイズを加える
q_1_noise = make_noise(q_1, noise)

for n in range(100):
    similar = np.dot(q_1_noise[0], q_1[0]) / 25
    print(similar)
    i = np.random.randint(25)
    if np.dot(W[i], q_1_noise[0]) - thete >= 0:
        q_1_noise[0][i] = 1
    else:
        q_1_noise[0][i] = -1