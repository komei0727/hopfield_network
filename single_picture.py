import numpy as np
from utils.noise import make_noise
from utils.recollection import recollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#ノイズと閾値の設定
noise = 20
thete = 0
N = 100

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

#画像を想起する
similar, q_1_rec = recollection(q_1[0], q_1_noise[0], W, thete, N)

#結果のグラフを表示する
Q_1_noise = q_1_noise.reshape(5,5)
Q_1_rec = q_1_rec.reshape(5,5)
x = np.linspace(1,100,100)
fig = plt.figure(figsize = (12,8))
gs = gridspec.GridSpec(2,3)
ax1 = fig.add_subplot(gs[0,0])
ax1.set_title('input')
ax1.imshow(Q_1, cmap="gray")
ax2 = fig.add_subplot(gs[0,1])
ax2.set_title('with noise')
ax2.imshow(Q_1_noise, cmap="gray")
ax3 = fig.add_subplot(gs[0,2])
ax3.set_title('output')
ax3.imshow(Q_1_rec, cmap="gray")
ax4 = fig.add_subplot(gs[1,:])
ax4.set_title('degree of similarity')
ax4.plot(x, similar)

plt.show()