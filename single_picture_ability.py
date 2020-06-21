import numpy as np
from utils.noise import make_noise
from utils.recollection import recollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#ノイズと閾値の設定
noise = 20
thete = 0
N = 100
epoch =1000
similar1_all = np.zeros((1000,100))
q_1_noise_all = np.zeros((1000, 25))
q_1_rec_all = np.zeros((1000, 25))

#画像の読み込み
Q_1 = np.loadtxt("picture/picture_6.csv", delimiter = ",", dtype = int)
q_1 = np.array([Q_1.flatten()])

#重み行列の作成
W = np.dot(q_1.T,q_1) 

#Wの対角成分をゼロに
W_diag = np.diag(W)
W_diag.flags.writeable = True
np.putmask(W_diag, W_diag > 0, 0)

recall_performance_1 = 0
correct_answer_1 = 0

for e in range(epoch):
    q_1_noise_all[e] = make_noise(q_1, noise)

    similar1_all[e], q_1_rec_all[e] = recollection(q_1[0], q_1_noise_all[e], W, thete, N)
    
    recall_performance_1 += similar1_all[e][-1]

    if similar1_all[e][-1] == 1.0:
        correct_answer_1 += 1

recall_performance_average_1 = recall_performance_1 / 1000

correct_answer_rate_1 = correct_answer_1 / 1000

print('<picture1>\n類似度の全試行平均:{} 正答率:{}'.format(recall_performance_average_1, correct_answer_rate_1))
