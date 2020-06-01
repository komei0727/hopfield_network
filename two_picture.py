import numpy as np
from utils.noise import make_noise
from utils.recollection import recollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#ノイズと閾値の設定
noise = 5
thete = 0
N = 100
epoch =1000
similar1_all = np.zeros((1000,100))
q_1_noise_all = np.zeros((1000, 25))
q_1_rec_all = np.zeros((1000, 25))
similar2_all = np.zeros((1000,100))
q_2_noise_all = np.zeros((1000, 25))
q_2_rec_all = np.zeros((1000, 25))

#画像の読み込み
Q_1 = np.loadtxt("picture/picture_3.csv", delimiter = ",", dtype = int)
q_1 = np.array([Q_1.flatten()])
Q_2 = np.loadtxt("picture/picture_4.csv", delimiter = ",", dtype = int)
q_2 = np.array([Q_2.flatten()])

#重み行列の作成
W = np.dot(q_1.T,q_1) + np.dot(q_2.T,q_2)

#Wの対角成分をゼロに
W_diag = np.diag(W)
W_diag.flags.writeable = True
np.putmask(W_diag, W_diag > 0, 0)

recall_performance_1 = 0
recall_performance_2 = 0
correct_answer_1 = 0
correct_answer_2 = 0

for e in range(epoch):
    q_1_noise_all[e] = make_noise(q_1, noise)
    q_2_noise_all[e] = make_noise(q_2, noise)

    similar1_all[e], q_1_rec_all[e] = recollection(q_1[0], q_1_noise_all[0], W, thete, N)
    similar2_all[e], q_2_rec_all[e] = recollection(q_2[0], q_2_noise_all[0], W, thete, N)
    
    recall_performance_1 += similar1_all[e][-1]
    recall_performance_2 += similar2_all[e][-1]

    if similar1_all[e][-1] == 1.0:
        correct_answer_1 += 1
    if similar2_all[e][-1] == 1.0:
        correct_answer_2 += 1

recall_performance_average_1 = recall_performance_1 / 1000
recall_performance_average_2 = recall_performance_2 / 1000
correct_answer_rate_1 = correct_answer_1 / 1000
correct_answer_rate_2 = correct_answer_2 / 1000

print('<picture1>\n類似度の全試行平均:{} 想起性能:{}\n<picture2>\n類似度の全試行平均:{} 想起性能:{}'.format(recall_performance_average_1, correct_answer_rate_1, recall_performance_average_2, correct_answer_rate_2))