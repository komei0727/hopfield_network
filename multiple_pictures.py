import numpy as np
from utils.noise import make_noise
from utils.recollection import recollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#ノイズと閾値の設定
thete = 0
N = 100
epoch =1000


#画像の読み込み
Q_1 = np.loadtxt("picture/picture_1.csv", delimiter = ",", dtype = int)
q_1 = np.array([Q_1.flatten()])
Q_2 = np.loadtxt("picture/picture_2.csv", delimiter = ",", dtype = int)
q_2 = np.array([Q_2.flatten()])
Q_3 = np.loadtxt("picture/picture_3.csv", delimiter = ",", dtype = int)
q_3 = np.array([Q_3.flatten()])
Q_4 = np.loadtxt("picture/picture_4.csv", delimiter = ",", dtype = int)
q_4 = np.array([Q_4.flatten()])
Q_5 = np.loadtxt("picture/picture_5.csv", delimiter = ",", dtype = int)
q_5 = np.array([Q_5.flatten()])
Q_6 = np.loadtxt("picture/picture_6.csv", delimiter = ",", dtype = int)
q_6 = np.array([Q_6.flatten()])

q = np.zeros([6,25])
q[0] = q_1[0]
q[1] = q_2[0]
q[2] = q_3[0]
q[3] = q_4[0]
q[4] = q_5[0]
q[5] = q_6[0]

#２種類の画像に関して
for n in range(4):
    noise = (n+1)*5
    print('noise{}%'.format(noise))
    W = np.zeros((6,6,25,25))
    W[0][1] = np.dot(q_1.T,q_1) + np.dot(q_2.T,q_2)
    W[0][2] = np.dot(q_1.T,q_1) + np.dot(q_3.T,q_3)
    W[0][3] = np.dot(q_1.T,q_1) + np.dot(q_4.T,q_4)
    W[0][4] = np.dot(q_1.T,q_1) + np.dot(q_5.T,q_5)
    W[0][5] = np.dot(q_1.T,q_1) + np.dot(q_6.T,q_6)
    W[1][2] = np.dot(q_2.T,q_2) + np.dot(q_3.T,q_3)
    W[1][3] = np.dot(q_2.T,q_2) + np.dot(q_4.T,q_4)
    W[1][4] = np.dot(q_2.T,q_2) + np.dot(q_5.T,q_5)
    W[1][5] = np.dot(q_2.T,q_2) + np.dot(q_6.T,q_6)
    W[2][3] = np.dot(q_3.T,q_3) + np.dot(q_4.T,q_4)
    W[2][4] = np.dot(q_3.T,q_3) + np.dot(q_5.T,q_5)
    W[2][5] = np.dot(q_3.T,q_3) + np.dot(q_6.T,q_6)
    W[3][4] = np.dot(q_4.T,q_4) + np.dot(q_5.T,q_5)
    W[3][5] = np.dot(q_4.T,q_4) + np.dot(q_6.T,q_6)
    W[4][5] = np.dot(q_5.T,q_5) + np.dot(q_6.T,q_6)
    for i in range(6):
        for j in range(6):
            if i < j:
                print('[picture{] - picture{}'.format(i+1,j+1))
                W_diag = np.diag(W[i][j])
                W_diag.flags.writeable = True
                np.putmask(W_diag, W_diag > 0, 0)

                similar_all = np.zeros((6,1000,100))
                q_noise_all = np.zeros((6,1000, 25))
                q_rec_all = np.zeros((6,1000, 25))

                recall_performance = np.zeros(6)
                correct_answer = np.zeros(6)

                for e in range(epoch):
                    q_noise_all[i][e] = make_noise(np.array([q[i]]), noise)
                    q_noise_all[j][e] = make_noise(np.array([q[j]]), noise)

                    similar_all[i][e], q_rec_all[i][e] = recollection(q[i], q_noise_all[i][e], W[i][j], thete, N)
                    similar_all[j][e], q_rec_all[j][e] = recollection(q[j], q_noise_all[j][e], W[i][j], thete, N)
                    
                    recall_performance[i] += similar_all[i][e][-1]
                    recall_performance[j] += similar_all[j][e][-1]

                    if similar_all[i][e][-1] == 1.0:
                        correct_answer[i] += 1
                    if similar_all[j][e][-1] == 1.0:
                        correct_answer[j] += 1

                recall_performance_average = recall_performance / 1000
                correct_answer_rate = correct_answer / 1000

                print('<picture{}>\n類似度の全試行平均:{} 想起性能:{}'.format(i+1,recall_performance_average[i], correct_answer_rate[i]))
                print('<picture{}>\n類似度の全試行平均:{} 想起性能:{}'.format(j+1,recall_performance_average[j], correct_answer_rate[j]))



