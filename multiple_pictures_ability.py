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

#重み行列の作成
W =  np.dot(q_3.T,q_3) + np.dot(q_5.T,q_5)

#Wの対角成分をゼロに
W_diag = np.diag(W)
W_diag.flags.writeable = True
np.putmask(W_diag, W_diag > 0, 0)



for n in range(4):
    noise = (n+1)*5
    print('noise{}%'.format(noise))
    similar_all = np.zeros((6,1000,100))
    q_noise_all = np.zeros((6,1000, 25))
    q_rec_all = np.zeros((6,1000, 25))

    recall_performance = np.zeros(6)
    correct_answer = np.zeros(6)

    for e in range(epoch):
        q_noise_all[0][e] = make_noise(q_1, noise)
        q_noise_all[1][e] = make_noise(q_2, noise)
        q_noise_all[2][e] = make_noise(q_3, noise)
        q_noise_all[3][e] = make_noise(q_4, noise)
        q_noise_all[4][e] = make_noise(q_5, noise)
        q_noise_all[5][e] = make_noise(q_6, noise)

        similar_all[0][e], q_rec_all[0][e] = recollection(q[0], q_noise_all[0][e], W, thete, N)
        similar_all[1][e], q_rec_all[1][e] = recollection(q[1], q_noise_all[1][e], W, thete, N)            
        similar_all[2][e], q_rec_all[2][e] = recollection(q[2], q_noise_all[2][e], W, thete, N)
        similar_all[3][e], q_rec_all[3][e] = recollection(q[3], q_noise_all[3][e], W, thete, N)
        similar_all[4][e], q_rec_all[4][e] = recollection(q[4], q_noise_all[4][e], W, thete, N)            
        similar_all[5][e], q_rec_all[5][e] = recollection(q[5], q_noise_all[5][e], W, thete, N)
            
        recall_performance[0] += similar_all[0][e][-1]
        recall_performance[1] += similar_all[1][e][-1]
        recall_performance[2] += similar_all[2][e][-1]
        recall_performance[3] += similar_all[3][e][-1]
        recall_performance[4] += similar_all[4][e][-1]
        recall_performance[5] += similar_all[5][e][-1]


        if similar_all[0][e][-1] == 1.0:
            correct_answer[0] += 1
        if similar_all[1][e][-1] == 1.0:
            correct_answer[1] += 1
        if similar_all[2][e][-1] == 1.0:
            correct_answer[2] += 1
        if similar_all[3][e][-1] == 1.0:
            correct_answer[3] += 1
        if similar_all[4][e][-1] == 1.0:
            correct_answer[4] += 1
        if similar_all[5][e][-1] == 1.0:
            correct_answer[5] += 1

    recall_performance_average = recall_performance / 1000

    correct_answer_rate = correct_answer / 1000

    print('<picture1>\n類似度の全試行平均:{} 正答率:{}'.format(recall_performance_average[0], correct_answer_rate[0]))
    print('<picture2>\n類似度の全試行平均:{} 正答率:{}'.format(recall_performance_average[1], correct_answer_rate[1]))
    print('<picture3>\n類似度の全試行平均:{} 正答率:{}'.format(recall_performance_average[2], correct_answer_rate[2]))
    print('<picture4>\n類似度の全試行平均:{} 正答率:{}'.format(recall_performance_average[3], correct_answer_rate[3]))
    print('<picture5>\n類似度の全試行平均:{} 正答率:{}'.format(recall_performance_average[4], correct_answer_rate[4]))
    print('<picture6>\n類似度の全試行平均:{} 正答率:{}'.format(recall_performance_average[5], correct_answer_rate[5]))

