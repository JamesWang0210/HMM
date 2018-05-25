# import sys
import numpy as np

# test_in = sys.argv[1]
# index_w = sys.argv[2]
# index_t = sys.argv[3]
# prior = sys.argv[4]
# emit = sys.argv[5]
# trans = sys.argv[6]
# predict = sys.argv[7]


class HMM:
    def __init__(self):
        self.index_w = {}  # Read Word Index from index_to_word.txt
        self.index_t = {}  # Read Tag Index from index_to_tag.txt

        self.X = []  # All words for each line
        self.Y = []  # All tags for each line

        self.prior = np.zeros(1)  # Prior Matrix
        self.trans = np.zeros(1)  # Transition Matrix
        self.emit = np.zeros(1)  # Emission Matrix

        self.alpha = np.zeros(1)  # Alpha Matrix for each line
        self.beta = np.zeros(1)  # Beta Matrix for each line

        self.Y_hat = []  # Predicted tags for each line

        self.log = 0  # log likelihood for each sequence

    def readIndex(self):
        id_w = open("toydata/toy_index_to_word.txt", 'r+').read().splitlines()
        # self.id_w = open(index_w, 'r+').read().splitlines()
        id_t = open("toydata/toy_index_to_tag.txt", 'r+').read().splitlines()
        # self.id_t = open(index_t, 'r+').read().splitlines()
        for i in range(0, len(id_w)):
            self.index_w[id_w[i]] = i + 1
        for j in range(0, len(id_t)):
            self.index_t[id_t[j]] = j + 1

    def readFile(self):
        test = open("toydata/toytest.txt", 'r+').read().splitlines()
        # test = open(test_in, 'r+').read().splitlines()

        for i in range(0, len(test)):
            line = test[i].split(' ')
            line_x = []
            line_y = []
            for j in range(0, len(line)):
                word = line[j].split('_')
                line_x.append(self.index_w[word[0]])
                line_y.append(self.index_t[word[1]])
            self.X.append(line_x)
            self.Y.append(line_y)

    def getMatrix(self):
        self.prior = np.genfromtxt("hmmprior_2.txt", dtype=np.float64)
        # self.prior = np.genfromtxt(prior, dtype=np.float64)
        self.trans = np.genfromtxt("hmmtrans_2.txt", dtype=np.float64)
        # self.trans = np.genfromtxt(trans, dtype=np.float64)
        self.emit = np.genfromtxt("hmmemit_2.txt", dtype=np.float64)
        # self.emit = np.genfromtxt(emit, dtype=np.float64)

    def getPredict(self):
        for i in range(0, len(self.X)):
            line_y_hat = []
            sentence = ""

            b_col = self.X[i][0] - 1
            alpha = np.transpose(np.array([self.prior * self.emit[:, b_col]])) # Alpha_1
            self.alpha = alpha  # Append Alpha_1 to Alpha Matrix

            beta = np.ones((len(self.prior),1))  # Beta_1
            self.beta = beta  # Append Beta_T to Beta Matrix
            for j in range(1, len(self.X[i])):
                b_col = self.X[i][j] - 1
                alpha = np.transpose(np.array([self.emit[:, b_col]])) * np.dot(np.transpose(self.trans), alpha)
                self.alpha = np.concatenate((self.alpha, alpha), axis=1)

                b_col = self.X[i][len(self.X[i])-j] - 1
                beta = np.dot(self.trans, (np.transpose(np.array([self.emit[:, b_col]])) * beta))
                self.beta = np.append(self.beta, beta, axis=1)

                # print self.alpha
                # print self.beta

            self.log = np.log(np.sum(self.alpha[:, -1]))

            for k in range(0, len(self.X[i])):
                prob = self.alpha[:, k] * self.beta[:, len(self.X[i])-1-k]
                new_id = np.argmax(prob, axis=0) + 1
                line_y_hat.append(new_id)

            self.Y_hat.append(line_y_hat)  # All predicted tags

            for m in range(0, len(self.X[i])):
                for key, value in self.index_w.iteritems():
                    if self.X[i][m] == value:
                        word = key
                        sentence += word + '_'
                for key, value in self.index_t.iteritems():
                    if line_y_hat[m] == value:
                        tag = key
                        if m != len(self.X[i]) - 1:
                            sentence += tag + ' '
                        else:
                            sentence += tag

            if i == 0:
                f = open("predictedtest_2.txt", "w")
                # f = open(predict, "w")
            else:
                f = open("predictedtest_2.txt", "a")
                # f = open(predict, "a")

            f.write(sentence + '\n')
            f.close()


h = HMM()

h.readIndex()

h.readFile()
# print h.X
# print h.Y

h.getMatrix()
# print h.prior
# print h.trans
# print h.emit
# print len(h.prior)
# print len(h.trans)
# print len(h.emit)

# b_col = h.X[0][0]
# # print h.trans[:, 4]
# # print type(h.prior)
# alpha = np.transpose(np.array([h.prior * h.emit[:, b_col]]))
# h.alpha = alpha
# print h.alpha
# b_col = h.X[0][1] - 1
# c = np.dot(np.transpose(h.trans), alpha)
# # print c
# # print h.emit[:, b_col]
# alpha = np.transpose(np.array([h.emit[:, b_col]]))*c
# print alpha
# # print alpha.ndim
# h.alpha = np.concatenate((h.alpha, alpha), axis=1)
# print h.alpha
h.getPredict()
# print len(h.X) == len(h.Y_hat)
# print len(h.Y_hat)
# print h.Y_hat[1622]
print h.log
