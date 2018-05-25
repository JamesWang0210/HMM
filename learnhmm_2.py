# import sys
import numpy as np

# train_in = sys.argv[1]
# index_w = sys.argv[2]
# index_t = sys.argv[3]
# prior = sys.argv[4]
# emit = sys.argv[5]
# trans = sys.argv[6]


class Learning:
    def __init__(self):
        self.X = []  # All words for each line
        self.Y = []  # All tags for each line

        self.index_w = {}  # Read Word Index from index_to_word.txt
        self.index_t = {}  # Read Tag Index from index_to_tag.txt

        self.prior = np.ones(1)  # Prior Matrix
        self.trans = np.ones(1)  # Transition Matrix
        self.emit = np.ones(1)  # Emission Matrix

    def readIndex(self):
        id_w = open("toydata/toy_index_to_word.txt", 'r+').read().splitlines()
        # id_w = open(index_w, 'r+').read().splitlines()
        id_t = open("toydata/toy_index_to_tag.txt", 'r+').read().splitlines()
        # id_t = open(index_t, 'r+').read().splitlines()
        for i in range(0, len(id_w)):
            self.index_w[id_w[i]] = i+1
        for j in range(0, len(id_t)):
            self.index_t[id_t[j]] = j+1

    def readFile(self):
        train = open("toydata/toytrain.txt", 'r+').read().splitlines()
        # train = open(train_in, 'r+').read().splitlines()

        for i in range(0, len(train)):
            line = train[i].split(' ')
            line_x = []
            line_y = []
            for j in range(0, len(line)):
                word = line[j].split('_')
                line_x.append(self.index_w[word[0]])
                line_y.append(self.index_t[word[1]])
            self.X.append(line_x)
            self.Y.append(line_y)

    def findPrior(self):
        self.prior = np.ones(len(self.index_t), dtype=np.float64)
        for i in range(0, len(self.Y)):
            j = self.Y[i][0]
            self.prior[j-1] += 1

        self.prior = self.prior/(np.sum(self.prior))
        np.savetxt("hmmprior_2.txt", self.prior)
        # np.savetxt(prior, self.prior)

    def findTrans(self):
        self.trans = np.ones((len(self.index_t), len(self.index_t)), dtype=np.float64)
        for i in range(0, len(self.Y)):
            for j in range(0, len(self.Y[i])-1):
                a = self.Y[i][j]
                b = self.Y[i][j+1]
                self.trans[a-1][b-1] += 1

        for i in range(0, len(self.trans)):
            self.trans[i] = self.trans[i] / (np.sum(self.trans[i]))

        np.savetxt("hmmtrans_2.txt", self.trans)
        # np.savetxt(trans, self.trans)

    def findEmit(self):
        self.emit = np.ones((len(self.index_t), len(self.index_w)), dtype=np.float64)
        for i in range(0, len(self.Y)):
            for j in range(0, len(self.Y[i])):
                a = self.Y[i][j]
                b = self.X[i][j]
                self.emit[a-1][b-1] += 1

        for i in range(0, len(self.emit)):
            self.emit[i] = self.emit[i] / (np.sum(self.emit[i]))

        np.savetxt("hmmemit_2.txt", self.emit)
        # np.savetxt(emit, self.emit)

# Learning Process
e = Learning()

e.readIndex()
# print e.index_w
# print e.index_t
# print len(e.index_w)
# print len(e.index_t)
# print e.index_t['O']
# print e.index_w[5]
# print type(e.index_w[5])
# print e.index_w[5] == "'"

e.readFile()
print e.X
print e.Y
# print e.X[239][1]
# print len(e.X) == len(e.Y)

e.findPrior()
# print e.prior

e.findTrans()
# print e.trans
# print len(e.trans)
# print e.trans[0]

e.findEmit()
# print e.emit
