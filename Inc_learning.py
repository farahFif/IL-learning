from collections import Counter
import os
import numpy as np
from collections import Counter
import os
import functools
import operator
from operator import itemgetter


class Incremental_setting():
    def __init__(self, dataset, nb_cl_phase=3, cur_phase=0):
        self.data = dataset
        # self.nb_phases = nb_phases
        self.cur_phase = cur_phase
        self.active_classes = []
        self.old_classes = []
        self.order = []
        self.nb_cl_phase = nb_cl_phase  # number of classes per phases
        self.limit_samp_per_phase_train = 15000  # including 5000 for the new class
        # self.limit_samp_per_phase_test =
        self.idx_per_class = {}
        self.new_test = [self.data.test_data]
        # self.new_test_labels = self.data.labels_test
        self.def_inc_per_classes()
        self.old_data = []

    def def_inc_per_classes(self):
        #if os.path.exists('order.npy'):
        #    self.order = np.load("order.npy", allow_pickle=True)
        if True:
            np.random.seed(10)
            mix = np.arange(self.data.classes)
            np.random.shuffle(mix)
            print(mix)
            temp = []
            order = {}
            cpt = 0
            for i, j in enumerate(mix):
                temp.append(j)
                if (i + 1) % self.nb_cl_phase == 0:
                    order[cpt] = temp
                    cpt += 1
                    temp = []
            if len(temp) != 0:
                order[i] = temp
            for i in order:
                self.order.append(order[i])
            self.order = np.asarray(self.order)
            np.save('order.npy', self.order)
        print("Classes order ------ ", self.order)

    def add_class(self, cls):
        if cls not in self.active_classes:
            self.active_classes.append(cls)

    def get_cur_data(self, ph):
        dat = []
        test = []
        for i in self.order[ph]:
            selection = self.data.labels.argmax(axis=1) == i
            selection2 = self.data.labels_test.argmax(axis=1) == i
            dat.extend(list(zip(self.data.labels[selection], self.data.train_data.data[selection])))
            test.extend(list(zip(self.data.labels_test[selection2], self.data.test_data.data[selection2])))
        return dat, test

    def adj_old_data(self, phase, limit=1000):
        prev = functools.reduce(operator.iconcat, self.order[:phase], [])  # classes
        print("previous classes are ", prev, " current phase is ", phase)
        old = []
        old_test = []
        Limit_per_class = limit // len(prev)
        old_test = []
        for i in prev:
            p = [k for k in self.old_data if np.array(k[0]).argmax() == i]
            old.extend(p[:Limit_per_class])
            selection = self.data.labels_test.argmax(axis=1) == i
            old_test.extend(list(zip(self.data.labels_test[selection],self.data.test_data.data[selection])))
        return old , old_test

    def adj_old_data_eq(self, phase):
        'Adjust old data with all equal length = 1/3'
        prev = functools.reduce(operator.iconcat, self.order[:phase], [])  # classes
        print("previous classes are ", prev, " current is ", phase)
        old = []
        Limit_per_class = 150  # Approximately 1/3 FOR EACH class
        for i in prev:
            p = [k for k in self.old_data if np.array(k[0]).argmax() == i]
            old.extend(p[:Limit_per_class])
        return old

    def get_approp_data(self, phase,limit_new=False,limit_new_sp=350):
        print(" --------------- PHASE  ------------", phase)

        new_classes = self.order[phase]
        for i in new_classes:
            self.add_class(i)
        print("Active classes : ", self.active_classes)

        cur_data_train, cur_data_test = self.get_cur_data(phase)
        total_test = cur_data_test
        if limit_new:
            # If the number of samples of the new data is limited
            idc = np.random.choice(len(cur_data_train), limit_new_sp * len(self.order[phase]))
            total_data = list(itemgetter(*idc)(cur_data_train))  # get only third
        else:
            total_data = cur_data_train

        if phase != 0:
            old, old_test = self.adj_old_data(phase)
            #total_data += old
            total_test += old_test
            print("ooooooooooooooooold",np.unique([np.array(k[0]).argmax() for k in old_test]))
            #print('cuuuuuuuur', type(old[0]))



        h = [np.asarray(x[0]).argmax() for x in total_data]
        jj = [np.asarray(x[0]).argmax() for x in total_test]

        self.old_data = total_data
        print(" NB samples per class", Counter(h) , Counter(jj))
        print('Nb classes ------ ',len(np.unique(h)),len(np.unique(jj)))
        #self.new_test.append(cur_data_test)

        #self.cur_labels = [r[0].argmax() for r in total_data]  # a modifier ce n est pas bcp optimisé
        #print('toootal data type', type(total_data))
        #print("teeeeest",total_data[0])
        return total_data, total_test

    def get_all_data(self):
        dat = []
        for i in range(10):
            selection = self.data.labels.argmax(axis=1) == i
            dat.extend(list(zip(self.data.labels[selection], self.data.train_data.data[selection])))
        return dat