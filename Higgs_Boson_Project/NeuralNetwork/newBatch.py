import numpy as np

class Dataset:

    def __init__(self, data, labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        self._labels = labels
        pass


    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size, shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
            self._labels = self.labels[idx]

        if start + batch_size <= self._num_examples:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]
        else:
            print('wrong batch size given')
            return 0, 0

        # go to the next batch
        # if start + batch_size > self._num_examples:
        #     self._epochs_completed += 1
        #     rest_num_examples = self._num_examples - start
        #     data_rest_part = self.data[start:self._num_examples]
        #     idx0 = np.arange(0, self._num_examples)  # get all possible indexes
        #     np.random.shuffle(idx0)  # shuffle indexes
        #     self._data = self.data[idx0]  # get list of `num` random samples
        #     self._labels = self.labels[idx0]  # get list of `num` random samples
        #
        #     start = 0
        #     self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
        #     end = self._index_in_epoch
        #     data_new_part = self._data[start:end]
        #     return np.concatenate((data_rest_part, data_new_part), axis=0)
        # else:
        #     self._index_in_epoch += batch_size
        #     end = self._index_in_epoch
        #     return self._data[start:end], self._labels[start:end]

