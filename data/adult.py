import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl

tfd = tf.contrib.distributions


def save_adult_datasets():
    adult_data = pd.read_csv('adult.data.txt', header=None, sep=', ').as_matrix()
    adult_test = pd.read_csv('adult.test.txt', header=None, sep=', ').as_matrix()

    def remove_question(df):
        idx = np.ones([df.shape[0]], dtype=np.int32)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                try:
                    if '?' in df[i, j]:
                        idx[i] = 0
                except TypeError:
                    pass
        df = df[np.nonzero(idx)]
        return df

    def remove_dot(df):
        for i in range(df.shape[0]):
            df[i, -1] = df[i, -1][:-1]
        return df

    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    adult_data = remove_question(adult_data)
    adult_test = remove_dot(remove_question(adult_test))
    adult_labels = gather_labels(adult_data)

    def transform_to_binary(df, labels):
        d = np.zeros([df.shape[0], 102])
        u = np.zeros([df.shape[0], 1])
        y = np.zeros([df.shape[0], 1])
        idx = 0
        for j in range(len(labels)):
            if type(labels[j]) is list:
                if len(labels[j]) > 2:
                    for i in range(df.shape[0]):
                        d[i, idx + int(labels[j].index(df[i, j]))] = 1
                    idx += len(labels[j])
                elif 'ale' in labels[j][0]:
                    for i in range(df.shape[0]):
                        u[i] = int(labels[j].index(df[i, j]))
                else:
                    for i in range(df.shape[0]):
                        y[i] = int(labels[j].index(df[i, j]))
            else:
                for i in range(df.shape[0]):
                    d[i, idx] = float(df[i, j] > labels[j])
                idx += 1
        return d.astype(np.bool), u.astype(np.bool), y.astype(np.bool)  # observation, protected, label

    adult_binary = transform_to_binary(adult_data, adult_labels)
    adult_test_binary = transform_to_binary(adult_test, adult_labels)

    with open('adult_binary.pkl', 'wb') as f:
        pkl.dump(adult_binary, f)
    with open('adult_test_binary.pkl', 'wb') as f:
        pkl.dump(adult_test_binary, f)


def create_adult_datasets(batch=64):
    with open('adult_binary.pkl', 'rb') as f:
        ab = pkl.load(f)
    with open('adult_test_binary.pkl', 'rb') as f:
        atb = pkl.load(f)
    adult_binary = tuple([a.astype(np.float32) for a in ab])
    adult_test_binary = tuple([a.astype(np.float32) for a in atb])
    train = tf.data.Dataset.from_tensor_slices(adult_binary).shuffle(adult_binary[0].shape[0]).batch(batch).prefetch(batch)
    test = tf.data.Dataset.from_tensor_slices(adult_test_binary).batch(batch).prefetch(batch)
    pu = tfd.Bernoulli(probs=np.mean(adult_binary[1]))
    return train, test, pu


if __name__ == '__main__':
    save_adult_datasets()
