'''
This is the Keras implementation of
'Domain-Adversarial Training of Neural Networks' by Y. Ganin

This allows domain adaptation (when you want to train on a dataset
with different statistics than a target dataset) in an unsupervised manner
by using the adversarial paradigm to punish features that help discriminate
between the datasets during backpropagation.

This is achieved by usage of the 'gradient reversal' layer to form
a domain invariant embedding for classification by an MLP.

The example here uses the 'MNIST-M' dataset as described in the paper.

Credits:
- Clayton Mellina (https://github.com/pumpikano/tf-dann) for providing
  a sketch of implementation (in TF) and utility functions.
- Yusuke Iwasawa (https://github.com/fchollet/keras/issues/3119#issuecomment-230289301)
  for Theano implementation (op) for gradient reversal.

Author: Vanush Vaswani (vanush@gmail.com)
'''

from keras.layers import Input, Dense, Dropout, Lambda
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import plot_model
from keras.utils import np_utils
import keras.backend as K

import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from keras.engine.training_utils import make_batches

import flipGradientTF

import pickle

# Helper functions
def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def batch_gen(batches, id_array, data, labels):
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = id_array[batch_start:batch_end]
        if labels is not None:
            yield data[batch_ids], labels[batch_ids]
        else:
            yield data[batch_ids]
        np.random.shuffle(id_array)


def evaluate_dann(X_test, y_test, batch_size):
    """Predict batch by batch."""
    size = batch_size // 2
    num_batches = X_test.shape[0] // size
    acc = 0
    for i in range(num_batches):
        _, prob = dann_model.predict_on_batch(X_test[i * size:i * size + size])
        predictions = np.argmax(prob, axis=1)
        actual = np.argmax(y_test[i * size:i * size + size], axis=1)
        acc += float(np.sum((predictions == actual))) / size
    return acc / num_batches


# Model parameters
batch_size = 128
nb_epoch_src, nb_epoch_trg = 75, 200
nb_classes = 6
nb_features = 601

#%% prepare input/output
# download data
data = pickle.load(open('data_gear_v2', "rb"))

x_data_s_pre = data[0]
x_data_t_pre = data[1]

data_flag = False
j = 0
for sublist_s, sublist_t in zip(x_data_s_pre, x_data_t_pre):  
    if not data_flag:
        x_data_s = np.copy(sublist_s.T)
        y_data_s = np.copy(j * np.ones(sublist_s.T.shape[0]))
        
        x_data_t = np.copy(sublist_t.T)
        y_data_t = np.copy(j * np.ones(sublist_t.T.shape[0]))
        data_flag = True
    else:
        x_data_s = np.vstack((x_data_s, sublist_s.T))
        y_data_s = np.hstack((y_data_s, j * np.ones(sublist_s.T.shape[0])))
        
        x_data_t = np.vstack((x_data_t, sublist_t.T))
        y_data_t = np.hstack((y_data_t, j * np.ones(sublist_t.T.shape[0])))
        
    j=j+1

# scale
scaler = StandardScaler()
scaler.fit(x_data_s)
x_data_s = scaler.transform(x_data_s)
x_data_t = scaler.transform(x_data_t)

# source
X_train, X_test, y_train, y_test = train_test_split(x_data_s, y_data_s,
                                                    test_size=0.2,
                                                    stratify=y_data_s)

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# target
XT_train, XT_test, yT_train, yT_test = train_test_split(x_data_t, y_data_t,
                                                        test_size=0.2,
                                                        stratify = y_data_t)

yT_train = np_utils.to_categorical(yT_train, nb_classes)
yT_test = np_utils.to_categorical(yT_test, nb_classes)

# make sure s and t have the size
number_of_rows = X_train.shape[0]
random_indices = np.random.choice(number_of_rows, size=(number_of_rows//(batch_size//2))*(batch_size//2), replace=False)
X_train = X_train[random_indices, :]
y_train = y_train[random_indices]

number_of_rows = XT_train.shape[0]
random_indices = np.random.choice(number_of_rows, size=(number_of_rows//(batch_size//2))*(batch_size//2), replace=False)
XT_train = XT_train[random_indices, :]
yT_train = yT_train[random_indices]

domain_labels = np.vstack([np.tile([0, 1], [batch_size // 2, 1]),
                           np.tile([1, 0], [batch_size // 2, 1])])

# Created mixed dataset for TSNE visualization
num_test = 500
combined_test_data = np.vstack([X_test[:num_test], XT_test[:num_test]])
combined_test_labels = np.vstack([y_test[:num_test], y_test[:num_test]])
combined_test_domain = np.vstack([np.tile([1, 0], [num_test, 1]),
                                  np.tile([0, 1], [num_test, 1])])

class DANNBuilder(object):
    def __init__(self):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        self.opt = SGD()

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Dense(256, activation='relu')(model_input)
        net = Dense(256, activation='relu')(net)
        net = Dropout(0.5)(net)
        self.domain_invariant_features = net
        return net

    def _build_classifier(self, model_input):
        net = Dense(128, activation='relu')(model_input)
        net = Dropout(0.5)(net)
        net = Dense(nb_classes, activation='softmax',
                    name='classifier_output')(net)
        return net

    def build_source_model(self, main_input, plot_model_cond=False):
        net = self._build_feature_extractor(main_input)
        net = self._build_classifier(net)
        model = Model(input=main_input, output=net)
        if plot_model_cond:
            plot_model(model, to_file='source_model.png', show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_dann_model(self, main_input, hp_lambda, plot_model_cond=False):
        net = self._build_feature_extractor(main_input)
        self.grl = flipGradientTF.GradientReversal(hp_lambda)
        branch = self.grl(net)
        branch = Dense(128, activation='relu')(branch)
        branch = Dropout(1./6.)(branch)
        branch = Dense(2, activation='softmax', name='domain_output')(branch)

        net = Lambda(lambda x: K.switch(K.learning_phase(),
                                        K.concatenate([x[:batch_size // 2], x[:batch_size // 2]], axis=0),
                                        x),
                     output_shape=lambda x: (x[0:]))(net)

        net = self._build_classifier(net)
        model = Model(input=main_input, output=[branch, net])
        if plot_model_cond:
            plot_model(model, to_file='dann_model.png', show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy',
                      'domain_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_tsne_model(self, main_input):
        '''Create model to output intermediate layer
        activations to visualize domain invariant features'''
        tsne_model = Model(input=main_input,
                           output=self.domain_invariant_features)
        return tsne_model


# main_input = Input(shape=(3, img_rows, img_cols), name='main_input')
main_input = Input(shape=(nb_features, ), name='main_input')

builder = DANNBuilder()
src_model = builder.build_source_model(main_input, plot_model_cond=True)
src_vis = builder.build_tsne_model(main_input)

hp_lambda = K.variable(1.0)
dann_model = builder.build_dann_model(main_input, hp_lambda, plot_model_cond=True)
dann_vis = builder.build_tsne_model(main_input)
print('Training source only model')
src_model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch_src, verbose=2,
              validation_data=(X_test, y_test))
print('Evaluating source samples on source-only model')
print('Accuracy: ', src_model.evaluate(X_test, y_test, verbose=0)[1])
print('Evaluating target samples on source-only model')
print('Accuracy: ', src_model.evaluate(XT_test, yT_test, verbose=0)[1])

# Broken out training loop for a DANN model.
src_index_arr = np.arange(X_train.shape[0])
target_index_arr = np.arange(XT_train.shape[0])

batches_per_epoch = len(X_train) / batch_size
num_steps = nb_epoch_trg * batches_per_epoch
j = 0

print('Training DANN model')

for i in range(nb_epoch_trg):

    batches = make_batches(X_train.shape[0], batch_size // 2)
    target_batches = make_batches(XT_train.shape[0], batch_size // 2)

    src_gen = batch_gen(batches, src_index_arr, X_train, y_train)
    target_gen = batch_gen(target_batches, target_index_arr, XT_train, None)

    losses = list()
    acc = list()

    for (xb, yb) in src_gen:

        # Update learning rate and gradient multiplier as described in
        # the paper.
        p = float(j) / num_steps
        l = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.01 / (1. + 10 * p)**0.75
        K.set_value(builder.grl.hp_lambda, l)
        K.set_value(builder.opt.lr, lr)

        if xb.shape[0] != batch_size // 2:
            continue

        try:
            xt = next(target_gen)
        except:
            # Regeneration
            target_gen = batch_gen(target_batches, target_index_arr, XT_train, None)
            xt = next(target_gen)

        # Concatenate source and target batch
        xb = np.vstack([xb, xt])
        yb = np.vstack([yb, yb])

        metrics = dann_model.train_on_batch({'main_input': xb},
                                            {'classifier_output': yb,
                                            'domain_output': domain_labels})
        j += 1

    acc_train = evaluate_dann(XT_train, yT_train, batch_size)
    acc_test = evaluate_dann(XT_test, yT_test, batch_size)
    print('Epoch ', i, 'Train Accuracy: ', acc_train, 'Test Accuracy: ', acc_test)

print('Evaluating source samples on source-only model')
print('Accuracy:', src_model.evaluate(X_train, y_train, verbose=0)[1])
print('Accuracy:', src_model.evaluate(X_test, y_test, verbose=0)[1])
print('Evaluating target samples on source-only model')
print('Accuracy:', src_model.evaluate(XT_test, yT_test, verbose=0)[1])

print('Evaluating source samples on DANN model')
print('Accuracy:', evaluate_dann(X_train, y_train, batch_size))
print('Accuracy:', evaluate_dann(X_test, y_test, batch_size))
print('Evaluating target samples on DANN model')
acc = evaluate_dann(XT_test, yT_test, batch_size)
print('Accuracy:', acc)
print('Visualizing output of domain invariant features')


src_embedding = src_vis.predict([combined_test_data])
src_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
tsne = src_tsne.fit_transform(src_embedding)

plot_embedding(tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'Source only')

dann_embedding = dann_vis.predict([combined_test_data])
dann_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
tsne = dann_tsne.fit_transform(dann_embedding)

plot_embedding(tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'DANN')

plt.show()
