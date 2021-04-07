import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers import Conv1D, Flatten
from keras.layers import LeakyReLU, BatchNormalization, MaxPooling1D 
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import plot_model
from keras import utils
import keras.backend as K

import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from keras.engine.training_utils import make_batches

import flipGradientKeras

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

    plt.show()


def batch_gen(batches, id_array, data, labels):
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = id_array[batch_start:batch_end]
        if labels is not None:
            yield data[batch_ids], labels[batch_ids]
        else:
            yield data[batch_ids]
        np.random.shuffle(id_array)

# Model parameters
batch_size = 256
nb_epoch_src, nb_epoch_trg = 200, 50
nb_classes = 7
nb_features = 320
mu_0 = 0.01
alpha = 10.
beta = 0.75
gamma = 10.

alpha_relu = 0.01
dropout_rate = 1./7.
dropout_rate_dann = 1./7.
learning_rate_source=2e-5
learning_rate_dann=5e-6
min_learning_rate=5e-6

dataset_num = 'dataset_num_0_4.pickle'
dataset_test = 'data_test.pickle'

#%% prepare input/output
with open(dataset_num, 'rb') as f:
    data_train = pickle.load(f)
    
with open(dataset_test, 'rb') as f:
    data_test = pickle.load(f)
    
X_train, XT_train = data_train['in'], data_test['in']
y_train, yT_train = data_train['class'], data_test['class']

# train, test, validate
X_train, X_test, \
y_train, y_test = train_test_split(
    X_train, y_train, 
    test_size=0.25, random_state=42069, 
    stratify=y_train)

# train, test, validate
X_train, X_valid, \
y_train, y_valid = train_test_split(
    X_train, y_train, 
    test_size=0.25, random_state=42069, 
    stratify=y_train)


XT_train, XT_test, \
yT_train, yT_test = train_test_split(
    XT_train, yT_train, 
    test_size=0.25, random_state=42069, 
    stratify=yT_train)

# normalize and flatten
# https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
scalerX = StandardScaler()
X_train = scalerX.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scalerX.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
X_valid = scalerX.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
XT_train = scalerX.transform(XT_train.reshape(-1, XT_train.shape[-1])).reshape(XT_train.shape)
XT_test = scalerX.transform(XT_test.reshape(-1, XT_test.shape[-1])).reshape(XT_test.shape)

y_train = utils.to_categorical(y_train, nb_classes)
y_test = utils.to_categorical(y_test, nb_classes)
y_valid = utils.to_categorical(y_valid, nb_classes)
yT_train = utils.to_categorical(yT_train, nb_classes)
yT_test = utils.to_categorical(yT_test, nb_classes)

# make sure s and t have the size
number_of_rows = X_train.shape[0]
random_indices = np.random.choice(number_of_rows, size=(number_of_rows//(batch_size//2))*(batch_size//2), replace=False)
random_indices = np.sort(random_indices)
X_train = X_train[random_indices, :]
y_train = y_train[random_indices]

number_of_rows = XT_train.shape[0]
random_indices = np.random.choice(number_of_rows, size=(number_of_rows//(batch_size//2))*(batch_size//2), replace=False)
random_indices = np.sort(random_indices)
XT_train = XT_train[random_indices, :]
yT_train = yT_train[random_indices]

domain_labels = np.vstack([np.tile([0, 1], [batch_size // 2, 1]),
                           np.tile([1, 0], [batch_size // 2, 1])])

# Created mixed dataset for TSNE visualization
num_test = 500
number_of_rows_s = X_test.shape[0]
number_of_rows_t = XT_test.shape[0]
random_indices_s = np.sort(np.random.choice(number_of_rows_s, size=num_test, replace=False))
random_indices_t = np.sort(np.random.choice(number_of_rows_t, size=num_test, replace=False))
combined_test_data = np.vstack([X_test[random_indices_s], XT_test[random_indices_t]])
combined_test_labels = np.vstack([y_test[random_indices_s], y_test[random_indices_t]])
combined_test_domain = np.vstack([np.tile([1, 0], [num_test, 1]),
                                  np.tile([0, 1], [num_test, 1])])

class DANNBuilder(object):
    def __init__(self):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        # self.opt = SGD(learning_rate=1e-3)
        self.opt_source = SGD(learning_rate=learning_rate_source)
        self.opt_dann = SGD(learning_rate=learning_rate_dann)
        

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Conv1D(filters=32, kernel_size=16, padding='same')(model_input)
        net = LeakyReLU(alpha=alpha_relu)(net)
        
        net = Conv1D(filters=32, kernel_size=16, padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=alpha_relu)(net)
        net = Dropout(dropout_rate)(net)
        
        net = MaxPooling1D(pool_size=4, strides=4) (net)
        
        net = Conv1D(filters=64, kernel_size=16, padding='same')(net)
        net = LeakyReLU(alpha=alpha_relu)(net)
        
        net = Conv1D(filters=64, kernel_size=16, padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=alpha_relu)(net)
        net = Dropout(dropout_rate)(net)
        
        net = MaxPooling1D(pool_size=4, strides=4)(net)
        
        net = Conv1D(filters=128, kernel_size=16, padding='same')(net)
        net = LeakyReLU(alpha=alpha_relu)(net)
        
        net = Conv1D(filters=128, kernel_size=16, padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=alpha_relu)(net)
        net = Dropout(dropout_rate)(net)
        
        net = MaxPooling1D(pool_size=4, strides=4) (net)
        
        net = Flatten()(net)
        
        self.domain_invariant_features = net
        return net

    def _build_classifier(self, model_input):
        net = Dense(256)(model_input)
        net = LeakyReLU(alpha=alpha_relu)(net)
        
        net = Dense(128)(net)
        net = LeakyReLU(alpha=alpha_relu)(net)
        
        net = Dense(nb_classes, activation='softmax',
            name='classifier_output')(net)

        return net

    def build_source_model(self, main_input, plot_model_cond=False):
        net = self._build_feature_extractor(main_input)
        net = self._build_classifier(net)
        model = Model(inputs=main_input, outputs=net)
        if plot_model_cond:
            plot_model(model, to_file='source_model.png', show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy'},
                      optimizer=self.opt_source, metrics=['accuracy'])
        return model

    def build_dann_model(self, main_input, hp_lambda, plot_model_cond=False):
        net = self._build_feature_extractor(main_input)
        self.grl = flipGradientKeras.GradientReversal(hp_lambda)
        branch = self.grl(net)
        branch = Dense(128, activation='relu')(branch)
        branch = Dropout(dropout_rate_dann)(branch)
        branch = Dense(2, activation='softmax', name='domain_output')(branch)
        
        net = Lambda(lambda x: K.switch(tf.convert_to_tensor(K.learning_phase()),
                                        K.concatenate([x[:batch_size // 2], x[:batch_size // 2]], axis=0),
                                        x),
                     output_shape=lambda x: (x[0:]))(net)

        net = self._build_classifier(net)
        model = Model(inputs=main_input, outputs=[branch, net])
        if plot_model_cond:
            plot_model(model, to_file='dann_model.png', show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy',
                      'domain_output': 'categorical_crossentropy'},
                      optimizer=self.opt_dann, metrics=['accuracy'])
        return model

    def build_tsne_model(self, main_input):
        '''Create model to output intermediate layer
        activations to visualize domain invariant features'''
        tsne_model = Model(inputs=main_input,
                           outputs=self.domain_invariant_features)
        return tsne_model


# main_input = Input(shape=(3, img_rows, img_cols), name='main_input')
input_shape = (320, 3)
main_input = Input(shape=input_shape, name='main_input')

builder = DANNBuilder()
src_model = builder.build_source_model(main_input, plot_model_cond=True)
src_vis = builder.build_tsne_model(main_input)

hp_lambda = K.variable(1.0)
dann_model = builder.build_dann_model(main_input, hp_lambda, plot_model_cond=True)
dann_vis = builder.build_tsne_model(main_input)

# earlyStopping = EarlyStopping(monitor='val_loss', patience=300)
# mcp_save = ModelCheckpoint('mdl_wts.hdf5', verbose=0, save_best_only=True, monitor='val_accuracy')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=5e-6)

print('Training source only model')
src_model.fit(X_train, y_train, batch_size=batch_size//2, epochs=nb_epoch_src, verbose=2,
              validation_data=(X_valid, y_valid),
              shuffle=True)
print('Evaluating source samples on source-only model')
print('Accuracy:', src_model.evaluate(X_test, y_test, verbose=0)[1])
print('Evaluating target samples on source-only model')
print('Accuracy:', src_model.evaluate(XT_test, yT_test, verbose=0)[1])

# Broken out training loop for a DANN model.
src_index_arr = np.arange(X_train.shape[0])
target_index_arr = np.arange(XT_train.shape[0])

domain_train = np.vstack([np.tile([0, 1], [X_train.shape[0], 1])])
domain_test = np.vstack([np.tile([0, 1], [X_test.shape[0], 1])])
domainT_train = np.vstack([np.tile([0, 1], [XT_train.shape[0], 1])])
domainT_test = np.vstack([np.tile([0, 1], [XT_test.shape[0], 1])])

batches_per_epoch = len(X_train) / batch_size
num_steps = nb_epoch_trg * batches_per_epoch

print('Training DANN model')

for i in range(nb_epoch_trg):

    batches = make_batches(X_train.shape[0], batch_size // 2)
    target_batches = make_batches(XT_train.shape[0], batch_size // 2)

    src_gen = batch_gen(batches, src_index_arr, X_train, y_train)
    target_gen = batch_gen(target_batches, target_index_arr, XT_train, None)
    
    nb_batches = len(batches)-1
    
    j = 0
    
    for (xb, yb) in src_gen:
        # Update learning rate and gradient multiplier 
        # as described in the paper.
        p = float(j) / nb_batches
        l = 2. / (1. + np.exp(-gamma * p)) - 1
        lr = mu_0 / (1. + alpha * p)**beta
        K.set_value(builder.grl.hp_lambda, l)
        K.set_value(builder.opt_dann.lr, lr)

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

    acc_train_src = dann_model.evaluate(X_train, y={'classifier_output': y_train, 'domain_output': domain_train}, batch_size=batch_size, verbose=0)
    acc_test_src = dann_model.evaluate(X_test, y={'classifier_output': y_test, 'domain_output': domain_test}, batch_size=batch_size, verbose=0)
    acc_train_trg = dann_model.evaluate(XT_train, y={'classifier_output': yT_train, 'domain_output': domainT_train}, batch_size=batch_size, verbose=0)
    acc_test_trg = dann_model.evaluate(XT_test, y={'classifier_output': yT_test, 'domain_output': domainT_test}, batch_size=batch_size, verbose=0)
    # print('Epoch ', i, 'Train Accuracy:', acc_train_trg, 'Test Accuracy:', acc_test_trg)
    print('Epoch ', i, 'Source Train Accuracy:', acc_train_src[4], 'Target Train Accuracy:', acc_train_trg[4])
    print('Epoch ', i, 'Source Test Accuracy:', acc_test_src[4], 'Target Test Accuracy:', acc_test_trg[4])
    #if acc_test > 0.70:
    #    break
    
    np.random.shuffle(src_index_arr)
    np.random.shuffle(target_index_arr)

    X_train = X_train[src_index_arr]
    y_train = y_train[src_index_arr]
    XT_train = XT_train[target_index_arr]
    yT_train = yT_train[target_index_arr]


print('Evaluating source samples on source-only model')
print('Accuracy:', src_model.evaluate(X_test, y_test, verbose=0)[1])
print('Evaluating target samples on source-only model')
print('Accuracy:', src_model.evaluate(XT_test, yT_test, verbose=0)[1])

domain_test = np.vstack([np.tile([0, 1], [X_test.shape[0], 1])])

print('Evaluating source samples on DANN model')
acc = dann_model.evaluate(X_test, y={'classifier_output': y_test, 'domain_output': domain_test}, batch_size=batch_size, verbose=0)
print('Accuracy:', acc[4])
print('Evaluating target samples on DANN model')
acc = dann_model.evaluate(XT_test, y={'classifier_output': yT_test, 'domain_output': domain_test[:2188]}, batch_size=batch_size, verbose=0)
print('Accuracy:', acc[4])
print('Visualizing output of domain invariant features')

# src_embedding = src_vis.predict([combined_test_data])
# src_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
# tsne = src_tsne.fit_transform(src_embedding)

# plot_embedding(tsne, combined_test_labels.argmax(1),
#                combined_test_domain.argmax(1), 'Source only')

# dann_embedding = dann_vis.predict([combined_test_data])
# dann_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
# tsne = dann_tsne.fit_transform(dann_embedding)

# plot_embedding(tsne, combined_test_labels.argmax(1),
#                combined_test_domain.argmax(1), 'DANN')

# plt.show()
