# Do simple classifier of Fashion-MNIST with RELU and SELU models -- just to see how it goes

import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)

# Toggle: If activator is "selu"" we use selu with recommended init. Otherwise go with relu
activator = "RELU".lower()
initializer = 'lecun_normal' if activator.lower() == 'selu' else 'glorot_normal'

# VERY simple model ....
layer_sizes = [256, 256, 128, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28), name="input_flatten"))
for idx, size in enumerate(layer_sizes):
    model.add(keras.layers.Dense(size,
                                 activation=activator.lower(),
                                 name=f"dense_{idx}",
                                 kernel_initializer=initializer,
                                 kernel_regularizer=keras.regularizers.l1()))

model.add(keras.layers.Dense(10, name='classification_top'))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
no_epochs = 1001

# Set up logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './logs/' + activator + '/' + current_time + '/train'
test_log_dir = './logs/' + activator + '/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# Set up learning
# Loss
def loss(_model, _x, _y, _training):
    return loss_object(y_true=_y, y_pred=model(_x, training=_training))


# Gradient
def grad(_model, _inputs, _targets):
    with tf.GradientTape() as tape:
        _loss = loss(_model, _inputs, _targets, _training=True)

    return _loss, tape.gradient(_loss, model.trainable_variables)


# Do it...
train_losses = []
train_acc = []
test_losses = []
test_acc = []
for epoch in range(no_epochs):
    loss_value, grads = grad(model, train_images, train_labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Store values to lists and show on screen and in Tensorboard
    # Training results
    train_losses.append(loss_value.numpy())
    train_acc.append(np.average(tf.argmax(model(train_images, training=False), axis=1).numpy() == train_labels))
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_losses[-1], step=epoch)
        tf.summary.scalar('accuracy', train_acc[-1], step=epoch)

    # Testing results
    test_losses.append(loss(_model=model, _x=test_images, _y=test_labels, _training=False))
    test_acc.append(np.average(tf.argmax(model(test_images, training=False), axis=1).numpy() == test_labels))
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_losses[-1], step=epoch)
        tf.summary.scalar('accuracy', test_acc[-1], step=epoch)

    print(f"Epoch {epoch:4d}: Training loss/acc = {train_losses[-1]:6.3f}/{train_acc[-1]:5.3f} --- " +
          f"Test loss/acc = {test_losses[-1]:6.3f}/{test_acc[-1]:5.3f}")

    with train_summary_writer.as_default():
        for g, v in zip(grads, model.trainable_variables):
            # Dump gradients as histograms
            tf.summary.histogram(f"{v.name}/grad/histogram", g, step=epoch)
            # Dump sparsity as scalars
            tf.summary.scalar(f"{v.name}/grad/sparsity", tf.nn.zero_fraction(g), step=epoch)
            # Dump mean of abs as scalar
            tf.summary.scalar(f"{v.name}/grad/mean-abs", tf.reduce_mean(tf.abs(g)), step=epoch)
            # Dump standard deviation as scalar
            tf.summary.scalar(f"{v.name}/grad/std", tf.math.reduce_std(tf.abs(g)), step=epoch)

            # Dump weights and biases of each layer
            tf.summary.histogram(f"{v.name}/values/histogram", v.numpy(), step=epoch)
            # Sparsity of weights
            tf.summary.scalar(f"{v.name}/values/sparsity", tf.nn.zero_fraction(v.numpy()), step=epoch)
            # Dump mean of abs as scalar
            tf.summary.scalar(f"{v.name}/values/mean-abs", tf.reduce_mean(tf.abs(v.numpy())), step=epoch)
            # Dump standard deviation as scalar
            tf.summary.scalar(f"{v.name}/values/dev", tf.math.reduce_std(v.numpy()), step=epoch)

        # Histograms of outputs from each layer
        # all layer outputs, except first (flatten) and last (classification)
        output_locations = []
        layer_names = []
        for layer in model.layers[1:-1]:
            output_locations.append(layer.output)
            layer_names.append(layer.name)
        # evaluation function
        layer_outputs = keras.backend.function(model.input, output_locations)([train_images, 1.])

        for name, value in zip(layer_names, layer_outputs):
            # All values as histogram
            tf.summary.histogram(f"{name}/output/histogram", value, step=epoch)
            # Dump sparsity as scalars
            tf.summary.scalar(f"{name}/output/sparsity", tf.nn.zero_fraction(value), step=epoch)
            # Dump mean of abs as scalar
            tf.summary.scalar(f"{name}/output/mean", tf.reduce_mean(value), step=epoch)
            # Dump standard deviation as scalar
            tf.summary.scalar(f"{name}/output/dev", tf.math.reduce_std(value), step=epoch)

        train_summary_writer.flush()
