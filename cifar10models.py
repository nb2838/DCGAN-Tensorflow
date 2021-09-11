import numpy as np
import tensorflow as tf
from .utils import make_grid
import pickle

import os 
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, LeakyReLU, ReLU

import time

def build_generator(Latent_dim=100):
    """
    Returns a generator as specified in the dcgan paper
    Latent dim is the dimension of the input to the generator
    """
    # The weight initialization and the slope are chosen to accord with the
    # Parameters in the paper. I only change padding when it seems neccesary to
    # mantain adequate dimensions. 
    weight_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    slope = 0.2
    
    inputs = keras.Input(shape=(1,1,100))
    # First convolutional layer
    fconv = Conv2DTranspose(
        512, 
        kernel_size=(4,4), 
        strides=1, 
        kernel_initializer=weight_initializer,
        padding='valid',
        use_bias=False
    )(inputs)
    x = BatchNormalization()(fconv)
    x = ReLU()(x)
    
    # Second convolutional layer
    sconv = Conv2DTranspose(
        kernel_initializer=weight_initializer,
        filters = 256,
        kernel_size = 4,
        strides = (2,2),
        padding = 'same',
        use_bias = False
    )(x)
    x = BatchNormalization()(sconv)
    x = ReLU()(x)
    
    # Third convolutional layer
    tconv = Conv2DTranspose(
        kernel_initializer=weight_initializer,
        filters = 128,
        kernel_size = 5,
        strides = (2,2),
        use_bias=False,
        padding = 'same',
    )(x)
    x = BatchNormalization()(tconv)
    x = ReLU()(x)

    # Fourth convolutional layer
    foconv = Conv2DTranspose(
        kernel_initializer=weight_initializer,
        filters = 3,
        kernel_size = (5,5),
        strides = (2,2),
        use_bias=False,
        padding = 'same',
        activation='tanh'
    )(x)
    # We don't include more layers to images of shape (32 X 32 X 3) shape
    model = keras.Model(inputs=inputs, outputs=foconv)
    return model


def build_discriminator():
    """
    Returns a discriminator built in almost the exact opposite way 
    of the generator in the paper. This is not very different from a normal 
    discriminator.
    """

    #Slope and weight initializer are chosen to match parmeters in the paper
    weight_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    slope = 0.2
    inputs = keras.Input(shape=(32,32,3))
    # For simplicity we add an additional layer that allows us to work with regular images without previous 
    #preprocessing 
    x = preprocessing.Rescaling(scale=1./127.5, offset=-1.)(inputs)
    # First conv layer
    fconv = Conv2D(
        64,
        4,
        2,
        padding='same',
        use_bias=False,
        kernel_initializer=weight_initializer
    )(x)
    x = LeakyReLU(alpha=slope)(fconv)

    # Second conv layer
    sconv = Conv2D(
        128,
        4,
        2,
        padding='same',
        use_bias=False,
        kernel_initializer=weight_initializer
    )(x)
    x = BatchNormalization()(sconv)
    x = LeakyReLU(alpha=slope)(x)
    x = layers.Dropout(0.5)(x)
    
    # Third conv layer
    tconv = Conv2D(
        256,
        4,
        2,
        padding='same',
        use_bias=False,
        kernel_initializer=weight_initializer
    )(x)
    x = BatchNormalization()(tconv)
    x = LeakyReLU(alpha=slope)(x)

    # Predictions. Note that we use logits so thhere is no activation at the end. 
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1,kernel_initializer=weight_initializer)(x)

    fpool = tf.keras.layers.GlobalMaxPooling2D()(fconv)
    spool = tf.keras.layers.GlobalMaxPooling2D()(sconv)
    tpool = tf.keras.layers.GlobalMaxPooling2D()(tconv)

    pool = tf.keras.layers.Concatenate()([fpool,spool,tpool])
    
    model = keras.Model(inputs=inputs, outputs=[x,pool])
    return model




def train(generator,
          discriminator,
          dataset,
          batch_size=128,
          epochs = 10,
          print_freq = 50,
          start_epoch = 0,
          checkpoint_dir = '',
          checkpoint = None,
          progression_images_dir = '',
          save_img_freq=400,
          data_len = 60000,
          lat_dim = 100
          
):
    """
    The loop below is loosely adapted from the GAN tutorial in the tensorflow documentation. 
    I initially tried using a loop more like the one described in the paper but I kept experiencing collapse. 
    With this loop I got slightly better results.
    The hyperparameters of ADAM are chosen to match those in the paper.
    In addition we use cross entropy cause this can be used to obtain the min max loss.

    Note: 
    1.  In line with other literature and as opposed to the the paper I will also maximize
    ln( D(G(x)) instead of the proper ln(1 - D(G(x)). This helps with gradients. 
    """

    # HYPERPARAMETERS
    lr = 0.0002
    beta_1 = 0.5
    gen_optim = tf.keras.optimizers.Adam(lr = lr,beta_1 = beta_1)
    dis_optim = tf.keras.optimizers.Adam(lr = lr,beta_1=beta_1)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint =  tf.train.Checkpoint(
        cifgenerator=generator, 
        cifdiscriminator=discriminator
    )
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # TRAINING IMAGES USED TO KEEP TRACK OF PROGRESSION THROUGHOUT THE EPOCHS
    tf.random.set_seed(123)
    prog_images = tf.random.normal([16, 1, 1, lat_dim])

    G_loss_history = []
    D_loss_history = []

    for epoch in range(start_epoch, epochs):
        # This will be later used to print things to the terminal 
        start = time.time()
        batch_step = 0
        for batch_step in range(int(data_len/batch_size)):
            image_batch = next(dataset)
            batch_step +=1 
            latent_vars = tf.random.normal([batch_size, 1, 1, lat_dim])


            # I found the strategy to accumulate gradients in
            # https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow
            # Which I will use as I found it being referenced as a possible help in obraining stability. 
            # This computation of the gradient in two steps was required in the CIFAR 10 dataset as opposed to the celeba one
            # We maximize the log(D(x)) + log(1 - D(G(X))
            accumulator = [tf.Variable(tf.zeros_like(v), trainable=False) for v in discriminator.trainable_variables]
            with  tf.GradientTape() as disc_tape :
                # we discard the second input so that it is not used for the computation of the gradient. 
                # we will use it later when doing the svm computation 
                real_logits, _ = discriminator(image_batch, training=True)
                # We use soft labels as a regularization strategy
                labels = tf.random.uniform(shape=real_logits.shape,minval=0.7,maxval=0.1)
                log_DX_loss = cross_entropy(labels, real_logits)
                grad_D1 = disc_tape.gradient(log_DX_loss, discriminator.trainable_variables)

            grad_D = [accumulator[i].assign_add(gradient) for i, gradient in enumerate(grad_D1)]
            
            # this is still part of the loss of D but with the trick of gradient accumulation it is 
            # calculated in two steps. We need to tapes for each of the acumulations. 
            with tf.GradientTape() as disc_tape:                
                fake_images = generator(latent_vars, training=True)
                fake_images = fake_images * 127.5 + 127.5
                fake_logits, _ = discriminator(fake_images, training=True)
                labels = tf.random.uniform(shape=fake_logits.shape,minval=0.0,maxval=0.3)
                log_DXG_loss = cross_entropy(labels, fake_logits)
                grad_D2 = disc_tape.gradient(log_DXG_loss, discriminator.trainable_variables)
                D_loss = log_DX_loss + log_DXG_loss
                
            # We accumulate the gradients before applying them
            grad_D = [accumulator[i].assign_add(gradient) for i, gradient in enumerate(grad_D2)]
            dis_optim.apply_gradients(zip(accumulator, discriminator.trainable_variables))   

            # This is the loss of G. We don't use soft labels.  as above. 
            # Notice that the latent_variables are shared throughout the process.    
            with tf.GradientTape() as gen_tape:
                fake_images = generator(latent_vars, training=True)
                fake_images = fake_images * 127.5 + 127.5
                fake_logits, _ = discriminator(fake_images, training=True)
                G_loss = cross_entropy(tf.random.uniform(
                    shape=fake_logits.shape,minval=1,maxval=1), fake_logits)

                grad_G = gen_tape.gradient(G_loss, generator.trainable_variables)
                gen_optim.apply_gradients(zip(grad_G, generator.trainable_variables))


            G_loss_history.append(G_loss.numpy())
            D_loss_history.append(D_loss.numpy())
            
            if batch_step % print_freq == 0: 
                print('[{}] - disc loss {:.4f} - gen_loss{:.4f} - {}/{}'.format(epoch ,D_loss, G_loss, batch_step, data_len/batch_size))
            if batch_step % save_img_freq == 0:
                fake_images = generator(prog_images, training=False) * 127.5 + 127.5
                save_location = os.path.join(progression_images_dir, 'progression_image_{}_{}'.format(epoch, batch_step))
                make_grid(fake_images, len(fake_images), save_location)                                

        #Epoch checkpoints and saving of progress images
        checkpoint.save(file_prefix=checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
        with open('history', 'wb') as handle:
            pickle.dump({'Gloss': G_loss_history,'Dloss': D_loss_history}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
    return (G_loss_history, D_loss_history)
        


