"""
This file contains the implementatino of the DCGAN architecture and 
training loop for the celeba datset
"""

import numpy as np
import tensorflow as tf
from .utils import make_grid

import os 
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, LeakyReLU

import time 

def build_generator(latent_dim=100):
    """
    Returns a generator as specified in the dcgan paper
    Latent dim is the dimension of the input to the generator
    """
    # The weight initialization and the slope are chosen to accord with the
    # Parameters in the paper. I only change padding when it seems neccesary to
    # to mantain adequate dimensons. 
    weight_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    slope = 0.3
    
    inputs = keras.Input(shape=(1,1,100))
    # First convolutional layer
    x = Conv2DTranspose(
        1024, 
        kernel_size=(4,4), 
        strides=1, 
        kernel_initializer=weight_initializer,
        padding='valid',
        use_bias=False
    )(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=slope)(x)
    
    # Second convolutional layer
    x = Conv2DTranspose(
        kernel_initializer=weight_initializer,
        filters = 512,
        kernel_size = 4,
        strides = (2,2),
        padding = 'same',
        use_bias = False
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=slope)(x)
    
    # Third convolutional layer
    x = Conv2DTranspose(
        kernel_initializer=weight_initializer,
        filters = 256,
        kernel_size = 5,
        strides = (2,2),
        use_bias=False,
        padding = 'same',
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=slope)(x)

    # Fourth convolutional layer
    x = Conv2DTranspose(
        kernel_initializer=weight_initializer,
        filters = 128,
        kernel_size = (5,5),
        strides = (2,2),
        use_bias=False,
        padding = 'same',
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=slope)(x)

    # Fifth convolutional layer
    x = Conv2DTranspose(
        kernel_initializer=weight_initializer,
        filters = 3,
        kernel_size = (5,5),
        use_bias=False,
        strides = (2,2),
        padding = 'same',
        activation='tanh'
    )(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model


def build_discriminator():
    """
    Returns a discriminator built in almost the exact opposite way 
    of the generator in the paper. This is not very different from a normal 
    generator.
    """

    #Slope and weight initializer are chosen to match parmeters in the paper
    weight_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    slope = 0.2
    inputs = keras.Input(shape=(64,64,3))
    x = preprocessing.Rescaling(scale=1./127.5, offset=-1.)(inputs)

    # First conv layer
    x = Conv2D(
        64,
        4,
        2,
        padding='same',
        use_bias=False,
        kernel_initializer=weight_initializer
    )(x)
    x = LeakyReLU(alpha=slope)(x)

    # Second conv layer
    x = Conv2D(
        128,
        4,
        2,
        padding='same',
        use_bias=False,
        kernel_initializer=weight_initializer
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=slope)(x)
    
    # Third conv layer
    x = Conv2D(
        256,
        4,
        2,
        padding='same',
        use_bias=False,
        kernel_initializer=weight_initializer
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=slope)(x)

    # Fourth conv layer
    x = Conv2D(
        512,
        4,
        2,
        padding='same',
        use_bias=False,
        kernel_initializer=weight_initializer
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=slope)(x)

    # Predictions. Note that we use logits so thhere is no activation at the end. 
    x = layers.Flatten()(x)
    x = layers.Dense(1,kernel_initializer=weight_initializer)(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
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
    dis_optim = tf.keras.optimizers.Adam(lr = lr, beta_1 = beta_1)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint =  tf.train.Checkpoint(
        generator=generator, 
        discriminator=discriminator
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
        for image_batch in dataset:
            batch_step +=1 
            latent_vars = tf.random.normal([batch_size, 1, 1, lat_dim])
            
            # We maximize the log(D(x)) + log(1 - D(G(X))
            # Here we take the gradients in two different steps. 
            # this is exactly what is done in the original gan paper.  Moreover we compute the loss using 
            # Cross entropy for its simplicity. We see that there is no label smoothing here. 
            with  tf.GradientTape() as disc_tape :
                real_logits = discriminator(image_batch, training=True)
                fake_images = generator(latent_vars, training=True)
                fake_images = fake_images * 127.5 + 127.5
                fake_logits = discriminator(fake_images, training=True)
                
                log_DX_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
                log_DXG_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
                
                D_loss = log_DX_loss + log_DXG_loss
                grad_D = disc_tape.gradient(D_loss, discriminator.trainable_variables)
                dis_optim.apply_gradients(zip(grad_D, discriminator.trainable_variables))

            with tf.GradientTape() as gen_tape:
                fake_images = generator(latent_vars, training=True)
                # This additional multiplication is done to obtain a real image. 
                fake_images = fake_images * 127.5 + 127.5
                fake_logits = discriminator(fake_images, training=True)
                G_loss = cross_entropy(tf.ones_like(fake_logits), fake_logits)

                grad_G = gen_tape.gradient(G_loss, generator.trainable_variables)
                gen_optim.apply_gradients(zip(grad_G, generator.trainable_variables))

            G_loss_history.append(G_loss.numpy())
            D_loss_history.append(D_loss.numpy())
            
            # We save the images just to make sure that there is progress. 
            # Given that the loss is not like a normal loss this is one of the more effective ways to monitor training. 
            if batch_step % print_freq == 0: 
                print('[{}] - disc loss {:.4f} - gen_loss{:.4f} - {}/{}'.format(epoch ,D_loss, G_loss, batch_step, len(dataset)))
            if batch_step % save_img_freq == 0:
                fake_images = generator(prog_images, training=False) * 127.5 + 127.5
                save_location = os.path.join(progression_images_dir, 'progression_image_{}_{}'.format(epoch, batch_step))
                make_grid(fake_images, len(fake_images), save_location)                                

        #Epoch checkpoints and saving of progress images
        checkpoint.save(file_prefix=checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        
    return (G_loss_history, D_loss_history)
        
          
          

