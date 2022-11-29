'''
File created by Reza Kalantar - 29/11/2022
'''

import utils
import config
from losses import *
import tensorflow as tf
from models import modelGenerator, modelDiscriminator

class CycleGAN():
    def __init__(self, args):

        # Build and compile the discriminators
        self.D_A = modelDiscriminator(tuple(config.params['roi_size']+[1])) # add channel dim
        self.D_B = modelDiscriminator(tuple(config.params['roi_size']+[1]))

        # Build the generators
        self.G_A2B = modelGenerator(tuple(config.params['roi_size']+[1]), args.g_residual_blocks)
        self.G_B2A = modelGenerator(tuple(config.params['roi_size']+[1]), args.g_residual_blocks)

        # Set optimizers
        self.G_A2B_optimizer = tf.keras.optimizers.Adam(args.lr_G, beta_1=0.5)
        self.G_B2A_optimizer = tf.keras.optimizers.Adam(args.lr_G, beta_1=0.5)

        self.discriminator_A_optimizer = tf.keras.optimizers.Adam(args.lr_D, beta_1=0.5)
        self.discriminator_B_optimizer = tf.keras.optimizers.Adam(args.lr_D, beta_1=0.5)

    @tf.function
    def train_step(self, real_A, real_B):
        # persistent is set to True because the tape is used more than once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            fake_B = self.G_A2B(real_A)
            cycle_A = self.G_B2A(fake_B)

            fake_A = self.G_B2A(real_B)
            cycle_B = self.G_A2B(fake_A)

            # same_A and same_B are used for identity loss.
            same_A = self.G_B2A(real_A)
            same_B = self.G_A2B(real_B)

            disc_real_A = self.D_A(real_A)
            disc_real_B = self.D_B(real_B)

            disc_fake_A = self.D_A(fake_A)
            disc_fake_B = self.D_B(fake_B)

            # calculate losses
            G_A2B_loss = generator_loss(disc_fake_B)
            G_B2A_loss = generator_loss(disc_fake_A)

            cycle_A_loss = cycle_loss(real_A, cycle_A)
            cycle_B_loss = cycle_loss(real_B, cycle_B)

            total_cycle_loss = cycle_A_loss + cycle_B_loss

            # Total generator loss = adversarial loss + cycle loss
            total_G_A2B_loss = G_A2B_loss + total_cycle_loss + identity_loss(real_B, same_B)
            total_G_B2A_loss = G_B2A_loss + total_cycle_loss + identity_loss(real_A, same_A)

            disc_A_loss = discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = discriminator_loss(disc_real_B, disc_fake_B)

        # Calculate the gradients for generator and discriminator
        G_A2B_gradients = tape.gradient(total_G_A2B_loss,
                                          self.G_A2B.trainable_variables)
        G_B2A_gradients = tape.gradient(total_G_B2A_loss,
                                          self.G_B2A.trainable_variables)

        discriminator_A_gradients = tape.gradient(disc_A_loss,
                                                   self.D_A.trainable_variables)
        discriminator_B_gradients = tape.gradient(disc_B_loss,
                                                   self.D_B.trainable_variables)

        # Apply the gradients to the optimizer
        self.G_A2B_optimizer.apply_gradients(zip(G_A2B_gradients,
                                              self.G_A2B.trainable_variables))

        self.G_B2A_optimizer.apply_gradients(zip(G_B2A_gradients,
                                              self.G_B2A.trainable_variables))

        self.discriminator_A_optimizer.apply_gradients(zip(discriminator_A_gradients,
                                                       self.D_A.trainable_variables))

        self.discriminator_B_optimizer.apply_gradients(zip(discriminator_B_gradients,
                                                       self.D_B.trainable_variables))

        return fake_A, fake_B, cycle_A, cycle_B, G_A2B_loss, G_B2A_loss, cycle_A_loss, cycle_B_loss, disc_A_loss, disc_B_loss
