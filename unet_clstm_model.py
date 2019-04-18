from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from IPython.display import Image

class Unet_clstm_model(object):
    """
    resnet reference from: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    """
    def __init__(self,
                 input_shape
                 ):
        self.input_shape = input_shape

    def resnet_layer(self,
                     inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """
        2D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)
        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_block(self,
                     inputs,
                     num_filters):
        x = self.resnet_layer(inputs=inputs,
                              num_filters=num_filters,
                              activation=None,
                              kernel_size=1,
                              batch_normalization=False,
                              conv_first=True)
        y = self.resnet_layer(inputs=x,
                              num_filters=int(num_filters/4),
                              activation='relu',
                              kernel_size=1,
                              batch_normalization=False,
                              conv_first=True)
        y = self.resnet_layer(inputs=y,
                              num_filters=int(num_filters/4),
                              activation='relu',
                              kernel_size=3,
                              batch_normalization=True,
                              conv_first=False)
        y = self.resnet_layer(inputs=y,
                              num_filters=num_filters,
                              activation=None,
                              kernel_size=1,
                              batch_normalization=False,
                              conv_first=False)
        x = add([x, y])
        x = Activation('relu')(x)
        return x

    def res_unet_model(self):
        inputs = Input((self.input_shape[1], self.input_shape[2], self.input_shape[3]))
        conv1 = self.resnet_block(inputs=inputs, num_filters=64)
        dilation1 = Conv2D(64, 3, strides=1, dilation_rate=2, padding='same')(conv1)
        strides1 = Conv2D(64, 1, strides=2, dilation_rate=1, padding='same')(dilation1)

        conv2 = self.resnet_block(inputs=strides1, num_filters=128)
        dilation2 = Conv2D(128, 3, strides=1, dilation_rate=2, padding='same')(conv2)
        strides2 = Conv2D(128, 1, strides=2, dilation_rate=1, padding='same')(dilation2)

        conv3 = self.resnet_block(inputs=strides2, num_filters=256)
        dilation3 = Conv2D(256, 3, strides=1, dilation_rate=2, padding='same')(conv3)
        strides3 = Conv2D(256, 1, strides=2, dilation_rate=1, padding='same')(dilation3)

        conv4 = self.resnet_block(inputs=strides3, num_filters=512)
        drop4 = Dropout(0.5)(conv4)
        dilation4 = Conv2D(512, 3, strides=1, dilation_rate=2, padding='same')(conv4)
        strides4 = Conv2D(512, 1, strides=2, dilation_rate=1, padding='same')(dilation4)

        conv5 = self.resnet_block(inputs=strides4, num_filters=1024)
        drop5 = Dropout(0.5)(conv5)

        up6 = self.resnet_block(inputs=UpSampling2D(size=(2, 2))(drop5), num_filters=512)
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = self.resnet_block(inputs=merge6, num_filters=512)

        up7 = self.resnet_block(inputs=UpSampling2D(size=(2, 2))(conv6), num_filters=256)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = self.resnet_block(inputs=merge7, num_filters=256)

        up8 = self.resnet_block(inputs=UpSampling2D(size=(2, 2))(conv7), num_filters=128)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = self.resnet_block(inputs=merge8, num_filters=128)

        up9 = self.resnet_block(inputs=UpSampling2D(size=(2, 2))(conv8), num_filters=64)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = self.resnet_block(inputs=merge9, num_filters=64)


        model = Model(inputs=inputs, outputs=conv9)
        # plot_model(model, to_file="res_unet.png", show_shapes=True)
        # Image('./model_res_unet.png')
        # model.summary()
        return model

    def res_unet_clstm(self):
        input_sequences = Input(shape=self.input_shape, name='main_input')
        processed_sequences = TimeDistributed(self.res_unet_model())(input_sequences)
        bi_input_sequences = Input(shape=self.input_shape, name='aux_input')
        bi_processed_sequences = TimeDistributed(self.res_unet_model())(bi_input_sequences)

        conv_lstm_1 = ConvLSTM2D(32, 3, padding='same', return_sequences=True)(processed_sequences)
        conv_lstm_1 = BatchNormalization()(conv_lstm_1)
        conv_lstm_2 = ConvLSTM2D(32, 3, padding='same', return_sequences=True)(bi_processed_sequences)
        conv_lstm_2 = BatchNormalization()(conv_lstm_2)

        concatenate_1 = concatenate([conv_lstm_1, conv_lstm_2])

        conv_3d = Conv3D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate_1)
        activation = Activation('softmax')(conv_3d)

        model = Model(inputs=[input_sequences, bi_input_sequences], outputs=activation)
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        # plot_model(model, to_file="model_unet_bi_clstm.png", show_shapes=True)
        # Image('./model_unet_bi_clstm.png')
        model.summary()
        return model


if __name__ == '__main__':

    unet_clstm = Unet_clstm_model(input_shape=(21, 128, 128, 1))
    model = unet_clstm.res_unet_model()
