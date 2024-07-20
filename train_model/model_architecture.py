from keras import models, layers, regularizers

class ModelArchitecture:
    def __init__(self, picture_size, output_size, l2_strength):
        self.picture_size = picture_size
        self.output_size = output_size
        self.l2_strength = l2_strength

    def get_modal(self):
        ''' Creates a sequential model with multiple layers such as Conv2D,
        MaxPooling2D, Flatten, Dense and Activation.
        '''
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength), input_shape=(self.picture_size, self.picture_size, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(self.output_size, activation='softmax', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))

        return model