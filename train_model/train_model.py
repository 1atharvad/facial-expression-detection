import os
from keras import models, layers, regularizers, utils
from keras.preprocessing.image import ImageDataGenerator

class TrainModel:
    def __init__(self):
        self.epoch = 10
        self.batch_size = 128
        self.picture_size = 48
        self.l2_strength = 0.0001
        self.folder_path = 'images/'
        self.model_json_file = 'model.json'
        self.model_keras = 'model_weights.keras'
        self.train_model()

    def get_modal(self):
        """ Creates a sequential model with multiple layers such as Conv2D,
        MaxPooling2D, Flatten, Dense and Activation.
        """
        model = models.Sequential()
        model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength), input_shape=(self.picture_size, self.picture_size, 1)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.MaxPooling2D((2, 2), (2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2=self.l2_strength)))
        model.add(layers.MaxPooling2D((2, 2), (2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(l2=self.l2_strength), activation='relu'))
        model.add(layers.Dense(7, kernel_regularizer=regularizers.l2(l2=self.l2_strength), activation='linear'))
        model.add(layers.Activation('softmax'))

        return model
    
    def get_train_test(self):
        train_set = ImageDataGenerator().flow_from_directory(
            os.path.join(self.folder_path, 'train'),
            target_size=(self.picture_size, self.picture_size),
            color_mode='grayscale',
            shuffle=True
        )

        validation_set = ImageDataGenerator().flow_from_directory(
            os.path.join(self.folder_path, 'validation'),
            target_size=(self.picture_size, self.picture_size),
            color_mode='grayscale',
            shuffle=True
        )

        return train_set, validation_set

    def train_model(self):
        utils.set_random_seed(6345)

        train_set, validation_set = self.get_train_test()
        model = self.get_modal()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=train_set, epochs=self.epoch, batch_size=self.batch_size, validation_data=validation_set)

        model_json = model.to_json()
        model.save_weights(self.model_keras)
        with open(self.model_json_file, 'w') as json_file:
            json_file.write(model_json)
