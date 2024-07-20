import os
import json
from pathlib import Path
import numpy as np
from keras import utils, Model, optimizers
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from model_architecture import ModelArchitecture

class TrainModel(Model, ModelArchitecture):
    def __init__(self):
        super(TrainModel, self).__init__()
        super(ModelArchitecture, self).__init__()

        self.image_folder_path = 'images/'
        self.get_model_details()

    def get_model_details(self):
        model_details = {
            'batch-size': 100,
            'class-list': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
            'epoch': 60,
            'l2-strength': 0.001,
            'learning-rate': 0.001,
            'model-architecture': 'type1',
            'output-size': 7,
            'picture-size': 96,
            'random-seed': 6345,
            'version': '0.1.1'
        }

        version_number = input('Enter the version number [0.1.1]: ')
        base_dir = Path(__file__).resolve().parent.parent
        model_dir = Path(base_dir, 'models', f'model-{version_number}')

        if not Path.exists(model_dir):
            os.mkdir(model_dir)
            model_details['version'] = version_number

            with open(f'{model_dir}/model-details.json', 'w') as json_file:
                json_file.write(json.dumps(model_details, indent=4))
        else:
            print(f'Model version {version_number} already exists.')
            replace_model = input('Do you want to re-train the existing model? [y/n]: ')

            if replace_model.lower() not in ['y', 'yes']:
                return
            
            with open(f'{model_dir}/model-details.json', 'r') as json_file:
                model_details = json.load(json_file)

        self.version = model_details['version']
        self.model_folder_path = Path('models', f'model-{version_number}')
        self.emotions_list = model_details['class-list']
        self.epoch = model_details['epoch']
        self.batch_size = model_details['batch-size']
        self.picture_size = model_details['picture-size']
        self.output_size = len(self.emotions_list)
        self.l2_strength = model_details['l2-strength']

        self.train_model(model_details['random-seed'])
    
    def get_train_test(self):
        ''' Gets the training and testing dataset from the images which are
        augmented before it is trained and tested on the model.
        '''
        train_set = ImageDataGenerator(
            rescale=1./255,
            rotation_range = 10,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        ).flow_from_directory(
            Path(self.image_folder_path, 'train'),
            batch_size=self.batch_size,
            classes=self.emotions_list,
            target_size=(self.picture_size, self.picture_size),
            color_mode='grayscale',
            shuffle=True
        )

        validation_set = ImageDataGenerator(
            rescale=1./255
        ).flow_from_directory(
            Path(self.image_folder_path, 'validation'),
            batch_size=self.batch_size,
            classes=self.emotions_list,
            target_size=(self.picture_size, self.picture_size),
            color_mode='grayscale',
            shuffle=True
        )

        return train_set, validation_set
    
    @staticmethod
    def get_categorical_entity_count(dataset):
        values = {}

        for index in range(len(dataset)):
            for j in np.argmax(dataset[index][1], axis=1):
                if j not in values.keys():
                    values[j] = 0
                values[j] += 1
        return values
    
    def plot_graph(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Test')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Test')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        fig.tight_layout()
        plt.savefig(f'models/model-{self.version}/accuracy_loss_plot.png', dpi=300)
        plt.show()

    def train_model(self, seed):
        utils.set_random_seed(seed)

        train_set, validation_set = self.get_train_test()
        model = self.get_modal()
        model.summary()

        checkpoint = ModelCheckpoint(
            filepath=f'models/model-{self.version}/model.keras',
            save_best_only=True,
            verbose=1,
            mode='max',
            monitor='accuracy'
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1,
            min_delta=0.0001
        )
        
        model.compile(
            optimizer=optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        history = model.fit(
            train_set,
            epochs=self.epoch,
            batch_size=self.batch_size,
            validation_data=validation_set,
            callbacks=[checkpoint, reduce_lr]
        )

        model.save(f'models/model-{self.version}/model.h5')
        self.plot_graph(history)

