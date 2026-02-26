import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import os     
import pathlib
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0, InceptionV3, Xception
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')

class RiceDiseasePredictor:
    def __init__(self, dataset_path='.'):
        self.dataset_path = dataset_path
        self.class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
        self.model = None
        self.train_data = None
        self.test_data = None
        self.train_data_aug = None
        self.test_data_aug = None
        
    def setup_data_directories(self):
        """Create train/test split directories"""
        # Create main directories
        train_test_path = os.path.join(self.dataset_path, 'Train_Test_Dataset')
        os.makedirs(train_test_path, exist_ok=True)
        
        # Create train and test directories
        for split in ['Train', 'Test']:
            split_path = os.path.join(train_test_path, split)
            os.makedirs(split_path, exist_ok=True)
            for class_name in self.class_names:
                class_path = os.path.join(split_path, class_name)
                os.makedirs(class_path, exist_ok=True)
        
        return train_test_path
    
    def split_dataset(self, train_test_path):
        """Split dataset into train and test sets (80-20 split)"""
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist")
                continue
                
            files = os.listdir(class_path)
            total_files = len(files)
            train_count = int(total_files * 0.8)
            
            for i, file_name in enumerate(files):
                source = os.path.join(class_path, file_name)
                if i < train_count:
                    dest = os.path.join(train_test_path, 'Train', class_name, file_name)
                else:
                    dest = os.path.join(train_test_path, 'Test', class_name, file_name)
                shutil.copy2(source, dest)
    
    def create_data_generators(self, train_test_path):
        """Create data generators for training and testing"""
        train_dir = os.path.join(train_test_path, 'Train')
        test_dir = os.path.join(train_test_path, 'Test')
        
        # Basic data generators
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_data = train_datagen.flow_from_directory(
            directory=train_dir,
            batch_size=20,
            target_size=(224, 224),
            seed=42,
            color_mode='rgb',
            class_mode='categorical',
            shuffle=True
        )
        
        self.test_data = test_datagen.flow_from_directory(
            directory=test_dir,
            batch_size=20,
            target_size=(224, 224),
            color_mode='rgb',
            seed=42,
            class_mode='categorical',
            shuffle=True
        )
        
        # Augmented data generators
        train_datagen_aug = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            shear_range=5,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )
        
        self.train_data_aug = train_datagen_aug.flow_from_directory(
            directory=train_dir,
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=20,
            shuffle=True,
            seed=42
        )
        
        self.test_data_aug = test_datagen.flow_from_directory(
            directory=test_dir,
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=20,
            shuffle=True,
            seed=42
        )
    
    def create_custom_cnn(self):
        """Create a custom CNN model"""
        model = Sequential([
            Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', input_shape=(224, 224, 3)),
            BatchNormalization(),
            Conv2D(32, 3, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Conv2D(64, 3, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            MaxPool2D(),
            Dropout(0.3),
            Conv2D(64, 3, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Conv2D(128, 3, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            MaxPool2D(),
            Dropout(0.3),
            Flatten(),
            Dropout(0.3),
            Dense(100, activation='relu', kernel_initializer='he_normal'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.8)
        )
        
        return model
    
    def create_xception_model(self):
        """Create Xception transfer learning model"""
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze early layers, unfreeze later layers
        base_model.trainable = True
        set_trainable = False
        
        for layer in base_model.layers:
            if layer.name == 'add_8':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        
        # Create model
        input_layer = base_model.input
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)
        output_layer = Dense(3, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=0.8)
        )
        
        return model
    
    def train_model(self, model, use_augmentation=False, epochs=20):
        """Train the model"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                restore_best_weights=True,
                patience=5,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                mode='min',
                min_lr=0.000001,
                verbose=1
            )
        ]
        
        if use_augmentation:
            train_data = self.train_data_aug
            test_data = self.test_data_aug
        else:
            train_data = self.train_data
            test_data = self.test_data
        
        history = model.fit(
            train_data,
            validation_data=test_data,
            batch_size=train_data.batch_size,
            validation_batch_size=test_data.batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_image(self, image_path, model=None):
        """Predict disease for a single image"""
        if model is None:
            model = self.model
        
        # Load and preprocess image
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image)
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': prediction[0].tolist()
        }
    
    def save_model(self, model, filepath):
        """Save the trained model"""
        model.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model

# Utility functions
def display_class_name(label):
    """Convert one-hot encoded label to class name"""
    class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    if label[0] == 1.0:
        return class_names[0]
    elif label[1] == 1.0:
        return class_names[1]
    else:
        return class_names[2]

def view_random_image(file_path, class_name, show_image=True):
    """View a random image from a class"""
    mypath = os.path.join(file_path, class_name)
    if not os.path.exists(mypath):
        print(f"Path {mypath} does not exist")
        return None
        
    no_of_images = os.listdir(mypath)
    my_image = [i for i in no_of_images if i.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.gif'))]
    
    if not my_image:
        print(f"No images found in {mypath}")
        return None
        
    image = random.choice(my_image)
    img = mpimg.imread(os.path.join(mypath, image))
    
    if show_image:
        plt.imshow(img)
        plt.title(f"Random image of '{class_name}' with shape {img.shape}")
        plt.axis('off')
        plt.show()
    
    return img 