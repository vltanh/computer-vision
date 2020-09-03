import numpy as np
import os
import keras
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.preprocessing import image

def vgg16():
    # Input
    inp = Input(shape=(224,224,3))
    # Block 1
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(inp)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_maxpool')(x)
    # Block 2
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block2_maxpool')(x)
    # Block 3
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block3_maxpool')(x)
    # Block 4
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block4_maxpool')(x)
    # Block 5
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    out = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block5_maxpool')(x)
    
    return Model(inp, out, name='vgg16')

def freeze_model_at(model, l):
    model.trainable = True
    for layer in model.layers[:l]:
        layer.trainable = False

def get_model():
    model = vgg16()
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    freeze_model_at(model, 18)

    inp = Input(shape=(224,224,3), name='inp_image')
    x = model(inp)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(units=100, activation='relu', name='fc1')(x)
    out = Dense(units=2, activation='sigmoid', name='fc2')(x)

    return Model(inp, out)

def get_generator(base_dir, batch_size):
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    img_size = (224, 224)

    # Rescale all images by 1./255
    train_datagen = image.ImageDataGenerator(rescale=1./255)
    validation_datagen = image.ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
                    train_dir,  # Source directory for the training images
                    target_size=img_size,  
                    batch_size=batch_size,
                    # Since we use binary_crossentropy loss, we need binary labels
                    class_mode='binary')

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
                    validation_dir, # Source directory for the validation images
                    target_size=img_size,
                    batch_size=batch_size,
                    class_mode='binary')

    return (train_generator, validation_generator)

def train():
    # Model
    model = get_model()
    model.summary()
    # Generator
    base_dir = "cats_and_dogs_filtered"
    batch_size = 16
    train_generator, val_generator = get_generator(base_dir, batch_size)
    # Compile model
    loss = 'sparse_categorical_crossentropy'
    optimizer = RMSprop(lr=2e-5)
    model.compile(loss=loss, 
                    optimizer=optimizer, 
                    metrics=['accuracy'])
    # Train model
    epochs = 10
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = val_generator.n // batch_size
    history = model.fit_generator(train_generator, 
                                    workers=3, 
                                    epochs=epochs, 
                                    steps_per_epoch=steps_per_epoch, 
                                    validation_steps=validation_steps, 
                                    validation_data=val_generator)

if __name__ == "__main__":
    train()