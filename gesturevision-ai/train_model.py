import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np

def build_model(input_shape=(64, 64, 1), num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(data_dir):
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        return

    # Automatically parse the nested dataset structure
    filepaths = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                # The gesture label is the parent directory's name
                label = os.path.basename(root)
                filepaths.append(filepath)
                labels.append(label)
    
    if not filepaths:
        print(f"No images found in {data_dir}.")
        return

    df = pd.DataFrame({
        'filename': filepaths,
        'class': labels
    })

    print(f"Found {len(df)} images across {df['class'].nunique()} classes.")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        y_col='class',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        y_col='class',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    model = build_model(num_classes=len(train_gen.class_indices))
    model.summary()
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks
    )
    
    os.makedirs('saved_model', exist_ok=True)
    model.save('saved_model/gesture_model.h5')
    
    # Save history
    np.save('saved_model/history.npy', history.history)
    print("Training complete and model saved.")

if __name__ == "__main__":
    dataset_path = r'C:\Users\Mahitha\Downloads\archive (3)\leapGestRecog'
    train(dataset_path)
