import os
import pandas 
import argparse
from tensorflow.keras.optimizers import Adam 
from utils import dataset
from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow import keras

parser = argparse.ArgumentParser(description='Train the model for self driving car')
parser.add_argument('--train_csv_file', metavar='string', type=str, required=True, help='Path to train driving log csv file')
parser.add_argument('--val_csv_file', metavar='string', type=str, required=True, help='Path to validation driving log csv file')
parser.add_argument('--batchsize', metavar='int', type=int, default=32, help='Batch size for training, default 32')
parser.add_argument('--epochs', metavar='int', type=int, default=30, help='epochs for training, default 30')

args = parser.parse_args()

batch_size = args.batchsize
epochs = args.epochs

train_csv_file_path = args.train_csv_file
val_csv_file_path = args.val_csv_file

train_dataset = dataset(train_csv_file_path, batch_size)
if val_csv_file_path is not None:
    val_dataset = dataset(val_csv_file_path, batch_size)
else:
    val_dataset = None

img_shape = train_dataset.get_img_size()

model = create_model(inputShape=img_shape)
#model = keras.models.load_model('models/save_at50.keras')
model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    os.path.join('models5', 'save_at{epoch}.keras'),
    save_best_only=True,
    save_weights_only=False,
    verbose=0,
    monitor='val_loss',
    mode='min'
)

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.0,
    monitor='val_loss',
    mode='min',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.000001,
    verbose=0,
    mode='min'
)

result = model.fit(
    train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size,
    callbacks=[checkpoint, early_stopping, reduce_lr], verbose = 1
)

