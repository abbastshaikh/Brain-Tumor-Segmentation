import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger

import warnings
warnings.filterwarnings(action = 'once')

from datagenerator import AxialGenerator
from models import unet
from metrics import dice, loss_func

mount = "/data"

name_mapping = pd.read_csv(mount + "/name_mapping.csv")
ids = name_mapping['BraTS_2020_subject_ID']

train_ids = ids[0:270]
validation_ids = ids[270:300]

training_generator = AxialGenerator(train_ids, mount)
validation_generator = AxialGenerator(validation_ids, mount)


dims = (240, 240, 1)
model = unet((240, 240, 1))
# model = load_model('.h5', custom_objects={'loss_func': loss_func, 'dice': dice})
# model.trainable = True
# model.summary()


learning_rate = 1e-4
num_epochs = 10

opt = Adam(learning_rate = learning_rate)
model.compile(optimizer=opt, loss=loss_func, metrics=[dice])

keras_callbacks = [
    CSVLogger('training.csv', separator=','),
    ModelCheckpoint('unet-{epoch:02d}.h5', save_best_only=False)]

model.fit_generator(generator = training_generator, 
                    validation_data = validation_generator,
                    epochs = num_epochs, 
                    callbacks = keras_callbacks,
                    verbose = 1,
                    use_multiprocessing=True)