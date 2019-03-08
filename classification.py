import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

base_model = Xception(include_top=False, weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(8, activation='softmax')(x)

model = Model(input=base_model.input, output=prediction)
model.summary()

for layer in base_model.layers:
    layer.trainable = False

history = model.compile(optimizer=Adam(),
                        loss=keras.losses.categorical_crossentropy,
                        metrics=['acc'])
history = model.fit_generator()