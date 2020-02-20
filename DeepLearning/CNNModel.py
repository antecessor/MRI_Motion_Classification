from keras import metrics
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from tensorflow import keras


def lstmTrain(DataLoaderTrain, DataLoaderTest, num_classes):
    input_shape = (256, 256, 1)
    model = Sequential()

    model.add(Conv2D(32, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(128, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='nadam',
                  metrics=[metrics.Recall()])
    print(model.summary())

    batch_size = 128
    epochs = 200

    model.fit(DataLoaderTrain,
              epochs=epochs, validation_data=DataLoaderTest)

    # out = np.round(model.predict(DataLoaderTest, batch_size=batch_size))
    # cm = ConfusionMatrix(actual_vector=np.argmax(Y_test, axis=1), predict_vector=np.argmax(out, axis=1))
    # print(cm)
    return model
