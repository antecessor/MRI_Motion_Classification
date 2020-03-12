import matplotlib.pyplot as plt
from keras import metrics, losses
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential


def CNNTrain(DataLoaderTrain, DataLoaderTest, num_classes, show=False):
    input_shape = (256, 256, 1)
    model = Sequential()

    model.add(Conv2D(16, 7, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(8, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(4, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss=losses.categorical_crossentropy, optimizer='adam',
                  metrics=[metrics.Recall()])
    print(model.summary())
    epochs = 50

    history = model.fit(DataLoaderTrain,
                        epochs=epochs, validation_data=DataLoaderTest)

    if show:
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['recall_1'])
        plt.plot(history.history['val_recall_1'])
        plt.title('model performance')
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    # out = np.round(model.predict(DataLoaderTest, batch_size=batch_size))
    # cm = ConfusionMatrix(actual_vector=np.argmax(Y_test, axis=1), predict_vector=np.argmax(out, axis=1))
    # print(cm)
    return model
