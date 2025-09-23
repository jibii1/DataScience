import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(X_train,y_train), (X_test,y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype("float32") / 255.0
x_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)


#Build a simple CNN model
model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),  #convolutional layer
    layers.MaxPooling2D((2,2)), #pooling layer
    layers.Flatten(),            #flatten into ID
    layers.Dense(64, activation='relu'), #fully connected layers
    layers.Dense(10, activation='softmax')  #output layer(10 classes)    
])

#compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64, #faster traning
    validation_data=(X_test,y_test),
    verbose=1   #shows progress bar
)

#Evulate on test data
test_loss,test_acc=model.evaluate(X_test,y_test,verbose=0)
print("Test Accuracy:",round(test_acc * 100,2),"%")


prediction=model.predict(X_test[:1])  #get prediction possibilities
predicted_label=prediction.argmax()  #find the most likely class

plt.imshow(X_test[0].reshape(28,28),cmap="gray")
plt.title("prediction:" + str(predicted_label))
plt.axis("off")
plt.show()