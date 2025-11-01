import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1) Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# 2) Model: 28x28 -> 784 -> 256 -> 10
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax'),
])

# 3) Train
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("src/pipeline/weight/mnist256.keras", save_best_only=True)
]

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2,
    callbacks=callbacks,
    shuffle=True
)

model.save("src/pipeline/weight/mnist256.keras")

y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis=1)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test,y_pred))