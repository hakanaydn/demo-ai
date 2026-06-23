import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

N = 20000
x1 = np.random.uniform(0.0, 10.0, N)
x2 = np.random.uniform(0.0, 10.0, N)
y = x1 + x2

X = np.column_stack((x1, x2))
idx = np.random.permutation(N)
split = int(N * 0.8)
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), verbose=2)

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.6f}")
print(f"Test MAE: {mae:.6f}")

model.save('saved_model')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\nPredictions:\n")
test_samples = np.array([[3.0, 4.0], [0.0, 0.0], [10.0, 10.0], [5.5, 3.5], [2.0, 8.0]])
preds = model.predict(test_samples, verbose=0)
for i, (a, b) in enumerate(test_samples):
    print(f"{a} + {b} = {preds[i][0]:.4f} (expected {a+b})")
