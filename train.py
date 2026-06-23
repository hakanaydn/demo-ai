import numpy as np
import tensorflow as tf

MODEL_VERSION = "1.0.0"
np.random.seed(42)
tf.random.set_seed(42)

N = 50000
x1 = np.random.uniform(0.0, 1000.0, N)
x2 = np.random.uniform(0.0, 1000.0, N)
y = x1 + x2

X = np.column_stack((x1, x2))
X_norm = X / 1000.0
y_norm = y / 2000.0

idx = np.random.permutation(N)
split = int(N * 0.8)
X_train, X_test = X_norm[idx[:split]], X_norm[idx[split:]]
y_train, y_test = y_norm[idx[:split]], y_norm[idx[split:]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), verbose=2)

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss (normalized): {loss:.6f}")
print(f"Test MAE (normalized): {mae:.6f}")

model.save('saved_model')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\nModel v{MODEL_VERSION} egitildi\n")

test_samples = np.array([[300.0, 400.0], [0.0, 0.0],
                         [1000.0, 1000.0], [500.0, 500.0],
                         [123.0, 456.0], [999.0, 1.0]])
preds = model.predict(test_samples / 1000.0, verbose=0)
for i, (a, b) in enumerate(test_samples):
    pred = preds[i][0] * 2000.0
    expected = a + b
    err = pred - expected
    err_pct = abs(err / expected) * 100.0 if expected != 0 else abs(err) * 100.0
    print(f"{a:.1f} + {b:.1f} = {pred:.2f} (expected {expected:.1f}, "
          f"error: {err:.2f} ({err_pct:.2f}%))")
