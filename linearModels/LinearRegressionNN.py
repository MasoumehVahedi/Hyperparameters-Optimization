import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import MeanSquaredError



# Load data
df = pd.read_csv("/content/all_median_0.001%.csv")

# Assuming df has columns ['AMS', 'ADNN', 'StandardDev1', 'StandardDev2', 'B', 'T', 'C*']
X = df[['AMS', 'ADNN', 'VMS', 'VDNN']].values
y = df[['T', 'Bf']].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


# Define neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2)  # Output layer with two values: B and T
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[MeanSquaredError()])


# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2, verbose=1)

# Evaluate the model on test data
test_loss, test_mse = model.evaluate(X_test, y_test)
print(f"Test Mean Squared Error: {test_mse}")

# Predict on the test set
y_pred_scaled = model.predict(X_test)

# Inverse transform to get predictions in the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# Display predictions alongside actual values
results = pd.DataFrame({'T_actual': y_test_original[:, 0], 'T_pred': y_pred[:, 0],
                        'Bf_actual': y_test_original[:, 1], 'Bf_pred': y_pred[:, 1]})
print(results.head())

# Predict on all data
X_all = df[['AMS', 'ADNN', 'VMS', 'VDNN']].values
X_all_scaled = scaler_X.transform(X_all)
y_pred_all_scaled = model.predict(X_all_scaled)
y_pred_all = scaler_y.inverse_transform(y_pred_all_scaled)

# Add predictions to the DataFrame
df['T_pred'] = y_pred_all[:, 0]
df['Bf_pred'] = y_pred_all[:, 1].astype(int)

# Save the updated DataFrame to a new CSV file
df.to_csv("predictions_with_NN_3%.csv", index=False)

# Optionally, display the first few rows to verify
print(df.head())