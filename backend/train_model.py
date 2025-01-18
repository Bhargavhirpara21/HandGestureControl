# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.callbacks import EarlyStopping
# import numpy as np
#
# # Load and preprocess data
# data = pd.read_csv('gesture_data.csv')
# data = data.dropna()  # Handle missing values
#
# X = data.iloc[:, :-1].values  # Features
# y = data.iloc[:, -1].astype('int').values  # Labels
#
# # Split dataset
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% each
#
# # Normalize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)
#
# # Save scaler for consistent preprocessing
# np.save('scaler_mean.npy', scaler.mean_)
# np.save('scaler_scale.npy', scaler.scale_)
#
# # Define model
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dense(64, activation='relu'),
#     Dense(3, activation='softmax')  # Three classes: 0, 1, 2
# ])
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Train model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)
# model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])
#
# # Save model
# model.save('gesture_model.keras')
# print("Model training complete and saved as 'gesture_model.keras'.")
#
# # Evaluate model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {loss:.2f}, Test Accuracy: {accuracy:.2f}")
#
#





#working
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Load data
data = pd.read_csv('gesture_data.csv')

# Drop missing values
data = data.dropna()

X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].astype('int').values  # Labels

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% val, 15% test

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save scaler for consistent preprocessing during testing
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Define multiclass classification model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Three classes: 0, 1, 2
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save model
model.save('gesture_model.keras')
print("Model training complete and saved as gesture_model.keras")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Loss: {test_loss:.2f}")





# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.callbacks import EarlyStopping
# import numpy as np
#
# # Load data
# data = pd.read_csv('gesture_data.csv')
#
# # Check dataset
# print("Dataset Preview:")
# print(data.head())
#
# print("\nLabel Distribution:")
# print(data.iloc[:, -1].value_counts())
#
# # Drop missing values
# data = data.dropna()
#
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].astype('int').values
#
# # Normalize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # Save scaler for consistent preprocessing during testing
# np.save('scaler_mean.npy', scaler.mean_)
# np.save('scaler_scale.npy', scaler.scale_)
#
# # Define binary classification model
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X.shape[1],)),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')  # Sigmoid for binary classification
# ])
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Use class weights to address imbalance
# class_weights = {0: 1.0, 1: len(data[data.iloc[:, -1] == 0]) / len(data[data.iloc[:, -1] == 1])}
#
# # Train model
# early_stopping = EarlyStopping(monitor='loss', patience=5)
# model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)
#
# # Save model
# model.save('gesture_model.keras')
#
# print("Model training complete and saved as gesture_model.keras")


'''
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Load data
data = pd.read_csv('gesture_data.csv')

# Check dataset
print("Dataset Preview:")
print(data.head())
print("\nLabel Distribution:")
print(data.iloc[:, -1].value_counts())

# Drop missing values
data = data.dropna()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].astype('int').values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler for consistent preprocessing during testing
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Define binary classification model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Use class weights to address imbalance
class_weights = {0: 1.0, 1: len(data[data.iloc[:, -1] == 0]) / len(data[data.iloc[:, -1] == 1])}

# Train model
early_stopping = EarlyStopping(monitor='loss', patience=5)
history = model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)

# Save model
model.save('gesture_model.keras')
print("Model training complete and saved as gesture_model.keras")
'''

