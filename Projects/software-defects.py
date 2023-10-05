import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

def load_dataset(filepath):
    """Load the dataset."""
    print(f"\nLoading dataset from {filepath}")
    dataset = pd.read_csv(filepath)
    print(f"Dataset info:\n{dataset.info()}")
    return dataset


def check_missing_values(dataset):
    """Check for missing values in the dataset."""
    print("\nChecking for missing values...")
    missing_values = dataset.isnull().sum()
    if missing_values.any():
        print(missing_values[missing_values > 0])
    else:
        print("- No missing values detected.")


def convert_column_type(dataset, column, dtype):
    """Convert a column's data type."""
    dataset[column] = dataset[column].astype(dtype)
    return dataset


def truncate(dataset, lower_quantile=0.0001, upper_quantile=0.9999):
    """Remove outliers from the dataset."""
    print(f"\nTruncating dataset between quantiles [{lower_quantile}, {upper_quantile}]...")
    lower_bound = dataset.quantile(lower_quantile)
    upper_bound = dataset.quantile(upper_quantile)
    filtered_dataset = dataset[(dataset >= lower_bound) & (dataset <= upper_bound)].dropna()
    print(f"Before: {len(dataset)}\nAfter: {len(filtered_dataset)}")
    return filtered_dataset


def calculate_statistics(dataset):
    """Calculate and print statistics for each column in the dataset."""
    print("\nCalculating statistics...")
    stats = dataset.describe().T[['mean', 'min', '50%', 'max', 'std']]
    stats['needs_scaling'] = (stats['max'] - stats['min']) > 2.5 * stats['std']
    print(stats)
    return stats


def scale_columns(dataset, stats):
    """Normalize columns that need scaling."""
    print("\nScaling columns...")
    scaler = StandardScaler()
    cols_to_scale = stats[stats['needs_scaling']].index.tolist()
    dataset[cols_to_scale] = scaler.fit_transform(dataset[cols_to_scale])
    return dataset


def identify_correlated_columns(dataset, threshold=0.9):
    """Identify and return columns that are highly correlated."""
    print("\nIdentifying correlated columns...")
    corr_matrix = dataset.corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    to_drop = [column for column in corr_matrix.columns if any(corr_matrix.where(upper_triangle)[column] > threshold)]
    return to_drop


def preprocess_data(dataset, target_col=None, is_train=True):
    """Preprocess the input dataset."""
    check_missing_values(dataset)
    ids = dataset.get("id", None)
    dataset = dataset.drop(columns=["id"], errors='ignore')
    if is_train:
        dataset = convert_column_type(dataset, target_col, 'int')
        dataset = truncate(dataset)
    stats = calculate_statistics(dataset)
    dataset = scale_columns(dataset, stats)
    return dataset if is_train else (dataset, ids)


def build_enhanced_model(input_dim):
    """Build and return a modified neural network model."""
    model = Sequential([
        # First dense layer with 256 neurons and L1 and L2 regularization
        Dense(256, input_dim=input_dim, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.25),
        
        # Second dense layer with 64 neurons and L1 and L2 regularization
        Dense(128, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.25),

        # Fourth dense layer with 32 neurons and L1 and L2 regularization
        Dense(32, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Activation('relu'),

        # Output layer
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_train_val_split(dataset, target_col, test_size=0.2, random_state=42):
    """Splits the dataset into training and validation sets."""
    X = dataset.drop(target_col, axis=1)
    y = dataset[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def main():
    # Paths
    train_filepath = os.path.join("..", "Data", "SoftwareDefects", "train.csv")
    test_filepath = os.path.join("..", "Data", "SoftwareDefects", "test.csv")

    # Preprocessing
    train_dataset = load_dataset(train_filepath)
    train_dataset = preprocess_data(train_dataset, target_col='defects')
    X_train, X_val, y_train, y_val = get_train_val_split(train_dataset, 'defects')
    
    test_dataset, test_ids = load_dataset(test_filepath), None
    if "id" in test_dataset.columns:
        test_dataset, test_ids = preprocess_data(test_dataset, is_train=False)
    X_test = test_dataset.drop('defects', axis=1, errors='ignore')

    # Model Building
    model = build_enhanced_model(X_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32, callbacks=callbacks)

    # Evaluation
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss:.4f}\nValidation Accuracy: {val_accuracy:.4f}")

    # Save & Predict
    model.save('trained_model_software-defects.h5')
    y_pred = model.predict(X_test)
    predictions_df = pd.DataFrame({'id': test_ids, 'prediction': y_pred.squeeze()})
    predictions_df.to_csv(os.path.join("..", "Data", "SoftwareDefects", "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
