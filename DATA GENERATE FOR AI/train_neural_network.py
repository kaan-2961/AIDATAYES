import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH     = "500_node_vrp_dataset.csv"  # dataset path
MODEL_PATH    = "best_model_with_weights.h5"
SCALER_PATH   = "scaler.pkl"
TEST_SIZE     = 0.1    # 10% for test, 90% train
RANDOM_STATE  = 42
# Optional manual feature subset (None = use all)
SELECT_FEATURES = None

# -----------------------------
# Load and prepare data
# -----------------------------
df = pd.read_csv(DATA_PATH)
all_features   = df.drop(columns=["instance_id", "k_true"]).columns.tolist()
feature_names  = SELECT_FEATURES if SELECT_FEATURES else all_features
X = df[feature_names].values
y = df["k_true"].values

# 90/10 train-test split
tX, tX_test, ty, ty_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(tX)
X_test_scaled  = scaler.transform(tX_test)
pickle.dump(scaler, open(SCALER_PATH, "wb"))

# -----------------------------
# Feature-weighting layer
# -----------------------------
class FeatureWeighting(Layer):
    def __init__(self, n_features, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.n_features,), initializer="ones",
            trainable=True, name="feature_weights"
        )
        super().build(input_shape)
    def call(self, inputs):
        return inputs * self.w

# -----------------------------
# Build & train neural model
# -----------------------------
n_inputs = Input(shape=(X_train_scaled.shape[1],), name="features")
x = FeatureWeighting(X_train_scaled.shape[1], name="feat_weights")(n_inputs)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(1, activation="relu", name="k_pred")(x)
model = Model(n_inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

print("Training neural model with FeatureWeighting...")
model.fit(
    X_train_scaled, ty,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=2
)
_, mae_nn = model.evaluate(X_test_scaled, ty_test, verbose=0)
print(f"Neural Test MAE: {mae_nn:.4f}\n")

# -----------------------------
# Feature importances (learned weights)
# -----------------------------
w = model.get_layer("feat_weights").get_weights()[0]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'nn_weight': w
}).sort_values(by='nn_weight', ascending=False)

print("Learned feature weights:")
print(importance_df.to_string(index=False), end="\n\n")

# -----------------------------
# Sample predictions
# -----------------------------
y_pred = model.predict(X_test_scaled).flatten()
print("Sample predictions:")
for idx in range(min(len(ty_test), 20)):
    true_k = ty_test[idx]
    raw = y_pred[idx]
    rnd = int(np.rint(raw))
    print(f"True k: {true_k}, Raw pred: {raw:.2f}, Rounded pred: {rnd}")
# -----------------------------
# Done
