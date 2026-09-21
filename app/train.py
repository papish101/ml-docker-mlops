from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully")
