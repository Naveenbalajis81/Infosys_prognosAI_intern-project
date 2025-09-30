from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def model_training():
    iris = load_iris()
    x=iris.data
    y=iris.target

    X_train, X_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy*100:.2f}%")

    joblib.dump(model, "iris_model.joblib")
    print("Model saved as iris_model.joblib")
