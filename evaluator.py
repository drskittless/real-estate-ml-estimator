import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:
    def evaluate(self, model, X_test, y_test):

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")

        return predictions

    def plot_actual_vs_predicted(self, y_test, predictions):
        plt.figure()
        plt.scatter(y_test, predictions)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted Prices")
        plt.show()

    def plot_feature_importance(self, model, feature_names):
        if hasattr(model, "feature_importances_"):
            plt.figure()
            plt.barh(feature_names, model.feature_importances_)
            plt.title("Feature Importance")
            plt.show()