from data_loader import DataLoader
from preprocessing import Preprocessor
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from evaluator import Evaluator
from dashboard import Dashboard


def main():

    loader = DataLoader("housing.csv")
    df = loader.load_data()

    # Feature Engineering
    fe = FeatureEngineer()
    df = fe.add_features(df)

    # Preprocessing
    pre = Preprocessor()
    df = pre.handle_missing(df)
    df = pre.remove_outliers(df)
    df = pre.encode_categorical(df)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_scaled = pre.scale_features(X)
    X_train, X_test, y_train, y_test = pre.split(X_scaled, y)

    # Model Training
    trainer = ModelTrainer()
    models = trainer.train(X_train, y_train)

    evaluator = Evaluator()

    # Choose best model (Random Forest typically strongest)
    best_model = models["Random Forest"]

    predictions = evaluator.evaluate(best_model, X_test, y_test)

    # Dashboard
    dashboard = Dashboard(df, best_model, X_test, y_test, predictions)
    dashboard.plot_dashboard()


if __name__ == "__main__":
    main()