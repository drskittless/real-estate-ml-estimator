import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dashboard:

    def __init__(self, df, model, X_test, y_test, predictions):
        self.df = df
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = predictions

    def plot_dashboard(self):

        fig = plt.figure(figsize=(18, 12))

        # ---------------------------------------------------
        # 1️⃣ Price Distribution
        # ---------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.hist(self.df["price"], bins=30)
        ax1.set_title("Price Distribution")
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Frequency")

        # ---------------------------------------------------
        # 2️⃣ Correlation Heatmap (Manual)
        # ---------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        corr = self.df.corr(numeric_only=True)
        cax = ax2.matshow(corr)
        fig.colorbar(cax, ax=ax2)
        ax2.set_xticks(range(len(corr.columns)))
        ax2.set_yticks(range(len(corr.columns)))
        ax2.set_xticklabels(corr.columns, rotation=90)
        ax2.set_yticklabels(corr.columns)
        ax2.set_title("Correlation Heatmap")

        # ---------------------------------------------------
        # 3️⃣ Actual vs Predicted
        # ---------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.scatter(self.y_test, self.predictions)
        ax3.set_xlabel("Actual Price")
        ax3.set_ylabel("Predicted Price")
        ax3.set_title("Actual vs Predicted")

        # ---------------------------------------------------
        # 4️⃣ Residual Error Distribution
        # ---------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        residuals = self.y_test - self.predictions
        ax4.hist(residuals, bins=30)
        ax4.set_title("Residual Error Distribution")
        ax4.set_xlabel("Error")

        # ---------------------------------------------------
        # 5️⃣ Feature Importance (If Tree-Based Model)
        # ---------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)

        if hasattr(self.model, "feature_importances_"):
            feature_names = self.df.drop("price", axis=1).columns
            importances = self.model.feature_importances_
            ax5.barh(feature_names, importances)
            ax5.set_title("Feature Importance")
        else:
            ax5.text(0.5, 0.5, "Feature Importance\nNot Available",
                     horizontalalignment='center',
                     verticalalignment='center')
            ax5.set_title("Feature Importance")

        # ---------------------------------------------------
        # 6️⃣ Risk Score by Area Group
        # ---------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)

        if "area_group" in self.df.columns:
            risk = self.df.groupby("area_group")["price"].std()
            ax6.bar(risk.index.astype(str), risk.values)
            ax6.set_title("Locality Risk Score (Std Dev)")
            ax6.set_ylabel("Price Volatility")

        plt.tight_layout()
        plt.show()