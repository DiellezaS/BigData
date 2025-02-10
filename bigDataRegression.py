import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Leximi i dataset-it
file_path = "real_estate.csv"
df = pd.read_csv(file_path)

# Shfaqja e disa rreshtave për të kuptuar strukturën e dataset-it
print(df.head())
print(df.describe())

# Line charts, histograms
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(data=df, x="House Age", y="House Price of Unit Area", ax=axes[0])
axes[0].set_title("Line Chart: House Age vs Price")
sns.histplot(df["House Price of Unit Area"], bins=30, kde=True, ax=axes[1])
axes[1].set_title("Histogram of House Prices")
plt.show()

# Scatter plot, Q-Q plot, heatmap, bar graph
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
sns.scatterplot(data=df, x="House Age", y="House Price of Unit Area", ax=axes[0, 0])
axes[0, 0].set_title("Scatter Plot: House Age vs Price")
stats.probplot(df["House Price of Unit Area"], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Q-Q Plot of House Prices")
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=axes[1, 0])
axes[1, 0].set_title("Correlation Heatmap")
sns.barplot(x=df.columns, y=df.var(), ax=axes[1, 1])
axes[1, 1].set_title("Variance of Variables")
plt.xticks(rotation=45)
plt.show()

# Matrica e korrelacionit
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matrica e Korrelacionit")
plt.show()

# Volatility over time graph
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="Transaction Date", y="House Price of Unit Area")
plt.title("Volatility Over Time: House Prices")
plt.show()

# Advanced visualization using plotly
fig = px.scatter(df, x="Distance to the Nearest MRT Station", y="House Price of Unit Area",
                 color="Number of Convenience Stores", title="Interactive Scatter Plot")
fig.show()

# Pairplot for variable relationships
sns.pairplot(df, diag_kind='kde')
plt.show()

# Box plots for variable distributions
plt.figure(figsize=(15, 5))
sns.boxplot(data=df[['House Age', 'Distance to the Nearest MRT Station', 'House Price of Unit Area']])
plt.title("Box Plot of Key Variables")
plt.show()

# Statistical Metrics
correlation_matrix = df.corr()
variance = df.var()
standard_deviation = df.std()
covariance_matrix = df.cov()
coefficient_of_variation = standard_deviation / df.mean()

print("Matrica e korrelimit: ")
print(correlation_matrix)
print("Varianca: ")
print(variance)
print("Devijimi standard: ")
print(standard_deviation)
print("Matrica e kovariances: ")
print(covariance_matrix)
print("Koeficienti i variances: ")
print(coefficient_of_variation)

# Visualizing statistical metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(x=variance.index, y=variance.values, ax=axes[0])
axes[0].set_title("Variance of Variables")
plt.xticks(rotation=45)
sns.barplot(x=standard_deviation.index, y=standard_deviation.values, ax=axes[1])
axes[1].set_title("Standard Deviation of Variables")
plt.xticks(rotation=45)
sns.barplot(x=coefficient_of_variation.index, y=coefficient_of_variation.values, ax=axes[2])
axes[2].set_title("Coefficient of Variation")
plt.xticks(rotation=45)
plt.show()

# Përgatitja e të dhënave për modelim
X = df[['House Age', 'Distance to the Nearest MRT Station', 'Number of Convenience Stores']]
y = df['House Price of Unit Area']

# Ndarja e të dhënave në trajnimi dhe testim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizimi i të dhënave
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Trajnimi dhe parashikimi për secilin model
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Vizualizimi i parashikimeve
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_test, color="red", label="Vlera reale", alpha=0.7)
    sns.scatterplot(x=y_test, y=y_pred, color="green", label="Vlera e parashikuar", alpha=0.7)
    plt.xlabel("Vlerat reale të çmimit")
    plt.ylabel("Vlerat e parashikuara të çmimit")
    plt.title(f"{name} - Vlera reale vs. Vlera e parashikuar")
    plt.show()

# Regresioni Linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Regresioni Polinomial
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
polynomial_model = LinearRegression()
polynomial_model.fit(X_poly_train, y_train)
y_pred_poly = polynomial_model.predict(X_poly_test)

# Decision Tree Regression
decision_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)

# models and predictions
models = {"Linear Regression": y_pred_linear, "Polynomial Regression": y_pred_poly, "Decision Tree": y_pred_tree}

evaluation_results = {}

for name, y_pred in models.items():
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    evaluation_results[name] = {
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2
    }

results_df = pd.DataFrame(evaluation_results).T  # Transpose to make models as rows
print(results_df)

# Vizualizimi i rezultateve
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, color="red", alpha=0.5, label="Linear Regression")
plt.scatter(y_test, y_pred_poly, color="blue", alpha=0.5, label="Polynomial Regression")
plt.scatter(y_test, y_pred_tree, color="green", alpha=0.5, label="Decision Tree")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.xlabel("Çmimet reale")
plt.ylabel("Çmimet e parashikuara")
plt.legend()
plt.title("Krahasimi i Parashikimeve: Linear vs Polynomial vs Decision Tree")
plt.show()
