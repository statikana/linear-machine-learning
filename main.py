import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm

data_root = "https://github.com/ageron/data/raw/main"
lifesat = pd.read_csv(f"{data_root}/lifesat/lifesat.csv")

x_title = "GDP per capita (USD)"
y_title = "Life satisfaction"
x = lifesat[[x_title]].values
y = lifesat[[y_title]].values

lifesat.plot(
    kind="scatter",
    grid=True,
    x=x_title,
    y=y_title
)
plt.axis([23500, 62500, 4, 9])
plt.show()

model = lm.LinearRegression()
model.fit(x, y)

x_hat = [[37655.2]]
print("y_hat:", model.predict(x_hat))