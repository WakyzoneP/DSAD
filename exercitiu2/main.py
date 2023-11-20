from pandas import *
import matplotlib.pylab as plt
from sklearn.preprocessing import *
import datetime

def print_data(data):
    print(data)
    print("=================================================================================================================================")

data_frame = read_csv("data.csv")
print_data(data_frame)
print_data(data_frame.describe())

height_mean = data_frame['height'].mean()
print_data(f"Media  inaltime: {height_mean}")

data_frame["height"].fillna(height_mean, inplace=True)
print_data(data_frame.head(30))

plt.hist(data_frame["height"])
plt.xlabel("Inaltime")
plt.ylabel("Frecventa")
plt.title("Histograma inaltime")
plt.show()

height_data_frame = DataFrame()

height_data_frame["height"] = data_frame["height"]
height_data_frame["height"] = height_data_frame["height"].apply(lambda x: round(x / 10) * 10)
height_group = height_data_frame.groupby('height')
height_counts = height_group.value_counts()

plt.bar(height_counts.index, height_counts.values, color = 'green')
plt.xlabel("Inalitme")
plt.ylabel("Frecventa")
plt.title("Titlu")
plt.show()

scaler = StandardScaler()
scaler.fit(data_frame)
scaled_data = scaler.transform(data_frame)
print_data(scaled_data[20:])

start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime(2022, 12, 31)
date_series = date_range(start=start_date, end=end_date, freq='2D')
data_frame["date"] = date_series[:100]

data_frame.set_index("date", inplace=True)
print_data(data_frame.head(30))

height_mean = data_frame.resample('M').mean()
print_data(height_mean)