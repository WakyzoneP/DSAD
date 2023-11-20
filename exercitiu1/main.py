from pandas import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import *

def print_data(data):
    print(data)
    print("==================================================================================================================")


data_frame = read_csv("data.csv")
print_data(data_frame)
print_data(data_frame.describe())

print_data(data_frame.dtypes)


gpa_mean = data_frame['gpa'].mean()
print_data(f"Media este: {gpa_mean}")

age_max = data_frame['age'].max()
print_data(f"Maximul este: {age_max}")

print_data(data_frame.head(20))

print_data(data_frame[data_frame["gpa"] > 3])

print_data(data_frame.sort_values(by=['age']))

data_frame['postal_code'].fillna('000-000', inplace=True)
print_data(data_frame)

plt.scatter(data_frame[data_frame['favorite_color'] == 'red']['age'], data_frame[data_frame['favorite_color'] == 'red']['gpa'], c = 'red', label = 'Red')
plt.scatter(data_frame[data_frame['favorite_color'] == 'blue']['age'], data_frame[data_frame['favorite_color'] == 'blue']['gpa'], c = 'blue', label = 'Blue')
plt.xlabel('Gpa value')
plt.ylabel('Age value')
plt.title('Title')
plt.legend()
plt.show()

favorite_colors_count = data_frame['favorite_color'].value_counts()
plt.bar(favorite_colors_count.index, favorite_colors_count.values, color="red")
plt.title("Titlu")
plt.xlabel("Color")
plt.ylabel("Count")
plt.show()

age_and_gpa_data_frame = data_frame[['age', 'gpa']]
scaler = StandardScaler()
scaler.fit(age_and_gpa_data_frame)
scaled_data = scaler.transform(age_and_gpa_data_frame)
print_data(scaled_data)