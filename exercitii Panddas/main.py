from pandas import DataFrame, read_csv, to_datetime
import matplotlib.pyplot as plt

# Load a CSV file called "data.csv" into a Pandas DataFrame. Then, perform the following tasks:
data_frame = read_csv('data.csv')
print(data_frame)
print("==============================================================================================")

# Display the first 5 rows of the DataFrame.
print(data_frame.head(5))
print("==============================================================================================")

# Display the basic statistics (mean, median, etc.) for numeric columns.
print(data_frame.describe())
print("==============================================================================================")

# Select only the rows where the 'age' column is greater than 30.
print(data_frame[data_frame['age'] > 30])
print("==============================================================================================")

# Filter the data to display only records where 'gender' is 'Female' and 'education' is 'Bachelor's Degree'.
print(data_frame[(data_frame['gender'] == 'Female') & (data_frame['education'] == "Bachelor's Degree")])
print("==============================================================================================")

# Create a new DataFrame that includes only the 'name' and 'salary' columns.
new_data_frame = data_frame[['name', 'salary']]
print(new_data_frame)
print("==============================================================================================")

# Group the data by 'gender' and calculate the average salary for each gender.
gender_group = data_frame.groupby('gender')
print(gender_group['salary'].mean())
print("==============================================================================================")

# Group the data by 'education' and find the maximum age for each education level.
education_group = data_frame.groupby('education')
print(education_group['age'].max())
print("==============================================================================================")

# Calculate the total count of individuals for each combination of 'gender' and 'education'.
print(gender_group['education'].value_counts())
print(education_group['gender'].value_counts())
print("==============================================================================================")

# Remove duplicates from the dataset.
data_frame.drop_duplicates(inplace=True)
print(data_frame)
print("==============================================================================================")

# Fill missing values in the 'income' column with the mean income.
mean_income = data_frame['income'].mean()
data_frame['income'].fillna(mean_income, inplace=True)
print(data_frame)
print("==============================================================================================")

# Create a new column 'age_group' that categorizes individuals into age groups (e.g., 'Under 30', '30-40', 'Over 40').
data_frame['age_group'] = data_frame['age'].apply(lambda x: 'Under 30' if x < 30 else ('30-40' if x < 40 else 'Over 40'))
print(data_frame)
print("==============================================================================================")

# A histogram of the 'age' column.
plt.hist(data_frame['age'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()

# A bar chart showing the count of each unique value in the 'education' column.
education_counts = data_frame['education'].value_counts()
plt.bar(education_counts.index, education_counts.values)
plt.xlabel('Education')
plt.ylabel('Count')
plt.title('Count of Each Unique Value in the Education Column')
plt.show()

# A scatter plot of 'age' vs. 'income' with different colors for 'gender'.
plt.scatter(data_frame[data_frame['gender'] == 'Male']['age'], data_frame[data_frame['gender'] == 'Male']['income'], c='blue', label='Male')
plt.scatter(data_frame[data_frame['gender'] == 'Female']['age'], data_frame[data_frame['gender'] == 'Female']['income'], c='red', label='Female')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Scatter Plot of Age vs. Income')
plt.legend()
plt.show()

# Merge the two DataFrames using a common column (e.g., 'user_id').
second_data_frame = read_csv('data2.csv')
print(second_data_frame)
print("==============================================================================================")
merged_data_frame = data_frame.merge(second_data_frame, on='user_id')
print(merged_data_frame)
print("==============================================================================================")

# Convert a column with date strings into a datetime data type.
data_frame['date_column'] = to_datetime(data_frame['date_column'])
print(data_frame)
print(data_frame.dtypes)
print("==============================================================================================")

# Set the date column as the index of the DataFrame.
data_frame.set_index('date_column', inplace=True)
print(data_frame)
print("==============================================================================================")

time_data_frame = DataFrame()
time_data_frame['date'] = data_frame.index
time_data_frame['salary'] = data_frame['income'].values
time_data_frame.set_index('date', inplace=True)
print(time_data_frame)
print("==============================================================================================")

# Resample the data to calculate the monthly average.
monthly_average = time_data_frame.resample('M').mean()
print(monthly_average)
