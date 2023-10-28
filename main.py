# Load the data from the titanic competition
import pandas as pd

titanic_data = pd.read_csv("./train.csv")

# Show the first five rows of the data
titanic_data.head()

# Number of total passengers
total = len(titanic_data)
print(total)

# Number of passengers 
# who survived
survived = (titanic_data.Survived == 1).sum()
print(survived)

# Number of passengers under 18
minors = (titanic_data.Age < 18).sum()
print(minors)