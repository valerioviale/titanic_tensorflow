# Load the data from the titanic competition
import pandas as pd
titanic_data = pd.read_csv("../input/titanic/train.csv")

# Show the first five rows of the data
titanic_data.head()