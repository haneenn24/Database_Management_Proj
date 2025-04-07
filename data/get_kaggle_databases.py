import pandas as pd
import kagglehub

# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv
# Load House Pricing dataset
house_pricing_df = pd.read_csv('data/house_pricing.csv', sep=';')

# https://www.kaggle.com/datasets/mathchi/diabetes-data-set?select=diabetes.csv
# Load Diabetes dataset
path = kagglehub.dataset_download("mathchi/diabetes-data-set")

print("Path to dataset files:", path)
diabetes_df = pd.read_csv('data/diabetes-data-set.csv', sep=';')

# https://archive.ics.uci.edu/dataset/360/air+quality
# Load Air Quality dataset
air_quality_df = pd.read_csv('data/AirQualityUCI.csv', sep=';')


# https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
# Load Lending Club dataset
lending_club_df = pd.read_csv('data/lending_club_loan_data.csv')
