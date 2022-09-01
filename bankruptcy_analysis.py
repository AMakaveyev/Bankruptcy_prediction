# Import packages
import pandas as pd
import numpy as np
from scipy.io import arff

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Specification of columnnames:
Attr_Dict = {
"Attr1" :	"net profit / total assets",
"Attr2" :	"total liabilities / total assets" ,
"Attr3" :	"working capital / total assets" ,
"Attr4" :	"current assets / short-term liabilities ",
"Attr5" :	"((cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)) * 365" ,
"Attr6" :	"retained earnings / total assets" ,
"Attr7" :	"EBIT / total assets" ,
"Attr8" :	"book value of equity / total liabilities" ,
"Attr9" :	"sales / total assets" ,
"Attr10" :	"equity / total assets" ,
"Attr11" :	"(gross profit + extraordinary items + financial expenses) / total assets",
"Attr12" :	"gross profit / short-term liabilities" ,
"Attr13" :	"(gross profit + depreciation) / sales" ,
"Attr14" :	"(gross profit + interest) / total assets" ,
"Attr15" :	"(total liabilities * 365) / (gross profit + depreciation)" ,
"Attr16" :	"(gross profit + depreciation) / total liabilities" ,
"Attr17" :	"total assets / total liabilities" ,
"Attr18" :	"gross profit / total assets",
"Attr19" :	"gross profit / sales" ,
"Attr20" :	"(inventory * 365) / sales" ,
"Attr21" :	"sales (n) / sales (n-1)" ,
"Attr22" :	"profit on operating activities / total assets" ,
"Attr23" :	"net profit / sales" ,
"Attr24" :	"gross profit (in 3 years) / total assets" ,
"Attr25" :	"(equity - share capital) / total assets" ,
"Attr26" :	"(net profit + depreciation) / total liabilities" ,
"Attr27" :	"profit on operating activities / financial expenses" ,
"Attr28" :	"working capital / fixed assets",
"Attr29" :	"logarithm of total assets" ,
"Attr30" :	"(total liabilities - cash) / sales" ,
"Attr31" :	"(gross profit + interest) / sales" ,
"Attr32" :	"(current liabilities * 365) / cost of products sold" ,
"Attr33" :	"operating expenses / short-term liabilities" ,
"Attr34" :	"operating expenses / total liabilities" ,
"Attr35" :	"profit on sales / total assets" ,
"Attr36" :	"total sales / total assets" ,
"Attr37" :	"(current assets - inventories) / long-term liabilities" ,
"Attr38" :	"constant capital / total assets" ,
"Attr39" :	"profit on sales / sales" ,
"Attr40" :	"(current assets - inventory - receivables) / short-term liabilities" ,
"Attr41" :	"total liabilities / ((profit on operating activities + depreciation) * (12/365))" ,
"Attr42" :	"profit on operating activities / sales" ,
"Attr43" :	"rotation receivables + inventory turnover in days" ,
"Attr44" :	"(receivables * 365) / sales" ,
"Attr45" :	"net profit / inventory" ,
"Attr46" :	"(current assets - inventory) / short-term liabilities",
"Attr47" :	"(inventory * 365) / cost of products sold" ,
"Attr48" :	"EBITDA (profit on operating activities - depreciation) / total assets" ,
"Attr49" :	"EBITDA (profit on operating activities - depreciation) / sales" ,
"Attr50" :	"current assets / total liabilities" ,
"Attr51" :	"short-term liabilities / total assets" ,
"Attr52" :	"(short-term liabilities * 365) / cost of products sold)" ,
"Attr53" :	"equity / fixed assets" ,
"Attr54" :	"constant capital / fixed assets" ,
"Attr55" :	"working capital",
"Attr56" :	"(sales - cost of products sold) / sales" ,
"Attr57" :	"(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)" ,
"Attr58" :	"total costs /total sales" ,
"Attr59" :	"long-term liabilities / equity" ,
"Attr60" :	"sales / inventory" ,
"Attr61" :	"sales / receivables" ,
"Attr62" :	"(short-term liabilities *365) / sales" ,
"Attr63" :	"sales / short-term liabilities" ,
"Attr64" :	"sales / fixed assets"}

# Functions

# Function to rename the attributes columns to their actual names.
def rename_attributes(df):
    """ Rename columns of a dataframe according to specification in the dictionary: Attr_Dict"""
    for i in range (1,65):
        col_name = "Attr" + str(i)
        df.rename(columns={col_name: Attr_Dict.get(col_name)}, inplace= True)


# Convert bancruptcy column to binary numeric format (0/1)
def make_class_binary(df):
    """make class numerical and rename the column """
    df["class"] = pd.to_numeric(df["class"])
    df.rename(columns={"class": "Bancruptcy Dummy"}, inplace= True)

# Import data sets
# Year 1
dataset = arff.loadarff(open(r'/Users/anatoliymakaveyev/Documents/Python Environment/BANKRUPTCY_PREDICTION_CASE/data/1year.arff'))
data_year_1 = pd.DataFrame(dataset[0])

# Year 2
dataset = arff.loadarff(open(r'/Users/anatoliymakaveyev/Documents/Python Environment/BANKRUPTCY_PREDICTION_CASE/data/2year.arff'))
data_year_2 = pd.DataFrame(dataset[0])

# Year 3
dataset = arff.loadarff(open(r'/Users/anatoliymakaveyev/Documents/Python Environment/BANKRUPTCY_PREDICTION_CASE/data/3year.arff'))
data_year_3 = pd.DataFrame(dataset[0])

# Year 4
dataset = arff.loadarff(open(r'/Users/anatoliymakaveyev/Documents/Python Environment/BANKRUPTCY_PREDICTION_CASE/data/4year.arff'))
data_year_4 = pd.DataFrame(dataset[0])

# Year 5
dataset = arff.loadarff(open(r'/Users/anatoliymakaveyev/Documents/Python Environment/BANKRUPTCY_PREDICTION_CASE/data/5year.arff'))
data_year_5 = pd.DataFrame(dataset[0])

# Create list of all dataframes
df_list= [data_year_1,data_year_2, data_year_3, data_year_4, data_year_5]

# Apply functions to all dataframes to rename the columns and convert bancruptcy variable to numeric format
[rename_attributes(df) for df in df_list]
[make_class_binary(df) for df in df_list]

data_year_1.describe().T

data_year_1["Bancruptcy Dummy"].value_counts()
data_year_2["Bancruptcy Dummy"].value_counts()
data_year_3["Bancruptcy Dummy"].value_counts()

data_year_4["Bancruptcy Dummy"].value_counts()
data_year_5["Bancruptcy Dummy"].value_counts()


data_year_1.info()
data_year_2.info()
data_year_3.info()
data_year_4.info()
data_year_5.info()



# Modeling

# Split data into input and target variable

# Independent Variables
X = data_year_1.loc[:, data_year_1.columns != "Bancruptcy Dummy"]
# Dependet Variable
Y = data_year_1.loc[:, data_year_1.columns == "Bancruptcy Dummy"]


# Divide data into train and test data set
random_seed = 117
# Specification of the size of the test set
test_size = 0.33

# Split data
# stratify = Y means that the proportions of the dependent variable are kept constant after the split.
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size=test_size, random_state=random_seed)



y_train.value_counts()
y_test.value_counts()

model = XGBClassifier()
model.fit(X_train, y_train)