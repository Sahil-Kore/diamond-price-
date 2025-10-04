import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../inputs/train.csv" , index_col="id")
test_df = pd.read_csv("../inputs/test.csv" , index_col="id")

y = df["price"]
cat_cols = df.select_dtypes('object').columns
num_cols = [c for c in df.select_dtypes('number').columns if c not in ["price"]]

num_cols
cat_cols 


for col in num_cols:
    sns.histplot(data= df , x = col)
    plt.title(col)
    plt.show()
    
    
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    combined_data = pd.concat([df[col] , test_df[col]]).astype(str)
    le.fit(combined_data)
    df[col]  = le.transform(df[col])
    
    test_df[col] = le.transform(test_df[col])
    

df.to_csv("../inputs/train_lb.csv")
test_df.to_csv("../inputs/test_lb.csv")


df = pd.read_csv("../inputs/train.csv" , index_col="id")
test_df = pd.read_csv("../inputs/test.csv" , index_col="id")

cat_cols = df.select_dtypes('object').columns
num_cols = [c for c in df.select_dtypes('number').columns if c not in ["price"]]


for col in num_cols:
    sns.scatterplot(data= df , x= col , y = y)
    plt.title(f"{col}  vs price")
    plt.show()
    
for col in cat_cols:
    sns.violinplot(df , x= col , y = y)
    plt.show()
    

from sklearn.preprocessing import OneHotEncoder
combined_data = pd.concat([df[cat_cols], test_df[cat_cols]])

# 2. Fit the encoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(combined_data)

# 3. Transform the data and create new DataFrames with proper column names
train_ohe = pd.DataFrame(ohe.transform(df[cat_cols]), columns=ohe.get_feature_names_out())
test_ohe = pd.DataFrame(ohe.transform(test_df[cat_cols]), columns=ohe.get_feature_names_out())

# 4. Align indexes to prevent joining issues
train_ohe.index = df.index
test_ohe.index = test_df.index

# 5. Drop the original categorical columns from the original DataFrames
df = df.drop(columns=cat_cols)
test_df = test_df.drop(columns=cat_cols)

# 6. Concatenate the new one-hot encoded columns
df = pd.concat([df, train_ohe], axis=1)
test_df = pd.concat([test_df, test_ohe], axis=1)

df.to_csv("../inputs/train_ohe.csv")
test_df.to_csv("../inputs/test_ohe.csv")