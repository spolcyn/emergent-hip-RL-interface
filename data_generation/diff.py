import pandas as pd
pd.options.display.width = 200

df1 = pd.read_csv('data.csv', header=0)
df2 = pd.read_csv('data.csv.old', header=0)

cols_common  = (df1.columns & df2.columns).tolist()
cols_added   = (df2.columns - df1.columns).tolist()
cols_deleted = (df1.columns - df2.columns).tolist()

print("\nAdded",   df2.ix[:, cols_added])
print("\nDeleted", df1.ix[:, cols_deleted])
print("\nChanged", df2.ix[:, cols_common])
