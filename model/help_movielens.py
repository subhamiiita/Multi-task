import pandas as pd
def fun():
    xyz=pd.read_csv('../objs/test.csv')
    return xyz['userId'].unique()
