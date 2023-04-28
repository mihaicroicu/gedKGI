import pandas as pd

def fetch_ids(level):
    if level == 'priogrid':
        df = pd.read_csv('priogrid.csv')
        return list(df.priogrid),''
    if level == 'month':
        return list(range(1,699)),''
    