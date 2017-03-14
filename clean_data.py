import pandas as pd
import numpy as np


def clean_train(file_name):
    df = pd.read_csv(file_name, low_memory=False)

    #split y
    y = df['SalePrice']

    #create dummy variables for categories
    categories = pd.get_dummies(df['fiProductClassDesc'])
    cats = categories[categories.columns[:-1]]
    df[cats.columns] = cats

    #fix year data
    fake_yr = df['YearMade'][df['YearMade'] != 1000].mean()
    fake_yr = np.around(fake_yr)
    df.loc[df['YearMade'] == 1000,'YearMade'] = fake_yr

    #extract sales year and month
    df['salesyear'] = df['saledate'].str.extract('(\d{4})',
                            expand=False).astype(int)
    df['salesmonth'] = df['saledate'].str.extract('(\A\d{1,2})',
                            expand=False).astype(int)
    df['salesmonth'] -= 1
    counted = df.groupby('salesmonth').count().sort_values('YearMade')
    counted.reset_index(inplace=True)
    new_mos = counted['salesmonth'].values
    df['salesmonth'] = new_mos[df['salesmonth']]
    #try with boolean
    df['good_month'] = df['salesmonth'].isin([12,9,6,2,3]).astype(int)

    df['better'] = (df['fiModelDesc'].str.len() > df['fiBaseModel'].str.len()).astype(int)

    #clean up enclosure data
    mode_enclosure = df['Enclosure'].mode().values[0]
    df['Enclosure'].fillna(mode_enclosure, inplace = True)
    df.loc[df['Enclosure'] == 'None or Unspecified',
        'Enclosure'] = mode_enclosure
    enclosures = pd.get_dummies(df['Enclosure'])
    enclosures['EROPS AC'] +=  enclosures['EROPS w AC']
    enclosures.drop('EROPS w AC', inplace=True, axis=1)
    df[enclosures.columns[:-1]] = enclosures[enclosures.columns[:-1]]

    #auctioneer id
    mode_auc = df['auctioneerID'].mode().values[0]
    df['auctioneerID'].fillna(mode_auc, inplace = True)
    auc = pd.get_dummies(df['auctioneerID'])
    df[auc.columns[:-1]] = auc[auc.columns[:-1]]

    #features to use
    feat_cols = (cats.columns).tolist() + (enclosures.columns[:-1]).tolist() + (auc.columns[:-1]).tolist()
    feat_cols.extend(['YearMade','salesmonth'])
    X = df[feat_cols]

    return X, y, feat_cols, (categories.columns).tolist(), \
        mode_enclosure, (enclosures.columns).tolist(), \
        new_mos,(auc.columns).tolist()

def clean_test(file_name, feat_cols, dummies,
                    mode_enclosure, encl_dummies,months_map, auc_id):
    df = pd.read_csv(file_name, low_memory=False)

    #create dummy variables for categories
    categories = pd.get_dummies(df['fiProductClassDesc'])
    df[categories.columns] = categories

    for dum in dummies:
        if dum not in categories.columns:
            df[dum] = np.zeros(df.shape[0])
    df.drop(dummies[-1], inplace = True, axis=1)

    #enclosure feature
    #clean data
    df['Enclosure'].fillna(mode_enclosure, inplace = True)
    df.loc[df['Enclosure'] == 'None or Unspecified',
        'Enclosure'] = mode_enclosure
    #get dummies
    enclosures = pd.get_dummies(df['Enclosure'])
    df[enclosures.columns] = enclosures
    #add missing from train
    for ecl_dum in encl_dummies:
        if ecl_dum not in enclosures.columns:
            df[ecl_dum] = np.zeros(df.shape[0])
    #merge AC
    enclosures['EROPS AC'] +=  enclosures['EROPS w AC']
    enclosures.drop('EROPS w AC', inplace=True, axis=1)
    #drop last one
    df.drop(encl_dummies[-1], inplace = True, axis=1)

    #get dummies
    auc = pd.get_dummies(df['auctioneerID'])
    df[auc.columns] = auc

    #auctioneer id
    for auc_dum in auc_id:
        if auc_dum not in auc.columns:
            df[auc_dum] = np.zeros(df.shape[0])

    df.drop(auc_id[-1], inplace = True, axis=1)

    #fix year data
    fake_yr = df['YearMade'][df['YearMade'] != 1000].mean()
    fake_yr = np.around(fake_yr)
    df.loc[df['YearMade'] == 1000,'YearMade'] = fake_yr

    #compare whether model better than base model
    df['better'] = (df['fiModelDesc'].str.len() > df['fiBaseModel'].str.len()).astype(int)

    #extract sales year and month
    df['salesyear'] = df['saledate'].str.extract('(\d{4})',
                            expand=False).astype(int)
    df['salesmonth'] = df['saledate'].str.extract('(\A\d{1,2})',
                            expand=False).astype(int)
    df['salesmonth'] -= 1
    df['salesmonth'] = months_map[df['salesmonth']]
    df['good_month'] = df['salesmonth'].isin([12,9,6,2,3]).astype(int)

    #features to use
    salesids = df['SalesID']
    feat_cols = dummies[:-1] + encl_dummies[:-1] + auc_id[:-1]
    feat_cols.extend(['YearMade','salesmonth'])
    X = df[feat_cols]

    return X, salesids
