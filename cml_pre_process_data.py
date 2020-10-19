import numpy as np
import pandas as pd
from cml_functions import process_author_name,process_dates_birth_death,process_technique,process_dimensions,process_colours

#Import Dataset
df_cml = pd.read_csv('CML\\artworks.csv',encoding='utf-16',index_col=0)

#Transport the author to the Author column
filt = (df_cml['Title'].str.contains('\(\d')) & (df_cml['Author'].isna()) & (~df_cml['Title'].str.lower().str.contains('álbum')) & (~df_cml['Title'].str.lower().str.contains('caricatura'))
df_cml.loc[filt,'Author'] = df_cml[filt]['Title'].str.extract(r'(.*?\))')[0].str.replace('(','- ').str.replace(')','')
#Removet the Author from the Title column
df_cml.loc[filt,'Title'] = df_cml[filt]['Title'].str.extract(r'\)(.*)')[0].str.replace('- ','')

#Fill missing values from the Author field with "no information"
df_cml['Author'] = df_cml['Author'].fillna('No information')

#Remove the entries without Technique information
df_cml = df_cml[df_cml['Technique'].isna() == False].copy()

#---------------------- Process Features ---------------------------
#Call function to process Author Name
df_cml = process_author_name(df_cml,'Author')

#Call function to process Birth and Death Dates
df_cml = process_dates_birth_death(df_cml,'Author')

#Call function to process Technique - Using Standford's Stanza
df_cml = process_technique(df_cml,'Technique')

#Call function to process Dimensions
df_cml = process_dimensions(df_cml,'Dimensions')

#Call function to extract and process colours from image
df_cml = process_colours(df_cml,resize_thumbnails=False)

#Process Data features
#Convert te Date of Auction End to datetime format
df_cml['Date of Auction End'] = pd.to_datetime(df_cml['Date of Auction End'])

#Create new columns with Year, Month, Week, Day, Day of Week, Hour
df_cml['Year'] = df_cml['Date of Auction End'].dt.year
df_cml['Month'] = df_cml['Date of Auction End'].dt.month
df_cml['Day'] = df_cml['Date of Auction End'].dt.day
df_cml['Day of Week'] = df_cml['Date of Auction End'].dt.day_name()
df_cml['Hour'] = df_cml['Date of Auction End'].dt.hour

#Create the Sold feature
df_cml['Sold'] = (df_cml['Sale Price'] != 'Not sold').astype('uint8')

#Create the Final Price feature
df_cml['Final Price'] = df_cml['Sale Price'].str.replace(',','').apply(lambda x: 0 if x=='Not sold' else x)

#Let's save this into a new DataFrame
df_cml.to_csv('.\cleaned_data\cleaned_df_cml.csv')