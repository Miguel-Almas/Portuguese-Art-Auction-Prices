"""
This file contains functions necessary for Data Treatment, Preprocessing and Hyperparameter Tuning of CML Art Bids
"""
import numpy as np
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error

def process_author_name(df,col_name):
    """
    This small function receives a the Dataframe and the column name containing the Author Name, cleans and processes it.
    The name column may in fact contain several artists names and this function will identify how many and break them by several columns. 
    It also creates a column with the number of authors.
    
    Inputs: df - Pandas DataFrame
            col_name - String with the name of the Author Name column
            
    Outputs: df - Procesed Pandas DataFrame
    """
    #Maintain copy of original column
    df[col_name + '_tmp'] = df[col_name]
    
    #Clean the Author Name
    df_author_name_to_clean = df[col_name + '_tmp'].str.lower().str.replace('(nasc. [0-9]+)|(séc. xix/xx)|(séc. xx/xxi)|(séc. xxi)|(séc. xx)|(séc. xix)|(nasc. xx)|(nasc. xix)|(nasc. xxi)|(nasc. xxi)|([0-9]+)','')
    df[col_name + '_tmp'] = df_author_name_to_clean.str.replace('(- )+$|(-)+|( - )+|( -)+$','').str.rstrip().str.rstrip(',')
    df[col_name + '_tmp'] = df[col_name + '_tmp'].str.replace(' ()','',regex=False)

    #Delete the temporary dataframe
    del df_author_name_to_clean
    
    #Get number of Authors on the Author Name field
    df['Number of Authors'] = df[col_name + '_tmp'].str.split('/|,').apply(lambda x: len(x))

    #Create columns with the name of authors (1 author, 2 author, etc.)
    for i in range(df['Number of Authors'].max()):
        df[str(i+1)+' '+col_name] = 'none'
        df.loc[df['Number of Authors'] >= i+1,str(i+1)+' '+col_name] = (df[df['Number of Authors'] >= i+1][col_name + '_tmp'].str.split('/|,')).apply(lambda x: x[i]).str.lstrip()

    #Drop the "Author Name Column"
    df = df.drop(col_name + '_tmp',axis=1)

    return df

def choose_date(x):
    """
    This function was created to be used with pandas apply. It receives a list and will extract the dates from it and infer which are date of birth and which are dates of death.
    
    Returns the list of dates list_dates.
    """
    len_list = len(x)
    list_dates = []
    counter = 1
        
    for i in range(len_list):
        if i == 0:
            list_dates.append(int(re.findall(r'\d+',x[i])[0]))
            counter += 1
        else:
            if 'nasc' in x[i] and counter % 2 != 0:
                list_dates.append(int(re.findall(r'\d+',x[i])[0]))
                counter += 1   
            elif 'nasc' in x[i] and counter % 2 == 0:
                list_dates.append('no information')
                counter += 1
                list_dates.append(int(re.findall(r'\d+',x[i])[0]))
                counter += 1 
            elif int(re.findall(r'\d+',x[i])[0]) < list_dates[counter-2] + 40 and counter % 2 == 0:
                list_dates.append('no information')
                counter += 1
                list_dates.append(int(re.findall(r'\d+',x[i])[0]))
                counter += 1
            else:
                list_dates.append(int(re.findall(r'\d+',x[i])[0]))
                counter += 1
    if (counter-1) % 2 != 0:
        list_dates.append('no information')
    return list_dates

def process_dates_birth_death(df,col):
    """
    This function extracts dates of birth and death from a supplied column of a Dataframe.
    
    Inputs: df - Pandas DataFrame
            col - String with column name
            
    Outputs: df - Pandas DataFrame
    """
    #Extract dates from various lines
    df_dates = df[col].str.lower().str.extractall('(nasc. [0-9]+)|(séc. xxi)|(séc. xx)|(séc. xix)|(nasc. xx)|(nasc. xix)|(nasc. xx/xxi)|(nasc. xxi)|(nasc. xxi)|([0-9]+)|(no information)')

    #Merge all columns into a single one
    df_dates = pd.DataFrame(df_dates.apply(lambda x: ','.join(x.dropna().astype(str)),axis=1))
    #Obtain index of rows that must be joined (cases where 2 or more dates exist for the same artwork)
    index_nbr = df_dates[df_dates.index.get_level_values(1) == 1].index.get_level_values(0).tolist()

    #For those indexes, join all dates in single row
    for i in index_nbr:
        df_dates.loc[i,0] = df_dates.loc[i].apply(lambda x: ','.join(x.dropna().astype(str)),axis=0).values[0]
    #Drop the the second level rows
    df_dates = df_dates.xs(0, level=1)

    #Import to new column of Dates
    df['Dates'] = df_dates[0]

    #Drop temporary dataframe
    del df_dates

    #Fill any missing values with "no information"
    df.loc[df['Dates'].isna(),'Dates'] = 'no information'

    #--------------------------------------------- Fill in Birth and Death Dates --------------------------------------------------
    #--------------------------------------------- Rows with only 1 Author --------------------------------------------------------
    filt = df['Number of Authors'] == 1
    #Assign the author Date of Birth
    df.loc[filt,'1 Author Birth'] = df[filt]['Dates'].str.split(',').apply(lambda x: x[0])
    #Assign the author Date of Death
    df.loc[filt,'1 Author Death'] = df[filt]['Dates'].str.split(',').apply(lambda x: x[1] if len(x)==2 else 'no information')

    #--------------------------------------------- Rows with more than 1 Author --------------------------------------------------------
    for i in range(2,df['Number of Authors'].max()+1):
        filt = df['Number of Authors'] == i
        counter = 0
        for j in range(i):
            df.loc[filt,str(j+1)+' Author Birth'] = df[filt]['Dates'].str.split(',').apply(choose_date).apply(lambda x: str(x[counter])) 
            df.loc[filt,str(j+1)+' Author Death'] = df[filt]['Dates'].str.split(',').apply(choose_date).apply(lambda x: str(x[counter+1])) 
            counter +=2

    #Deal with missing values
    df[[i for i in df.columns.tolist() if ('Death' in i) or ('Birth' in i)]] = df[[i for i in df.columns.tolist() if ('Death' in i) or ('Birth' in i)]].fillna('no information')

    #Ensure al are str
    df[[i for i in df.columns.tolist() if ('Death' in i) or ('Birth' in i)]] = df[[i for i in df.columns.tolist() if ('Death' in i) or ('Birth' in i)]].astype('str')

    #Clean any possible artifacts from name
    for i in [i for i in df.columns.tolist() if ('Death' in i) or ('Birth' in i)]:
        df[i] = df[i].str.replace('nasc. ','')
        #Correct possible date mistakes
        df.loc[df[i] == '19',i] = 'séc. xx'
        df.loc[df[i] == '18',i] = 'séc. xix'
        for j in range(10):
            df.loc[df[i] == '19'+str(j),i] = '19'+str(j)+'0'
            
    df = df.drop([col,'Dates'],axis=1)
    
    return df

def process_technique(df,col,path_stopwords_file='Support/stopwords-pt.txt'):
    """
    This function will perform some processing to the text about the techniques used on the art to allow for future feature extraction.
    
    Inputs: df - DataFrame with data
            col - Name of the column with the technique information
            
    Outputs: df - DataFrame with data after processing
    """
    import stanza
    stanza.download('pt')
    nlp = stanza.Pipeline(lang='pt',processors='tokenize,lemma')

    #Let's obtain a list of stopwords (from https://github.com/stopwords-iso/stopwords-pt), without punctuation.
    list_stopwords = pd.read_csv(path_stopwords_file,header=None)[0].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').tolist()

    #Let's add some words to the list of stopwords.
    list_stopwords.append('paragrafo')
    list_stopwords.append('sec')
    list_stopwords.append('xxi')
    list_stopwords.append('xx')
    list_stopwords.append('xix')
    list_stopwords.append('xx/xxi')
    list_stopwords.append('xix/xx')
    list_stopwords.append('-')
    list_stopwords.append('')
    list_stopwords.append('xxxxi')
    list_stopwords.append('xixxx')

    #Removing punctuation
    df[col] = df[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower()

    #Remove certain patterns
    df['Technique Preprocessed'] = df[col].str.replace(',','').str.replace('(','').str.replace(')','').str.replace('\d+','').str.replace(r'/','').str.replace(r'-','').str.replace('.','').str.replace('?','')

    #Tokenize
    df['Technique Tokenized'] = df['Technique Preprocessed'].str.split(' ')

    #Remove stopwords
    df['Technique Tokenized'] = df['Technique Tokenized'].apply(lambda x: [word for word in x if word not in list_stopwords])

    #Lemmatization
    #Store the technique as a list of phrases
    list_phrases = df['Technique Tokenized'].apply(lambda x: ' '.join(x)).tolist()
    
    def get_lemma(phrase):
        "Get lemmas"
        lemma = ""
        for sent in nlp(phrase).sentences:
            for word in sent.words:
                lemma += word.lemma + "\t"
                
        return '+'.join(lemma.split())
    
    tmp = df['Technique Tokenized'].apply(lambda x: ' '.join(x))
    tmp = tmp[tmp != '']
    #Store lemmas on the dataframe
    df['Technique Lemmatized'] = tmp.apply(get_lemma)
    #Drop missing values
    df = df.dropna(subset=['Technique Lemmatized'])
    #Specific replacement - words that were not lemmatized
    df['Technique Lemmatized'] = df['Technique Lemmatized'].str.replace('assinada','assinar')
    df['Technique Lemmatized'] = df['Technique Lemmatized'].str.replace('datada','datar')
    df['Technique Lemmatized'] = df['Technique Lemmatized'].str.replace('datado','datar')

    df = df.drop('Technique Preprocessed',axis=1)
    
    #Convert list of strings to string to allow storing as csv
    df['Technique Tokenized'] = df['Technique Tokenized'].str.join('+')

    return df    

def process_dimensions(df,col='Dimensions'):
    """
    This function extracts 3 columns of dimensions from a supplied column of a Dataframe.
    
    Inputs: df - Pandas DataFrame
            col - String with column name
            
    Outputs: df - Pandas DataFrame
    """
    #Initialize the new columns
    df['Dim 1'] = np.nan
    df['Dim 2'] = np.nan
    df['Dim 3'] = np.nan

    #Change cases where dimensions column does not contain dimensions to "no informaton"
    df.loc[df[col].str.contains('Dim')==False,col] = 'no information'

    #Replace missing values
    df[col] = df[col].fillna('no information')

    #Replace , with . from dimensions (in portuguese decimals use , but internationally it's .)
    df[col] = df[col].str.replace(', ','.').str.replace(',','.')

    #Extract list with dimensions
    df['Dimension List'] = df[col].str.findall(r'(\d+\,+\d*|\d+\.+\d*|\d+|no information)')
    #Extract number of dimensions
    df['Number of Dimensions'] = df['Dimension List'].apply(lambda x: len(x))
    #Extract unit of dimensions
    df['Dimension Units'] = df[col].str.findall(r'(no information)$|(cm)$|(mm)$').apply(lambda x: x[0][0] if x[0][0] != '' else (x[0][1] if x[0][1] != '' else x[0][2]))

    #We need to deal with cases where there are more than 3 dimensions (these can be cases where the painted and paper area are both specified or cases where multiple objects or artworks are sold together)
    #-------------------------- For cases with 1 dimension --------------------------------------
    #Replace number of dimensions when 'no information' from 1 to 0
    df.loc[(df['Number of Dimensions'] == 1) & (df[col].str.lower() == 'no information'),'Number of Dimensions'] = 0

    #Identify number of artworks in a lot
    df['Number of Artworks'] = 0

    list_words_nbr = ['^dois |^duas |^par |^2 ',
                      '^tres |^três |^3 ',
                      '^quatro |^4 ',
                      '^cinco |^5 ',
                      '^seis |^6 ',
                      '^sete |^7 ',
                      '^oito |^8 ',
                      '^nove |^9 ',
                      '^dez |^10 ','^onze |^11 ','^doze |^12 ','^treze |^13 ','^catorze |^14 ','^quinze |^15 ','^dezasseis |^16 ','^dezassete |^17 ','^dezoito |^18 ','^dezanove |^19 ',
                     '^vinte |^20 ','^vinte e um |vinte um |^21 ','^vinte e dois |vinte dois |^22 ','^vinte e tres |vinte e três |vinte tres |vinte três |^23 ','^vinte e quatro |vinte quatro |^24 ',
                     '^vinte e cinco |vinte cinco |^25 ','^vinte e seis |vinte seis |^26 ','^vinte e sete |vinte sete |^27 ','^vinte e oito |vinte oito |^28 ','^vinte e nove |vinte nove |^29 ',
                     '^trinta |^30 ','^quarenta |^40 ','^cinquenta |^50 ','^sexta |^60 ','^setenta |^70 ','^oitenta |^80 ','^noventa |^90 ','^cem |^100 ']
    counter = 2
    for i in list_words_nbr:
        df.loc[(df['Title'].str.lower().str.contains(i)) | (df['Technique'].str.lower().str.contains(i)),'Number of Artworks'] = counter
        counter += 1

    #Correct the number of artworks for cases where multiple artworks cannot be identified via title or technique, but with dimensions information.
    df.loc[(df[col] != 'no information') & (df['Number of Artworks'] == 0),'Number of Artworks'] = 1

    #------------------------------------------- Cases with 1 artwork -------------------------------------------------------------------------
    #Not many cases exist of single artworks with more than 3 dimensions. We'll have to decide a criteria for these cases.
    #It seems that most of these cases falls into 1 of 2 categories: 
    #1 - They are paintings that are detailing the size of the painted area as well as the total "paper" area.
    #2 - They are a set of objects with the dimensions specified for them.
    #In all cases, a ; or / divides them and we can use this to our advantage. Also for paintings (we'll look for serigrafia, tela, oleo etc.), we will consider the size of the smallest dimensions as those are the dimensions of interest. For 2, we'll consider only the first 3 dimensions.

    # For 1:
    filt = (df['Technique Lemmatized'].apply(lambda x: any(item for item in ['serigrafia','oleo','tela','papel'] if item in x)))& (df['Number of Artworks'] == 1) & (df['Number of Dimensions'] > 3)
    df.loc[filt,'Dimension List'] = df[filt]['Dimension List'].apply(lambda x: x[:2] if float(x[0])*float(x[1]) < float(x[2])*float(x[3]) else x[2:])

    # For 2:
    filt = (df['Number of Artworks'] == 1) & (df['Number of Dimensions'] > 3) & (df['Technique Lemmatized'].apply(lambda x: any(item for item in ['serigrafia','oleo','tela','papel'] if item in x)) == False)
    df.loc[filt,'Dimension List'] = df[filt]['Dimension List'].apply(lambda x: x[:3])

    #--------------------------------------------- Multiple artworks ---------------------------------------------------------------------------
    #So for multiple artworks we have 2 main cases... When we have more dimensions than artworks and vice versa.
    #More dimensions than artworks:
    #    Even number of dimensions (> 3) - We can obtain N dimensions, N being Number of Dimensions / Number of Artworks.
    #    Uneven number of dimensions (> 3) - Only one instance of this exists with more than 3 dimensions. In this case, we will take the first 3 dimensions on the list.    
    #    Dimensions between 1 and 3 - Store all dimensions in Dim 1 to 3.  
    #More artworks than dimensions:
    #    No cases exist with more artwoks than dimensions with more than 3 dimensions. Therefore, in this situation all dimensions will be stored in Dim 1 to 3.    

    #Fill Dim 1 to 3 for 1 Artwork
    filt = df['Number of Artworks'] == 1
    for i in range(1,4):
        #Fill in Dim 1 to 3
        df.loc[filt,'Dim '+str(i)] = df[filt]['Dimension List'].apply(lambda x: x[i-1] if len(x) >= i else np.nan)

    #Fill Dim 1 to 3 for 2+ Artworks with 3 or less dimensions
    filt = (df['Number of Artworks'] > 1) &  (df['Number of Dimensions'] <= 3)
    for i in range(1,4):
        #Fill in Dim 1 to 3
        df.loc[filt,'Dim '+str(i)] = df[filt]['Dimension List'].apply(lambda x: x[i-1] if len(x) >= i else np.nan)

    #Fill Dim 1 to 3 for 2+ Artworks with more than 3 dimensions - Uneven dimensions
    filt = (df['Number of Artworks'] > 1) & (df['Number of Dimensions'] > 3) & ((df['Number of Dimensions'] / df['Number of Artworks']) % 1 > 0)
    for i in range(1,4):
        #Fill in Dim 1 to 3
        df.loc[filt,'Dim '+str(i)] = df[filt]['Dimension List'].apply(lambda x: x[i-1] if len(x) >= i else np.nan)

    #Even Number of dimensions - for this to work we need to select only a subset of the dimensions, so we'll choose the first set of dimensions.
    filt = (df['Number of Artworks'] > 1) & (df['Number of Dimensions'] > 3) & ((df['Number of Dimensions'] / df['Number of Artworks']) % 1 == 0)
    list_nbr_dimensions = (df.loc[filt,'Number of Dimensions'] / df.loc[filt,'Number of Artworks']).values.tolist()
    idx = (df.loc[filt,'Number of Dimensions'] / df.loc[filt,'Number of Artworks']).index

    for i in range(len(idx)):
        for j in range(int(list_nbr_dimensions[i])):
            df.loc[idx[i],'Dim '+ str(j+1)] = df.loc[idx[i],'Dimension List'][j]
            
    df['Dim 1'] = df['Dim 1'].fillna('')
    
    return df

def subplot_hyperparameter(df_results,best_params):
    """
    This function receives a DataFrame with results from Hyperparameter tuning and creates a graph for the evolution of MEAN RMSE of training and validation for each specific hyperparameter.
    Inputs - df_results: A DataFrame with the results from the Hyperparameter tuning
             best_params: A dictionary with the hyperparameter as key and best foun value as value.
    """
    #If only 1 parameter has been supplied to grid search, then plot it directly else use subplots.
    if len(df_results['params'][0].keys()) == 1:
        #Sort dataframe by the parameter in question
        parameter = list(df_results['params'][0].keys())[0]
        df_results = df_results.sort_values('param_'+parameter,ascending=True)
        #Get mean train and test scores
        mean_train = -df_results['mean_train_score']
        mean_test = -df_results['mean_test_score']
        x_feature = df_results['param_'+parameter]
        #Plot
        fig = go.Figure(data=[
            go.Scatter(name='Mean Training RMSE', x=x_feature, y=mean_train),
            go.Scatter(name='Mean Validation RMSE', x=x_feature, y=mean_test),
            go.Scatter(name='Best performing '+parameter, x=[best_params[parameter]],
                        y=-df_results[df_results['param_'+parameter] == best_params[parameter]]['mean_test_score'])
        ])
        # Change the bar mode
        #fig.update_layout(barmode='stack',title='Sold and unsold artworks of top 70 artists by auctioned pieces')
        fig.update_xaxes(title_text=parameter)
        fig.update_yaxes(title_text="RMSE")
        fig.update_layout(hovermode="x unified")
        fig.show()
    else:
        row_nbr=1
        counter=1
        #Calculate number of rows
        n_rows = int(np.ceil(len(df_results['params'][0].keys())/2))
        #Start subplots figure
        fig = make_subplots(rows=n_rows, cols=2)    
        for parameter in df_results['params'][0].keys():
            #Get subset of DataFrame with all other parameters with equal value
            tmp = df_results.sort_values(['param_'+ parameter]+[i for i in df_results.columns if (i != 'param_'+ parameter) & ('param_' in i)]).drop_duplicates('param_'+parameter)
            mean_train = -tmp['mean_train_score']
            mean_test = -tmp['mean_test_score']
            x_feature = tmp['param_'+parameter]
            if counter % 2 == 0:
                list_params_hover = ['param_'+ parameter]+[i for i in df_results.columns if (i != 'param_'+ parameter) & ('param_' in i)]
                fig.add_trace(
                go.Scatter(name='Mean Training RMSE', x=x_feature, y=mean_train),
                row=row_nbr, col=2
                )    
                fig.add_trace(
                go.Scatter(name='Mean Validation RMSE', x=x_feature, y=mean_test),
                row=row_nbr, col=2
                )    
                fig.add_trace(
                go.Scatter(name='Best performing '+parameter, x=[best_params[parameter]],
                            y=-tmp[tmp['param_'+parameter] == best_params[parameter]]['mean_test_score']),
                row=row_nbr, col=2
                )    
                fig.update_xaxes(title_text=str.title(parameter), row=row_nbr, col=2)
                fig.update_yaxes(title_text="Mean RMSE", row=row_nbr, col=2)
                row_nbr +=1
                counter +=1
            else:
                fig.add_trace(
                go.Scatter(name='Mean Training RMSE', x=x_feature, y=mean_train),
                row=row_nbr, col=1
                )   
                fig.add_trace(
                go.Scatter(name='Mean Validation RMSE', x=x_feature, y=mean_test),
                row=row_nbr, col=1
                )   
                fig.add_trace(
                go.Scatter(name='Best performing '+parameter, x=[best_params[parameter]],
                            y=-tmp[tmp['param_'+parameter] == best_params[parameter]]['mean_test_score']),
                row=row_nbr, col=1
                )
                fig.update_xaxes(title_text=str.title(parameter), row=row_nbr, col=1)
                fig.update_yaxes(title_text="Mean RMSE", row=row_nbr, col=1)
                counter += 1
        #Set title and show image
        fig.update_layout(title_text="Training and test results for various hyperparameters")
        fig.update_layout(hovermode="x unified")
        fig.show()

def perform_tuning(df_train,df_test,model,params,cv,type_search,list_features_drop,target_col,
                   return_importance=False,return_coeffs=False,n_iter=100,seed=101,n_jobs=-1,scoring='neg_root_mean_squared_error',y_log_transformed=False,suppress_graph=False,subplots=False):
    """
    This function performs Randomized Search Hyperparameter Search
    
    Inputs:
        X_test - A DataFrame with training set
        y_test - A DataFrame with test set
        model - The initialized model to use
        params - A dictionary containing the parameter search spaces to consider
        cv - The crossvalidation number or initialized CV object
        type_search - String with the name of the type of search (options: Random or Grid)
        list_features_drop - A list of strings with the features that should not be considered when training and predicting.
        target_col - String with the target column name
        return_importance - Boolean to signal need to return feature importance
        return_coeffs - Boolean to signal need to return feature coefficients
        n_iter - Number of search iterations
        seed - Random seed number
        n_jobs - Value to set the number of cores to use for job
        scoring - Type of scoring to use on tuning,
        y_log_transformed - If y has been log transformed or not.
        suppress_graph - Boolean, to identify if graphs should be output or not
        subplots - Boolean to identify if subplots of mean training and validation RMSE should be plotted for each specific hyperparameter that was grid tunned.
        
    Outputs:
        model - The created model 
        df_results - A Dataframe with the results of the Randomized Search
        best_params - The best parameters found for the model
        best_estimator - The best estimator found on the Search
        df_feature_importance - Optional. A DataFrame of Feature Importance
        df_feature_coeffs - Optional. A DataFrame of Feature coefficients.
    
    """
    #Get X_train, y_train, X_test, y_test from df_train/df_test and list of columns to drop
    X_train = df_train.drop(list_features_drop+[target_col],axis=1)
    X_test = df_test.drop(list_features_drop+[target_col],axis=1)
    y_train = df_train[target_col]
    y_test = df_test[target_col]
    
    # Use the random grid to search for best hyperparameters
    if str.lower(type_search) == 'random':
        model_search = RandomizedSearchCV(estimator = model, 
                                          param_distributions = params, 
                                          n_iter = n_iter, 
                                          cv = cv, 
                                          verbose=2, 
                                          random_state=seed, 
                                          n_jobs = n_jobs,
                                          scoring=scoring,
                                          return_train_score=True)
    elif str.lower(type_search) == 'grid':
        model_search = GridSearchCV(estimator = model, param_grid=params, cv = cv, scoring=scoring, n_jobs = n_jobs,verbose=2,return_train_score=True)
    else:
        return print('Choose either Grid or Random on the type_search parameter')
    # Fit the random search model

    #Train model
    model_search.fit(X_train,y_train)
    
    df_results = pd.DataFrame(model_search.cv_results_)
    df_results = df_results.sort_values('rank_test_score',ascending=False)
    best_params = model_search.best_params_
    best_estimator = model_search.best_estimator_
    
    if return_importance == True:
        df_feature_importance = pd.DataFrame(np.transpose(model_search.best_estimator_.feature_importances_),
                                             index=X_train.columns,
                                             columns=['Importance']).sort_values('Importance',ascending=False)
    if return_coeffs == True:
        df_feature_coeffs = pd.DataFrame(np.transpose(model_search.best_estimator_.coef_),
                                             index=X_train.columns,
                                             columns=['Importance']).sort_values('Importance',ascending=False)        
    
    #Make predictions and output RMSE for test set
    preds_train = best_estimator.predict(X_train)
    y_true_train = y_train

    preds_test = best_estimator.predict(X_test)
    y_true_test = y_test
    
    if y_log_transformed == True:
        preds_train = np.exp(preds_train)
        preds_test = np.exp(preds_test)
        y_true_train = np.exp(y_true_train)
        y_true_test = np.exp(y_true_test)


    print('--- Best Model predictions (note: predictions were "untransformed" before calculation of RMSE---')
    print('Training set RMSE: {}'.format(np.sqrt(mean_squared_error(y_true_train,preds_train))))
    print('Test set RMSE: {}'.format(np.sqrt(mean_squared_error(y_true_test,preds_test))))

    print('\n------------------------------ Auctioneer\'s estimated prices-------------------------------')
    print('Training set RMSE: {}'.format(np.sqrt(mean_squared_error(y_true_train,df_train['Estimated Price']))))
    print('Test set RMSE: {}'.format(np.sqrt(mean_squared_error(y_true_test,df_test['Estimated Price']))))
    
    if suppress_graph == False:
        fig = go.Figure(data=[
            go.Scatter(name='Mean CV train RMSE', x=df_results['rank_test_score'], y=-df_results['mean_train_score']),
            go.Scatter(name='Mean CV validation RMSE', x=df_results['rank_test_score'], y=-df_results['mean_test_score'])
        ])
        #Change the bar mode
        fig.update_layout(barmode='stack',title='Sold and unsold artworks of top 70 artists by auctioned pieces')
        fig.update_xaxes(title_text="Rank Test Score")
        if y_log_transformed:
            fig.update_yaxes(title_text="RMSE of log normalized target")
        else:
            fig.update_yaxes(title_text="RMSE")
        fig.update_layout(hovermode="x unified")
        fig.show()
        
    if subplots == True:
        if str.lower(type_search) == 'grid':
            subplot_hyperparameter(df_results,best_params)
        else:
            print('In randomized search, parameters are changed all at once and therefore no subplots will be produced.')
    if return_importance:
        return model, df_results, best_params, best_estimator,df_feature_importance
    elif return_coeffs:
        return model, df_results, best_params, best_estimator,df_feature_coeffs
    else:
        return model, df_results, best_params, best_estimator