#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as integr
from scipy.stats import expon

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from tqdm import tqdm

FILE_IN = ''
# Excel file which has at least the rows 
# AnimalID, Sex, Date of Birth,	Data, Birthtype, Region where 
#   Date of Birth has YYYY-mm-dd format,
#   Sex = Female / Male
#   Data = Date of deat in YYYY-mm-dd
#   Birthtyoe = CaptiveBorn / WildBorn

FILE_OUT = ''
# Output excel file of prepare_data_for_analysis()

OUTPUT_GIRAFFE_MODEL = '' # folder
OUTPUT_OKAPI_MODEL = '' # folder

def sort_stuff(infile = FILE_IN, outfile = FILE_OUT, normalise = 1):
    # normalise = positive float number, scales the age by normalise (in years) 
    
    out = {
        'Individual_ID': [],
        'Sex':[],
        'Birth' : [],
        'Year_of_Birth': [],
        'Status': [],
        'Age_of_Death_Days': [],
        'Age_of_Death_Years': [],
        'Place_Birth' : [],
        'Region': []
        }
    
    df = pd.read_excel(infile)
    
    for row in df.index:
        
        out['Individual_ID'].append(int( df.loc[row, 'AnimalID'] ))
        out['Sex'].append( df.loc[row, 'Sex'] )
        out['Birth'].append( df.loc[row, 'Date of Birth'] )
        out['Year_of_Birth'].append(int( df.loc[row, 'Date of Birth'].split('-')[0] ))
        
        out['Place_Birth'].append(df.loc[row, 'Birthtype'])
        out['Region'].append(df.loc[row, 'Region'])
        
        death_date = df.loc[row, 'Data']
        
        if death_date == None or pd.isnull(death_date):
            out['Status'].append('alive')
            out['Age_of_Death_Days'].append(np.nan)
            out['Age_of_Death_Years'].append(np.nan)
        else:            
            out['Status'].append('death')
            
            if type(death_date) == str:
                death_date_format = datetime.strptime(death_date, '%Y-%m-%d')
            else:
                death_date_format = death_date
            birth_date_format = datetime.strptime(df.loc[row, 'Date of Birth'], '%Y-%m-%d')            
            delta = death_date_format - birth_date_format    
            
            if normalise == 1:
                out['Age_of_Death_Days'].append( int(delta.days) )
                out['Age_of_Death_Years'].append( int(delta.days) / 365.25 )
            else:
                out['Age_of_Death_Days'].append( min(100 * int( delta.days) / 365.25 * 1 / normalise, 100) )
                out['Age_of_Death_Years'].append( min(100 * int( delta.days) / 365.25 * 1 / normalise, 100) ) 
    
    pd.DataFrame(out).to_excel(outfile)
            
        

def empirical_cumulative_mortality( df, filtering = {} ):
    # e.g. empirical_cumulative_mortality(filtering = { 'Year_of_Birth': ['<', 1990] , 'Status': ['=', 'death']})
    # returns a list (x,y) with x being the age in days and y being the proportion of dead individuals
    
    
    for k, val_list in filtering.items():
        for v in val_list:
            if v[0] == '=':
                df = df[ df[k] == v[1] ]
            if v[0] == '<':
                df = df[ df[k] <= v[1] ]
            if v[0] == '>':
                df = df[ df[k] >= v[1] ]
    
    max_age = int(max( df['Age_of_Death_Years'] ) )
    all_individuals = len(df['Age_of_Death_Years'])
    
    ret_x = []
    ret_y = []
    for j in range(0, max_age + 1):
        ret_x.append(j)
        ret_y.append( len( df[ df['Age_of_Death_Years'] <= j ]) / all_individuals )
    
    ret_x = [ x for x in ret_x ]

    return ret_x, ret_y
    


def empirical_survival( df, filtering = {} ):
    ret_x, ret_y = empirical_cumulative_mortality( df , filtering )
    ret_y = [ 1-y for y in ret_y ]
    return ret_x, ret_y

def empirical_cumulative_hazard( df, filtering = {} ):
    ret_x, ret_y = empirical_cumulative_mortality( df , filtering )
    
    ret_xnew = []
    ret_ynew = []
    
    for j in range(len(ret_y)):
        y = ret_y[j]
        x = ret_x[j]
        if y == 1:
            continue
        ret_ynew.append( np.log(1 / (1-y)) )
        ret_xnew.append(x)
    
    ret_ynew = np.array(ret_ynew) 

    return ret_xnew, list(ret_ynew)

def empirical_hazard_rate( df, filtering = {} ):
    ret_x, ret_y = empirical_cumulative_hazard( df , filtering )    
    ret_xnew = []
    ret_ynew = []
    
    for j in range(1, len(ret_x)-1):
        ret_xnew.append( ret_x[j] )
        ret_ynew.append( 0.5 * (ret_y[j+1] - ret_y[j-1]) )
    
    return ret_xnew, ret_ynew


def siler_survival(x, a1, a2, c, b1, b2):
    return np.exp( -1.0* siler_cumulative_hazard(x, a1, a2, c, b1, b2))



def siler_survival_log_product(x, a1, a2, c, b1, b2, verbose = 0):
    surv = siler_survival(x, a1, a2, c, b1, b2)
    if surv > 0.9999 and surv < 1.0001:
        return 0
    if surv < 0.0001:
        return 0
    return siler_survival(x, a1, a2, c, b1, b2) * np.log(siler_survival(x, a1, a2, c, b1, b2))


def siler_cumulative_hazard(x, a1, a2, c, b1, b2):
    return a1/a2 * (1 - np.exp( -a2*x )) + b1/b2 * ( np.exp(b2*x) - 1) + c*x


def siler_hazard_function(x, a1, a2,c, b1, b2):
    return a1 * np.exp( -a2*x ) + b1 * np.exp(b2*x) + c



def siler_life_expectancy(xvals, a1, a2,c, b1, b2):
    ret = []
    for x in xvals:
        ret.append( integr.quad(siler_survival, x, np.infty, args = (a1,a2,c,b1,b2))[0] / siler_survival(x, a1, a2,c, b1, b2) )
    return np.array(ret).astype(float)

def siler_lifespan_inequality(xvals, a1, a2,c, b1, b2):
    ret = []
    for x in xvals:        
        expectancy = siler_life_expectancy([x], a1, a2,c, b1, b2)[0]
        stuff = -1/expectancy * integr.quad( siler_survival_log_product, x, np.infty, args = (a1,a2,c,b1,b2) )[0]
        ret.append(stuff)
    return np.array(ret).astype(float)
    
def siler_lifespan_equality(xvals, a1, a2,c, b1, b2):
    ret = []
    for x in xvals: 
        ret.append(-1* np.log( siler_lifespan_inequality([x], a1, a2,c, b1, b2)[0] ))
    return np.array(ret).astype(float)




def make_siler_model_giraffe(df, iterations = 50, outfile = OUTPUT_GIRAFFE_MODEL +  'giraffe_siler_bootstrap.xlsx'):
    filtering_list = [
    { 'Year_of_Birth' : [ ['>', 1930], ['<', 1959]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1930], ['<', 1959]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1930], ['<', 1959]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    ]
    
    out = {'Lower_Year': [],
           'Upper_Year': [],
           'Sex': [],
           'a1': [], 'a2': [], 'c':[], 'b1': [], 'b2':[]}
    
    
    
    
    
    for step in tqdm(range(iterations)):
        
        # sample death age
        df_new = sample_dead_ages_giraffe(df)
        
        for filtering in filtering_list:       
        
            df_curr = df_new.copy()
            
            for k, val_list in filtering.items():
                
                # apply filter
                for v in val_list:
                    if v[0] == '=':
                        df_curr = df_curr[ df_curr[k] == v[1] ]
                    if v[0] == '<':
                        df_curr = df_curr[ df_curr[k] <= v[1] ]
                    if v[0] == '>':
                        df_curr = df_curr[ df_curr[k] >= v[1] ]
                        
            # bootstrap sample
            n_samples = len(df_curr.index)                
            df_sample = df_curr.sample(n = n_samples, replace = True)
            
            if 'Sex' in filtering.keys():
                outp = 'bootstrap_data/giraffen/{}/giraffe_siler_bootstrap_test_{}_{}_{}.xlsx'.format(str(filtering['Year_of_Birth'][0][1]), str(filtering['Year_of_Birth'][0][1]), filtering['Sex'][0][1], step)
            else:
                outp = OUTPUT_GIRAFFE_MODEL + 'bootstrap_data/giraffen/{}/giraffe_siler_bootstrap_test_{}_{}_{}.xlsx'.format(str(filtering['Year_of_Birth'][0][1]), str(filtering['Year_of_Birth'][0][1]), 'all', step)
            
            outf = os.path.dirname(outp)
            if not os.path.exists(outf):
                os.makedirs(outf)
                
            df_sample.to_excel(outp)
                           
            # fit model
            ret_x_emp_surv, ret_emp_surv = empirical_survival(df_sample, filtering) 
            popt2, pcov2 = opt.curve_fit( siler_survival, np.array(ret_x_emp_surv), np.array(ret_emp_surv), maxfev=15000, bounds = ([0, 3, 0, 0, 0], [4, 10, 0.5, 2, 2]) ) 
            
            a1, a2, c, b1, b2 = popt2[0], popt2[1], popt2[2], popt2[3], popt2[4]
            
            out['Lower_Year'].append( filtering['Year_of_Birth'][0][1] )
            out['Upper_Year'].append( filtering['Year_of_Birth'][1][1] )
            
            if not 'Sex' in filtering.keys():
                out['Sex'].append('all')
            else:
                out['Sex'].append( filtering['Sex'][0][1] )
            
            out['a1'].append(a1)
            out['a2'].append(a2)
            out['c'].append(c)
            out['b1'].append(b1)
            out['b2'].append(b2)
    
    pd.DataFrame(out).to_excel(outfile, index = False)
    
    
def make_siler_model_okapi(df, iterations = 500, outfile =  'okapi_siler_bootstrap.xlsx' ):
    filtering_list = [

    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    ]
    
    out = {'Lower_Year': [],
           'Upper_Year': [],
           'Sex': [],
           'a1': [], 'a2': [], 'c':[], 'b1': [], 'b2':[]}
        
    
    
    
    for step in tqdm(range(iterations)):
        
        # sample death age
        df_new = sample_dead_ages_okapi(df)
        
        
        for filtering in filtering_list:       
        
            df_curr = df_new.copy()
            
            for k, val_list in filtering.items():
                
                # apply filter
                for v in val_list:
                    if v[0] == '=':
                        df_curr = df_curr[ df_curr[k] == v[1] ]
                    if v[0] == '<':
                        df_curr = df_curr[ df_curr[k] <= v[1] ]
                    if v[0] == '>':
                        df_curr = df_curr[ df_curr[k] >= v[1] ]
                        
            # bootstrap sample
            n_samples = len(df_curr.index)                
            df_sample = df_curr.sample(n = n_samples, replace = True)
            
            if 'Sex' in filtering.keys():
                outp = OUTPUT_OKAPI_MODEL + 'bootstrap_data/okapi/{}/okapi_siler_bootstrap_test_{}_{}_{}.xlsx'.format(str(filtering['Year_of_Birth'][0][1]), str(filtering['Year_of_Birth'][0][1]), filtering['Sex'][0][1], step)
            else:
                outp = OUTPUT_OKAPI_MODEL + 'bootstrap_data/okapi/{}/okapi_siler_bootstrap_test_{}_{}_{}.xlsx'.format(str(filtering['Year_of_Birth'][0][1]), str(filtering['Year_of_Birth'][0][1]), 'all', step)
            
            outf = os.path.dirname(outp)
            if not os.path.exists(outf):
                os.makedirs(outf)
                
            df_sample.to_excel(outp)
                           
            # fit model
            ret_x_emp_surv, ret_emp_surv = empirical_survival(df_sample, filtering) 
            popt2, pcov2 = opt.curve_fit( siler_survival, np.array(ret_x_emp_surv), np.array(ret_emp_surv), maxfev=15000, bounds = (10**-5,5.5) ) 
            
            a1, a2, c, b1, b2 = popt2[0], popt2[1], popt2[2], popt2[3], popt2[4]
            
            out['Lower_Year'].append( filtering['Year_of_Birth'][0][1] )
            out['Upper_Year'].append( filtering['Year_of_Birth'][1][1] )
            
            if not 'Sex' in filtering.keys():
                out['Sex'].append('all')
            else:
                out['Sex'].append( filtering['Sex'][0][1] )
            
            out['a1'].append(a1)
            out['a2'].append(a2)
            out['c'].append(c)
            out['b1'].append(b1)
            out['b2'].append(b2)
    
    pd.DataFrame(out).to_excel(outfile, index = False)

def make_siler_model_okapi_relative(df, iterations = 500, outfile = OUTPUT_OKAPI_MODEL + 'okapi_relative_siler_bootstrap.xlsx'):
    filtering_list = [

    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    ]
    
    out = {'Lower_Year': [],
           'Upper_Year': [],
           'Sex': [],
           'a1': [], 'a2': [], 'c':[], 'b1': [], 'b2':[]}
        
    
    
    
    for step in tqdm(range(iterations)):
        
        # sample death age
        df_new = sample_dead_ages_okapi(df)
        
        # normalise data to 100 years and days
        df_new['Age_of_Death_Days'] = 100*df_new['Age_of_Death_Years'] / 33
        df_new['Age_of_Death_Years'] = 100*df_new['Age_of_Death_Years'] / 33
        
        for filtering in filtering_list:       
        
            df_curr = df_new.copy()
            
            for k, val_list in filtering.items():
                
                # apply filter
                for v in val_list:
                    if v[0] == '=':
                        df_curr = df_curr[ df_curr[k] == v[1] ]
                    if v[0] == '<':
                        df_curr = df_curr[ df_curr[k] <= v[1] ]
                    if v[0] == '>':
                        df_curr = df_curr[ df_curr[k] >= v[1] ]
                        
            # bootstrap sample
            n_samples = len(df_curr.index)                
            df_sample = df_curr.sample(n = n_samples, replace = True)
            
            if 'Sex' in filtering.keys():
                outp = OUTPUT_OKAPI_MODEL + 'bootstrap_data/okapi_relativ/{}/okapi_siler_bootstrap_test_{}_{}_{}.xlsx'.format(str(filtering['Year_of_Birth'][0][1]), str(filtering['Year_of_Birth'][0][1]), filtering['Sex'][0][1], step)
            else:
                outp = OUTPUT_OKAPI_MODEL + 'bootstrap_data/okapi_relativ/{}/okapi_siler_bootstrap_test_{}_{}_{}.xlsx'.format(str(filtering['Year_of_Birth'][0][1]), str(filtering['Year_of_Birth'][0][1]), 'all', step)
            
            outf = os.path.dirname(outp)
            if not os.path.exists(outf):
                os.makedirs(outf)
                
            df_sample.to_excel(outp)
                           
            # fit model
            ret_x_emp_surv, ret_emp_surv = empirical_survival(df_sample, filtering) 
            popt2, pcov2 = opt.curve_fit( siler_survival, np.array(ret_x_emp_surv), np.array(ret_emp_surv), maxfev=15000, bounds = ([0, 3, 0, 0, 0], [6, 30, 0.5, 2, 2]) ) 
            
            a1, a2, c, b1, b2 = popt2[0], popt2[1], popt2[2], popt2[3], popt2[4]
            
            out['Lower_Year'].append( filtering['Year_of_Birth'][0][1] )
            out['Upper_Year'].append( filtering['Year_of_Birth'][1][1] )
            
            if not 'Sex' in filtering.keys():
                out['Sex'].append('all')
            else:
                out['Sex'].append( filtering['Sex'][0][1] )
            
            out['a1'].append(a1)
            out['a2'].append(a2)
            out['c'].append(c)
            out['b1'].append(b1)
            out['b2'].append(b2)
    
    pd.DataFrame(out).to_excel(outfile, index = False)
    
    
def make_siler_model_giraffe_relative(df, iterations = 500, outfile = OUTPUT_GIRAFFE_MODEL + 'giraffe_relative_siler_bootstrap.xlsx'):
    filtering_list = [
    { 'Year_of_Birth' : [ ['>', 1930], ['<', 1959]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1930], ['<', 1959]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1930], ['<', 1959]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Sex': [ ['=', 'Male'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Sex': [ ['=', 'Female'] ], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    ]
    
    out = {'Lower_Year': [],
           'Upper_Year': [],
           'Sex': [],
           'a1': [], 'a2': [], 'c':[], 'b1': [], 'b2':[]}
        
    
    
    
    for step in tqdm(range(iterations)):
        
        # sample death age
        df_new = sample_dead_ages_giraffe(df)
        
        # normalise data to 100 years and days
        df_new['Age_of_Death_Days'] = 100*df_new['Age_of_Death_Years'] / 39
        df_new['Age_of_Death_Years'] = 100*df_new['Age_of_Death_Years'] / 39
        
        for filtering in filtering_list:       
        
            df_curr = df_new.copy()
            
            for k, val_list in filtering.items():
                
                # apply filter
                for v in val_list:
                    if v[0] == '=':
                        df_curr = df_curr[ df_curr[k] == v[1] ]
                    if v[0] == '<':
                        df_curr = df_curr[ df_curr[k] <= v[1] ]
                    if v[0] == '>':
                        df_curr = df_curr[ df_curr[k] >= v[1] ]
                        
            # bootstrap sample
            n_samples = len(df_curr.index)                
            df_sample = df_curr.sample(n = n_samples, replace = True)
            
            if 'Sex' in filtering.keys():
                outp = OUTPUT_GIRAFFE_MODEL + 'bootstrap_data/giraffen_relativ/{}/giraffen_siler_bootstrap_test_{}_{}_{}.xlsx'.format(str(filtering['Year_of_Birth'][0][1]), str(filtering['Year_of_Birth'][0][1]), filtering['Sex'][0][1], step)
            else:
                outp = OUTPUT_GIRAFFE_MODEL + 'bootstrap_data/giraffen_relativ/{}/giraffen_siler_bootstrap_test_{}_{}_{}.xlsx'.format(str(filtering['Year_of_Birth'][0][1]), str(filtering['Year_of_Birth'][0][1]), 'all', step)
            
            outf = os.path.dirname(outp)
            if not os.path.exists(outf):
                os.makedirs(outf)
                
            df_sample.to_excel(outp)
                           
            # fit model
            ret_x_emp_surv, ret_emp_surv = empirical_survival(df_sample, filtering) 
            popt2, pcov2 = opt.curve_fit( siler_survival, np.array(ret_x_emp_surv), np.array(ret_emp_surv), maxfev=15000, bounds = ([0, 1, 0, 0, 0], [4, 5, 0.5, 2, 2]) ) 
            
            a1, a2, c, b1, b2 = popt2[0], popt2[1], popt2[2], popt2[3], popt2[4]
            
            out['Lower_Year'].append( filtering['Year_of_Birth'][0][1] )
            out['Upper_Year'].append( filtering['Year_of_Birth'][1][1] )
            
            if not 'Sex' in filtering.keys():
                out['Sex'].append('all')
            else:
                out['Sex'].append( filtering['Sex'][0][1] )
            
            out['a1'].append(a1)
            out['a2'].append(a2)
            out['c'].append(c)
            out['b1'].append(b1)
            out['b2'].append(b2)
    
    pd.DataFrame(out).to_excel(outfile, index = False)
            

def fit_cum_hazard(df, filtering_list = [{}], print_raw_data = True, xlimits = [0, 42], maximum_age = 39, fit_bounds = (0.001,5.5), 
              labels = [], animal = 'giraffe', bootstrap_iterations = 1, normalise = True):
    
    
    
    filtering_list = [
    { 'Year_of_Birth' : [ ['>', 1930], ['<', 1959]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 1960], ['<', 1979]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,  
    { 'Year_of_Birth' : [ ['>', 1980], ['<', 1999]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    { 'Year_of_Birth' : [ ['>', 2000], ['<', 2019]], 'Place_Birth' : [ ['=', 'CaptiveBorn'] ] }  ,
    ]
    
    
    if animal == 'giraffe':
        df_new = sample_dead_ages_giraffe(df)
       
    
        if normalise:
            df_new['Age_of_Death_Days'] = 100*df_new['Age_of_Death_Years'] / 39
            df_new['Age_of_Death_Years'] = 100*df_new['Age_of_Death_Years'] / 39
            
    if animal == 'okapi':
        df_new = sample_dead_ages_okapi(df)
       
    
        if normalise:
            df_new['Age_of_Death_Days'] = 100*df_new['Age_of_Death_Years'] / 33
            df_new['Age_of_Death_Years'] = 100*df_new['Age_of_Death_Years'] / 33
        
    x_scatters_haz = {}
    x_scatters_surv = {}
    x_silers = {}
    cum_hazards_emp = {}
    cum_hazards_mod = {}
    emp_surv = {}
    siler_surv = {}
    siler_ineq = {}
    siler_equal = {}
    siler_exp = {}
    siler_hazard = {}
    
    counter = 0
    for filtering in filtering_list:
        
        # Fitting the cum hazard function
        ret_x_emp_cumhaz, ret_y = empirical_cumulative_hazard(df_new, filtering)
        #popt2, pcov2 = opt.curve_fit( siler_cumulative_hazard, np.array(ret_x_emp_cumhaz), np.array(ret_y), maxfev=5000, bounds = (0, 2) )        
        
        # Fitting based on survival function
        ret_x_emp_surv, ret_emp_surv = empirical_survival(df_new, filtering) 
        popt2, pcov2 = opt.curve_fit( siler_survival, np.array(ret_x_emp_surv), np.array(ret_emp_surv), maxfev=5000, bounds = fit_bounds ) 
        
        
        
        print("Parameters", popt2)
        
        x_plot = [0.1*x for x in range(xlimits[0], 10*xlimits[1])]
        
           
        
        y_fitcum_hazard = siler_cumulative_hazard(np.array(x_plot), *popt2)
        y_fit_survival = siler_survival(np.array(x_plot), *popt2)
        y_fit_inequality = siler_lifespan_inequality(np.array(x_plot), *popt2)
        y_fit_exp = siler_life_expectancy(np.array(x_plot), *popt2)
        y_fit_eq = siler_lifespan_equality(np.array(x_plot), *popt2)
        y_fit_hazard = siler_hazard_function(np.array(x_plot), *popt2)

        key = counter
        
        x_scatters_haz[key] = ret_x_emp_cumhaz
        x_scatters_surv[key] = ret_x_emp_surv
        x_silers[key] = x_plot
        cum_hazards_emp[key] = ret_y 
        cum_hazards_mod[key] = y_fitcum_hazard
        emp_surv[key] = ret_emp_surv 
        siler_surv[key] = y_fit_survival
        siler_ineq[key] = y_fit_inequality
        siler_equal[key] = y_fit_eq
        siler_exp[key] = y_fit_exp 
        siler_hazard[key] = y_fit_hazard
        
        counter += 1
        
    
    for filteringstring in x_scatters_surv.keys():        
        plt.plot(np.array(x_silers[filteringstring]), siler_ineq[filteringstring], label=labels[filteringstring])      
        plt.axvline(x = 1, color = 'b')
        plt.axvline(x = 4, color = 'b')
    plt.title('Lifespan inequality')
    plt.legend()
    plt.show()
    
    
    for filteringstring in x_scatters_surv.keys():        
        plt.plot(np.array(x_silers[filteringstring]), siler_exp[filteringstring], label=labels[filteringstring])        
        plt.axvline(x = 1, color = 'b')
        plt.axvline(x = 4, color = 'b')
    plt.title('Life expectancy')
    plt.legend()
    plt.show()
    
    for filteringstring in x_scatters_surv.keys():        
        plt.plot(np.array(x_silers[filteringstring]), siler_equal[filteringstring], label=labels[filteringstring])    
        plt.axvline(x = 1, color = 'b')
        plt.axvline(x = 4, color = 'b')
    plt.title('Life equality')
    plt.legend()
    plt.show()
    


def plot_basis_data(df, 
              filtering = {}):
    df = sample_dead_ages_giraffe(df)   
    for k, val_list in filtering.items():
        for v in val_list:
            if v[0] == '=':
                df = df[ df[k] == v[1] ]
            if v[0] == '<':
                df = df[ df[k] <= v[1] ]
            if v[0] == '>':
                df = df[ df[k] >= v[1] ]
    plt.scatter(np.array(df['Year_of_Birth'].values), np.array(df['Age_of_Death_Years'].values), marker='.')
    plt.title('Overview about filtered data (sanity check)')
    plt.show()
    
    plt.hist(np.array(df[ (df['Year_of_Birth'] > 2000) & (df['Age_of_Death_Years'] > 3) ]['Age_of_Death_Years'].values), bins = 20)
    plt.title('from 2000')
    plt.show()
    
    plt.hist(np.array(df[ (df['Status'] == 'death') & (df['Age_of_Death_Years'] > 3) ]['Age_of_Death_Years'].values), bins = 20)
    plt.title('all death')
    plt.show()
    


def sample_dead_ages_okapi(df_input):
    
    df_in = df_input.copy()
    df_in = df_in [ df_in['Place_Birth'] == 'CaptiveBorn' ]

    df = df_in[ df_in['Status'] == 'death' ]
    df_reference = df[ df['Year_of_Birth'] > 1980 ]
    

    conditional_expected_age = {}
    for age in range(0, 34, 1):        
            
        ages_to_live = [ int(x) - age for x in list(df_reference[ df_reference['Age_of_Death_Years']  >= age ]['Age_of_Death_Years'].values) ]
        mean_age_to_live = np.mean(ages_to_live)
        geometric_parameter = 0.9 / (1+mean_age_to_live)
        
        conditional_expected_age[age] = geometric_parameter
        if age >= 32:
            conditional_expected_age[age] = 1.0

    to_drop = []
    for row in df_in.index:
        if df_in.loc[row, 'Status'] == 'death':
            continue
        
        
        current_age = 2022 - df_in.loc[row, 'Year_of_Birth'] 
        random_die_age = current_age + np.random.geometric(p = conditional_expected_age[current_age] ) 
        
        if random_die_age > 33:
            to_drop.append(row)
            continue
            # censor data
            
        #random_die_age = current_age + np.random.choice(conditional_expected_age[current_age], size = 1)
        #print(current_age, random_die_age)
        df_in.loc[row, 'Age_of_Death_Years'] = min(int(random_die_age), 33)
        df_in.loc[row, 'Age_of_Death_Days'] = int(min(int(random_die_age), 33)*365.25)
        df_in.loc[row, 'Status'] = 'modified'
    
    df_in = df_in.drop( to_drop ) 
    return df_in



def sample_dead_ages_giraffe(df_input):
    
    df_in = df_input.copy()
    df_in = df_in [ df_in['Place_Birth'] == 'CaptiveBorn' ]

    df = df_in[ df_in['Status'] == 'death' ]
    df_reference = df[ df['Year_of_Birth'] > 1980 ]
    

    conditional_expected_age = {}
    for age in range(0, 40, 1):        
            
        ages_to_live = [ int(x) - age for x in list(df_reference[ df_reference['Age_of_Death_Years']  >= age ]['Age_of_Death_Years'].values) ]
        mean_age_to_live = np.mean(ages_to_live)
        geometric_parameter = 0.9 / (1 + mean_age_to_live)
        
        conditional_expected_age[age] = geometric_parameter
        if age >= 32:
            conditional_expected_age[age] = 0.5
    
    to_drop = []
    for row in df_in.index:

        if df_in.loc[row, 'Status'] == 'death':
            continue
        
        
        current_age = 2022 - df_in.loc[row, 'Year_of_Birth'] 
        random_die_age = current_age + np.random.geometric(p = conditional_expected_age[current_age] ) 
        
        if random_die_age > 39:
            to_drop.append(row)
            continue
            # censor data
        df_in.loc[row, 'Age_of_Death_Years'] = min(int(random_die_age), 39)
        df_in.loc[row, 'Age_of_Death_Days'] = int(min(int(random_die_age), 39)*365.25)
        df_in.loc[row, 'Status'] = 'modified'
    
    df_in = df_in.drop( to_drop )
        
    return df_in


def apply_model(path =  'F:/Nextcloud/Dierkes/Age_Modelling/',
                filetype = 'okapi',
                max_age = 100):
    
    
    bootstrap_file = path + filetype + '_siler_bootstrap.xlsx'
    output_raw = path + 'results/' + filetype + '_raw_data.xlsx'
    output_mean = path + 'results/' + filetype + '_mean.csv'
    output_std = path + 'results/' + filetype + '_std.csv'
    output_sem = path + 'results/' + filetype + '_sem.csv'
    
    df = pd.read_excel(bootstrap_file)
    
    ret = {
        'population' : [],
        'x' : [],
        'hazard': [],
        'survival':[],
        'inequality': [],
        'expectancy': [],
        'equality': []
        }
    
    x_values = [ 0.25*j for j in range(0, 4*max_age + 1)  ]
    
    for row in tqdm(df.index):
        lower = df.loc[row, 'Lower_Year']
        upper = df.loc[row, 'Upper_Year']
        a1 = float(df.loc[row, 'a1'])
        a2 = float(df.loc[row, 'a2'])
        c = float(df.loc[row, 'c'])
        b1 = float(df.loc[row, 'b1'])
        b2 = float(df.loc[row, 'b2'])
        sex = df.loc[row, 'Sex']
    
        key_name = '{}-{}_{}'.format(lower, upper, sex)
        popt2 = [a1, a2, c, b1, b2]
        
        

        y_fit_survival = siler_survival(np.array(x_values), *popt2)
        y_fit_inequality = siler_lifespan_inequality(np.array(x_values), *popt2)
        y_fit_exp = siler_life_expectancy(np.array(x_values), *popt2)
        y_fit_eq = siler_lifespan_equality(np.array(x_values), *popt2)
        y_fit_hazard = siler_hazard_function(np.array(x_values), *popt2)

        
        for j in range(len(x_values)):
            ret['x'].append(x_values[j])
            ret['population'].append(key_name)
            ret['hazard'].append(y_fit_hazard[j])
            ret['survival'].append(y_fit_survival[j])
            ret['inequality'].append(y_fit_inequality[j])
            ret['expectancy'].append(y_fit_exp[j])
            ret['equality'].append(y_fit_eq[j])
        
    df_raw = pd.DataFrame(ret)
    df_raw.to_csv(output_raw)
    df_raw.groupby(['population', 'x']).mean().to_csv(output_mean)
    df_raw.groupby(['population', 'x']).std().to_csv(output_std)
    df_sem = df_raw.groupby(['population', 'x']).std() / np.sqrt(df_raw.groupby(['population', 'x']).count() )
    df_sem.to_csv(output_sem)
    
    
        
def expectancy_plots():
    
    
    def draw_images(xlsx_mean, xlsx_std, output_path, marker, zeitpunkte, animalstring, exp_x_lim):
        
        colors = {
            '1900-1929': 'skyblue',
            '1930-1959': 'mediumblue',
            '1960-1979': 'rebeccapurple',
            '1980-1999': 'red',
            '2000-2019': 'gold'
            
            }
        
        if not os.path.exists(output_path + 'inequality'):
            os.makedirs(output_path + 'inequality')
            os.makedirs(output_path + 'equality')
            
        df_mean = pd.read_csv(xlsx_mean)
        df_std = pd.read_csv(xlsx_std)
        
        draw_eq = {}
        draw_ineq = {}
        for zeitpunkt in zeitpunkte:
            draw_eq[zeitpunkt] = {'all':[], 'Female':[], 'Male':[]}
            draw_ineq[zeitpunkt] = {'all':[], 'Female':[], 'Male':[]}
        
        for row in df_mean.index:
            pop = df_mean.loc[row, 'population']
            year = pop.split('_')[0]
            sex = pop.split('_')[1] # 1960-1979_Female
            zeitpunkt = float(df_mean.loc[row, 'x'].replace(',', '.'))
            
            if not zeitpunkt in zeitpunkte:
                continue
            
            expectancy = float(df_mean.loc[row, 'expectancy'].replace(',', '.'))
            inequality = float(df_mean.loc[row, 'inequality'].replace(',', '.'))
            equality = float(-1* np.log( inequality ) )
            
            exp_std = float(df_std.loc[row, 'expectancy'].replace(',', '.')) # /np.sqrt(1000)
            ineq_var = float(df_std.loc[row, 'inequality'].replace(',', '.'))
            ineq_std = ineq_var # /np.sqrt(1000)
            eq_std = ineq_var # /np.sqrt(1000)
            
            draw_eq[zeitpunkt][sex].append( [year, expectancy, equality, eq_std, exp_std  ] )
            draw_ineq[zeitpunkt][sex].append( [year, expectancy, inequality, ineq_std, exp_std  ] )
        
        for zeitpunkt, sex_dict in draw_eq.items():
            for sex in sex_dict.keys():
                title = '{}-{}-Age_{}'.format(animalstring, sex, zeitpunkt)
                plt.cla()
                plt.figure(figsize = (10,10), dpi = 300)
                x_vals = []
                y_vals = []
                cols = []
                yerr = []
                xerr = []
                for year, exp, equa, eq_std, exp_std in sex_dict[sex]:
                    x_vals.append(exp)
                    y_vals.append(equa)
                    cols.append(colors[year])
                    yerr.append(eq_std)
                    xerr.append(exp_std)
                    
                plt.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, linestyle='None', ecolor = 'grey', capsize = 1.0, elinewidth = .5)
                plt.scatter(x_vals, y_vals, c = cols, marker = marker, linestyle='None', s = 750)
                plt.ylabel('equality')
                plt.xlabel('expectancy')
                plt.xlim(exp_x_lim)
                plt.ylim((-2, 2))
                
                plt.title(title)
                plt.savefig(output_path + '/equality/' + title +"-equality.jpg", bbox_inches="tight")
                #plt.savefig(output_path + title +"-equality.pdf", format="pdf", bbox_inches="tight")
                
            
        for zeitpunkt, sex_dict in draw_ineq.items():
            for sex in sex_dict.keys():
                title = '{}-{}-Age_{}'.format(animalstring, sex, zeitpunkt)
                plt.cla()
                plt.figure(figsize = (10,10), dpi = 300)
                x_vals = []
                y_vals = []
                cols = []
                yerr = []
                xerr = []
                for year, exp, equa, ineq_std, exp_std in sex_dict[sex]:
                    x_vals.append(exp)
                    y_vals.append(equa)
                    cols.append(colors[year])
                    yerr.append(ineq_std)
                    xerr.append(exp_std)
                    
                plt.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, linestyle='None', ecolor = 'grey', capsize = 1.0, elinewidth = .5)
                plt.scatter(x_vals, y_vals, c = cols, marker = marker, linestyle='None', s = 750)
                plt.ylabel('inequality')
                plt.xlabel('expectancy')
                plt.xlim(exp_x_lim)
                plt.ylim((0, 2))
                
                plt.title(title)
                plt.savefig(output_path + '/inequality/' + title +"-inequality.jpg", bbox_inches="tight")
        
            
            
    def draw_images_combined(xlsx_mean_oka, xlsx_std_oka, 
                             xlsx_mean_gir, xlsx_std_gir,
                             output_path, marker_oka, marker_gir, 
                             zeitpunkte_oka, zeitpunkte_gir, 
                             animalstring, exp_x_lim):
        
        colors = {
            '1900-1929': 'skyblue',
            '1930-1959': 'mediumblue',
            '1960-1979': 'rebeccapurple',
            '1980-1999': 'red',
            '2000-2019': 'gold'
            
            }
        
        if not os.path.exists(output_path + 'inequality'):
            os.makedirs(output_path + 'inequality')
            os.makedirs(output_path + 'equality')
            
        df_mean_oka = pd.read_csv(xlsx_mean_oka)
        df_std_oka = pd.read_csv(xlsx_std_oka)
        df_mean_gir = pd.read_csv(xlsx_mean_gir)
        df_std_gir = pd.read_csv(xlsx_std_gir)
        
        draw_eq = {}
        draw_ineq = {}
        for zeitpunkt in [0,1,4]:
            draw_eq[zeitpunkt] = {'all': {'gir':[], 'oka':[]}, 'Female':{'gir':[], 'oka':[]}, 'Male':{'gir':[], 'oka':[]}}
            draw_ineq[zeitpunkt] = {'all':{'gir':[], 'oka':[]}, 'Female':{'gir':[], 'oka':[]}, 'Male':{'gir':[], 'oka':[]}}
        
        for row in df_mean_oka.index:
            pop = df_mean_oka.loc[row, 'population']
            year = pop.split('_')[0]
            sex = pop.split('_')[1] # 1960-1979_Female
            zeitpunkt = float(df_mean_oka.loc[row, 'x'].replace(',', '.'))
            
            if not zeitpunkt in zeitpunkte_oka.keys():
                continue
            
            expectancy = float(df_mean_oka.loc[row, 'expectancy'].replace(',', '.'))
            inequality = float(df_mean_oka.loc[row, 'inequality'].replace(',', '.'))
            equality = float(-1* np.log( inequality ) )
            
            exp_std = float(df_std_oka.loc[row, 'expectancy'].replace(',', '.')) # /np.sqrt(1000)
            ineq_var = float(df_std_oka.loc[row, 'inequality'].replace(',', '.'))
            ineq_std = ineq_var # /np.sqrt(1000)
            eq_std = ineq_var # /np.sqrt(1000)
            
            draw_eq[zeitpunkte_oka[zeitpunkt]][sex]['oka'].append( [year, expectancy, equality, eq_std, exp_std  ] )
            draw_ineq[zeitpunkte_oka[zeitpunkt]][sex]['oka'].append( [year, expectancy, inequality, ineq_std, exp_std ] )
            
        for row in df_mean_gir.index:
            pop = df_mean_gir.loc[row, 'population']
            year = pop.split('_')[0]
            sex = pop.split('_')[1] # 1960-1979_Female
            zeitpunkt = float(df_mean_gir.loc[row, 'x'].replace(',', '.'))
            
            if not zeitpunkt in zeitpunkte_gir.keys():
                continue
            
            expectancy = float(df_mean_gir.loc[row, 'expectancy'].replace(',', '.'))
            inequality = float(df_mean_gir.loc[row, 'inequality'].replace(',', '.'))
            equality = float(-1* np.log( inequality ) )
            
            exp_std = float(df_std_gir.loc[row, 'expectancy'].replace(',', '.')) # /np.sqrt(1000)
            ineq_var = float(df_std_gir.loc[row, 'inequality'].replace(',', '.'))
            ineq_std = ineq_var # /np.sqrt(1000)
            eq_std = ineq_var # /np.sqrt(1000)
            
            draw_eq[zeitpunkte_gir[zeitpunkt]][sex]['gir'].append( [year, expectancy, equality, eq_std, exp_std ] )
            draw_ineq[zeitpunkte_gir[zeitpunkt]][sex]['gir'].append( [year, expectancy, inequality, ineq_std, exp_std ] )
              
            
            
            
        
        for zeitpunkt, sex_dict in draw_eq.items():
            for sex in sex_dict.keys():
                title = '{}-{}-Age_{}'.format(animalstring, sex, zeitpunkt)
                plt.cla()
                plt.figure(figsize = (10,10), dpi = 300)
                
                
                x_vals = []
                y_vals = []
                cols = []
                yerr = []
                xerr = []
                for year, exp, equa, eq_std, exp_std in sex_dict[sex]['gir']:
                    x_vals.append(exp)
                    y_vals.append(equa)
                    cols.append(colors[year])
                    yerr.append(eq_std)
                    xerr.append(exp_std)

                plt.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, linestyle='None', ecolor = 'grey', capsize = 1.0, elinewidth = .5)
                plt.scatter(x_vals, y_vals, c = cols, marker = marker_gir, linestyle='None', s = 750)
                
                
                x_vals = []
                y_vals = []
                cols = []
                yerr = []
                xerr = []
                for year, exp, equa, eq_std, exp_std in sex_dict[sex]['oka']:
                    x_vals.append(exp)
                    y_vals.append(equa)
                    cols.append(colors[year])
                    yerr.append(eq_std)
                    xerr.append(exp_std)

                plt.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, linestyle='None', ecolor = 'grey', capsize = 1.0, elinewidth = .5)
                plt.scatter(x_vals, y_vals, c = cols, marker = marker_oka, linestyle='None', s = 750)
                
                
                
                plt.ylabel('equality')
                plt.xlabel('expectancy')
                plt.xlim(exp_x_lim)
                plt.ylim((-2, 2))
                
                plt.title(title)
                plt.savefig(output_path + '/equality/' + title +"-equality.jpg", bbox_inches="tight")
                #plt.savefig(output_path + title +"-equality.pdf", format="pdf", bbox_inches="tight")
                
            
            
            
            
            
        for zeitpunkt, sex_dict in draw_ineq.items():
            for sex in sex_dict.keys():
                title = '{}-{}-Age_{}'.format(animalstring, sex, zeitpunkt)
                plt.cla()
                plt.figure(figsize = (10,10), dpi = 300)
                
                
                x_vals = []
                y_vals = []
                cols = []
                yerr = []
                xerr = []
                for year, exp, inequa, ineq_std, exp_std in sex_dict[sex]['gir']:
                    x_vals.append(exp)
                    y_vals.append(inequa)
                    cols.append(colors[year])
                    yerr.append(ineq_std)
                    xerr.append(exp_std)

                plt.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, linestyle='None', ecolor = 'grey', capsize = 1.0, elinewidth = .5)
                plt.scatter(x_vals, y_vals, c = cols, marker = marker_gir, linestyle='None', s = 750)
                
                
                x_vals = []
                y_vals = []
                cols = []
                yerr = []
                xerr = []
                for year, exp, inequa, ineq_std, exp_std in sex_dict[sex]['oka']:
                    x_vals.append(exp)
                    y_vals.append(inequa)
                    cols.append(colors[year])
                    yerr.append(ineq_std)
                    xerr.append(exp_std)

                plt.errorbar(x_vals, y_vals, yerr=yerr, xerr=xerr, linestyle='None', ecolor = 'grey', capsize = 1.0, elinewidth = .5)
                plt.scatter(x_vals, y_vals, c = cols, marker = marker_oka, linestyle='None', s = 750)
                
                
                
                plt.ylabel('inequality')
                plt.xlabel('expectancy')
                plt.xlim(exp_x_lim)
                plt.ylim((0, 2))
                
                plt.title(title)
                plt.savefig(output_path + '/inequality/' + title +"-inequality.jpg", bbox_inches="tight")
                #plt.savefig(output_path + title +"-inequality.pdf", format="pdf", bbox_inches="tight")
        
    
    
    
    # Example for drawing images
    path_mean = ''
    path_std = ''
    zeitpunkte = [0, 3, 12]
    marker = '2'
    output_path = '' 
    animalstring = 'Oka_Rel'
    exp_x_lim = (0,50)
    draw_images(path_mean, path_std, output_path, marker, zeitpunkte, animalstring, exp_x_lim)
    
       
    # Example for drawing joined images  
    draw_images_combined(xlsx_mean_oka = '', 
                         xlsx_std_oka = '', 
                         xlsx_mean_gir = '', 
                         xlsx_std_gir = '',
                         output_path = '', 
                         marker_oka = '2', 
                         marker_gir = 'x', 
                         zeitpunkte_oka = { 0:0, 3:1, 12:4 }, 
                         zeitpunkte_gir = { 0:0, 2.5:1, 10.25:4 }, 
                         animalstring = 'Combined_rel', 
                         exp_x_lim = (0,50))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    