import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def _generate_secondary_dist(df):
    if df["v115ca1"] == 1 or df["v115cb1"] == 1:
        return 0
    else:
        return df["v115c2"]

def _generate_middle_dist(df):
    if df["v115ba1"] == 1 or df["v115bb1"] == 1:
        return 0
    else:
        return df["v115b2"]

def _gen_bpl(hv134):
    if hv134 == 1:
        return 1
    elif hv134 == 2:
        return 0
    else:
        return np.nan

def _gen_treat(age, treatment, control):
    if age in treatment:
        return 1
    elif age in control:
        return 0
    else:
        return np.nan    
    

 
def gen_dlhs_reg_data(dlhs_long_wdist):
    df = dlhs_long_wdist.copy()
    df = df.loc[(df['state'] == 10)| (df['state'] ==20)]
    df['inschool'] = df['school'].apply(lambda x: 1 if x==1 else 0)
    df['currgrade'] = df.apply(lambda x: x['grade'] + 1 if x['inschool'] == 1 else np.nan, axis=1)
    df['enrollment_secschool'] = df.apply(lambda x: 1 if x['currgrade'] == 9 or x['grade'] >= 9 else 0, axis=1)
    df['enrollment_middleschool'] = df.apply(lambda x: 1 if x['currgrade'] == 8 or x['grade'] >= 8 else 0, axis=1)
    # Create dummies 
    
    ## Female dummy
    df['female'] = df['sex'].apply(lambda x: 1 if x == 2 else 0)
    ## Bihar/Jharkhand dummy
    df['bihar'] = df['state'].apply(lambda x: 1 if x == 10 else 0)
    ## Relationship Indicator 
    df['child_sample'] = df['relationship'].apply(lambda x: 1 if x in [3, 5, 8, 10]  else 0)
    ## Scheduled caste/scheduled tribe
    df['sc'] = df['hv116b'].apply(lambda x: 1 if x == 1 else 0)
    df['st'] = df['hv116b'].apply(lambda x: 1 if x == 2 else 0)
    df['obc'] = df['hv116b'].apply(lambda x: 1 if x == 3 else 0)
    df['highcaste'] = df['hv116b'].apply(lambda x: 1 if x==4 else 0)
    df['scst'] = df['hv116b'].apply(lambda x: 1 if x in [1, 2] else 0)
    ## Dummy for Religion, hindu , muslim and others 
    df['hindu'] = df['hv115'].apply(lambda x: 1 if x == 1 else 0)
    df['muslim'] = df['hv115'].apply(lambda x: 1 if x == 2 else 0)
    df['other'] = df['hv115'].apply(lambda x: 1 if x > 2 else 0)
    ## Generate electricity dummy 
    df['electricity'] = df['hv129a'].apply(lambda x: 1 if x == 1 else 0)
    df['mattress'] = df['hv129b'].apply(lambda x: 1 if x == 1 else 0)
    df['cooker'] = df['hv129c'].apply(lambda x: 1 if x == 1 else 0)
    df['chair'] = df['hv129d'].apply(lambda x: 1 if x == 1 else 0)
    df['sofa'] = df['hv129e'].apply(lambda x: 1 if x == 1 else 0)
    df['bed'] = df['hv129f'].apply(lambda x: 1 if x == 1 else 0)
    df['table'] = df['hv129g'].apply(lambda x: 1 if x == 1 else 0)
    df['fan'] = df['hv129h'].apply(lambda x: 1 if x == 1 else 0)
    df['sewing'] = df['hv129l'].apply(lambda x: 1 if x == 1 else 0)
    df['phone'] = df.apply(lambda x: 1 if x['hv129m'] == 1 or x['hv129n'] == 1 else 0, axis=1)
    df['computer'] = df['hv129o'].apply(lambda x: 1 if x == 1 else 0)
    df['fridge'] = df['hv129p'].apply(lambda x: 1 if x == 1 else 0)
    df['washing_machine'] = df['hv129q'].apply(lambda x: 1 if x == 1 else 0)
    df['watch'] = df['hv129r'].apply(lambda x: 1 if x == 1 else 0)
    df['motor_cycle'] = df['hv129t'].apply(lambda x: 1 if x == 1 else 0)
    df['animal_cart'] = df['hv129u'].apply(lambda x: 1 if x == 1 else 0)
    df['car'] = df['hv129v'].apply(lambda x: 1 if x == 1 else 0)
    df['tractor'] = df['hv129w'].apply(lambda x: 1 if x == 1 else 0)
    df['water_pump'] = df['hv129x'].apply(lambda x: 1 if x == 1 else 0)
    df['thresher'] = df['hv129y'].apply(lambda x: 1 if x == 1 else 0)
    ## Bike ownership
    df['bike'] = df['hv129s'].apply(lambda x: 1 if x == 1 else 0)
    ## Ownership of radio/tv - idea= access to govt schemes
    df['media'] = df.apply(lambda x: 1 if x['hv129i'] == 1 or x['hv129j'] == 1 or x['hv129k'] == 1 else 0, axis=1)
    ## Indicator for poor farmers having less than 5 acres of land 
    df['land'] = df.apply(lambda x: 1 if x['hv130'] == 2 or x['hv131'] < 5 else 0, axis = 1)
    ## Owns BPL card
    df['bpl'] = df['hv134'].apply(lambda x: 1 if x == 1 else 0)
    ## Generate electricity dummy 
    df['electricity'] = df['hv129a'].apply(lambda x: 1 if x == 1 else 0)
    df['mattress'] = df['hv129b'].apply(lambda x: 1 if x == 1 else 0)
    df['cooker'] = df['hv129c'].apply(lambda x: 1 if x == 1 else 0)
    df['chair'] = df['hv129d'].apply(lambda x: 1 if x == 1 else 0)
    df['sofa'] = df['hv129e'].apply(lambda x: 1 if x == 1 else 0)
    df['bed'] = df['hv129f'].apply(lambda x: 1 if x == 1 else 0)
    df['table'] = df['hv129g'].apply(lambda x: 1 if x == 1 else 0)
    df['fan'] = df['hv129h'].apply(lambda x: 1 if x == 1 else 0)
    df['sewing'] = df['hv129l'].apply(lambda x: 1 if x == 1 else 0)
    df['phone'] = df.apply(lambda x: 1 if x['hv129m'] == 1 or x ['hv129n'] else 0, axis 
    = 1)
    df['computer'] = df['hv129o'].apply(lambda x: 1 if x == 1 else 0)
    df['fridge'] = df['hv129p'].apply(lambda x: 1 if x == 1 else 0)
    df['washing_machine'] = df['hv129q'].apply(lambda x: 1 if x == 1 else 0)
    df['watch'] = df['hv129r'].apply(lambda x: 1 if x == 1 else 0)
    df['motor_cycle'] = df['hv129t'].apply(lambda x: 1 if x == 1 else 0)
    df['animal_cart'] = df['hv129u'].apply(lambda x: 1 if x == 1 else 0)
    df['car'] = df['hv129v'].apply(lambda x: 1 if x == 1 else 0)
    df['tractor'] = df['hv129w'].apply(lambda x: 1 if x == 1 else 0)
    df['water_pump'] = df['hv129x'].apply(lambda x: 1 if x == 1 else 0)
    df['thresher'] = df['hv129y'].apply(lambda x: 1 if x == 1 else 0)
    ## Bike ownership
    df['bike'] = df['hv129s'].apply(lambda x: 1 if x == 1 else 0)
    ## Ownership of radio/tv - idea= access to govt schemes
    df['media'] = df.apply(lambda x: 1 if x['hv129i'] == 1 or x['hv129j'] == 1 or 
    x['hv129k'] == 1 else 0, axis=1)
    ## Indicator for poor farmers having less than 5 acres of land 
    df['land'] = df.apply(lambda x: 1 if x['hv130'] == 2 or x['hv131'] == 1 else 0, axis = 1)
    ## Owns BPL card
    df['bpl'] = df['hv134'].apply(_gen_bpl)
    ## Creating Distance to School Variables
    ## Create indicator variable - if secondary school is present in the village 
    df['secschool'] = df['v115ca1'].apply(lambda x: 1 if x == 1 else 0)
    df['secondarydist'] = df.apply(_generate_secondary_dist, axis=1)
    df['longdist'] = df['secondarydist'].apply(lambda x: 1 if x > 3 else 0)
    df['secondarydistq'] = df['secondarydist'].apply(lambda x: x ** 2)
    df['highsecondarydist'] = df.apply(lambda x: 0 if x['v115da1'] == 1 or x['v115db1'] == 1 else x['v115d2'], axis = 1)
    ## Drop some weird observations - missing distances etc 
    df = df.round({'secondarydist': 0, 'highsecondarydist': 0})
    df.drop(df.loc[df['secondarydist'] == 99.].index, inplace=True)
    df.drop(df.loc[df['highsecondarydist'] == 99.].index, inplace=True)
    ## Generating distance to secondary school in logs and interacting 
    df['lsecondarydist'] = np.log(df['secondarydist'])
    ## Access to middle school (govt)- omitted variable, it is possible that a kid wont 
    # go to secondary school and up if there is no middle school in the village
    df['middle'] = df['v115bb1'].apply(lambda x: 1 if x == 1  else 0)
    ## Create the distance variable to the nearest middle school 
    df['middledist'] = df.apply(_generate_middle_dist, axis=1)
        ## Create indicator variable- if primary school is present in the village
    df['primary'] = df['v115aa1'].apply(lambda x: 1 if x == 1 else 0)
        ## Create the distance variable to the nearest primary school
    df['primarydist'] = df.apply(lambda x: 0 if x['v115aa1'] == 1 or x['v115ab1'] == 1 else 0, axis = 1)
    ## Access to college- controlling as this could change incentives in a way that might bias the estimates upwards
    df['pubcollege'] = df['v115ea1'].apply(lambda x: 1 if x == 1 else 0)
    df['pvtcollege'] = df['v115eb1'].apply(lambda x: 1 if x == 1 else 0)
    df['college'] = df.apply(lambda x: 1 if x['pubcollege'] == 1 or x ['pvtcollege'] else 0, axis = 1)
    ## Create variables - facilities available in the village
    df['postoff'] = df['v122a'].apply(lambda x: 1 if x == 1 else 0)
    df['bank'] = df['v122d'].apply(lambda x: 1 if x == 1 else 0)
    ## Create distance to local facilities by village
    df['towndist'] = df['v110']
    df['hqdist'] = df['v111']
    df['railwaydist'] = df['v112']
    df['busdist'] = df['v113']
        ## Long Distance to different centers
        # Railway station 
    df['longraildist'] = df['railwaydist'].apply(lambda x: 1 if x > 3 else 0)
    # Bus distance
    df['longbusdist'] = df['busdist'].apply(lambda x: 1 if x > 3 else 0)
    # Distance to district Headquarter
    df['longhqdist'] = df['hqdist'].apply(lambda x: 1 if x > 3 else 0)
    # Distance to nearest town
    df['longtowndist'] = df['towndist'].apply(lambda x: 1 if x > 3 else 0)
    ## Create population measures
    df['censuspop'] = df['v101a']
    df['lcensuspop'] = np.log(df['censuspop'])
    df['currpop'] = df['v101b']
    df['lcurrpop'] = np.log(df['currpop'])
        ## Create no. of hh in the village
    df['tothhvill'] = df['v102']
        ## Creating district borders - Bihar (BH) and Jharkhand (JH)
        # BH (District name with district codes) = Katihar = 1010, Bhagalpur= 1022, Banka = 1023, Rohtas =1032, Aurangabad = 1034, Gaya = 1035, Nawada = 1036, Jamui = 1037 *
        # JH (District name with district codes) = Garawah =2001, Palamu = 2002, Chatra= 2003, Hazaribagh = 2004, Kodarma = 2005, Giridih = 2006, Deoghar = 2007, Godda= 2008, Sahibganj = 2009, Dumka = 2011 *
    df['distborder'] = df['dist'].apply(lambda x: 1 if x == 1010 or x == 1022 or x == 1023 or x == 1032 or x == 1034 or
    x == 1035 or x == 1036 or x == 1037 or x == 2001 or x == 2002 or x == 2003 or x == 2004 or x == 2005 or x == 2006
    or x == 2007 or x == 2008 or x == 2009 or x == 2011 else 0)
    ## Generating various treatments and controls
    # Treatment 1 = treat1 = is 14/15 years old, control is 16/17 years old. This is the main treatment variable




    df['treat1'] = df['age'].apply(_gen_treat, args=([14, 15], [16, 17]))
    df['treat2'] = df['age'].apply(_gen_treat, args=([13, 14, 15], [16, 17]))
    # Treatment 3= treat3 = is 14/15 years old, control is 16 years old
    df['treat3'] = df['age'].apply(_gen_treat, args=([14, 15], [16]))
    # Treatment 4 = treat4 = is 13/14/15 years old, control is 16 years old
    df['treat4'] = df['age'].apply(_gen_treat, args=([13, 14, 15], [16]))
    # Treatment 5 = treat5 = is 13/14 years old, control is 15/16 years old. This is for girls in grade 8 
    df['treat5'] = df['age'].apply(_gen_treat, args=([13, 14], [15, 16]))
    ## Various Interactions needed to perform DD, DDD, DDDD Analysis
    #  DD Interactions 
    # Heterogeneity by Caste
    df['treat1_scst'] = df['treat1'] * df['scst']
    df['treat1_sc'] = df['treat1'] * df['sc']
    df['treat1_st'] = df['treat1'] * df['st']
    df['treat1_obc'] = df['treat1'] * df['obc']
    df['treat1_obc'] = df['treat1'] * df['obc']

    df['treat2_scst'] = df['treat2'] * df['scst']
    df['treat2_sc'] = df['treat2'] * df['sc']
    df['treat2_st'] = df['treat2'] * df['st']
    df['treat2_obc'] = df['treat2'] * df['obc']

    df['treat3_scst'] = df['treat3'] * df['scst']
    df['treat3_sc'] = df['treat3'] * df['sc']
    df['treat3_st'] = df['treat3'] * df['st']
    df['treat3_obc'] = df['treat3'] * df['obc']

    df['treat4_scst'] = df['treat4'] * df['scst']
    df['treat4_sc'] = df['treat4'] * df['sc']
    df['treat4_st'] = df['treat4'] * df['st']
    df['treat4_obc'] = df['treat4'] * df['obc']

    df['treat5_scst'] = df['treat5'] * df['scst']
    df['treat5_sc'] = df['treat5'] * df['sc']
    df['treat5_st'] = df['treat5'] * df['st']
    df['treat5_obc'] = df['treat5'] * df['obc']

    df['female_sc'] = df['female'] * df['sc']
    df['female_st'] = df['female'] * df['st']
    df['female_obc'] = df['female'] * df['obc']

    df['bihar_sc'] = df['bihar'] * df['sc']
    df['bihar_st'] = df['bihar'] * df['st']
    df['bihar_obc'] = df['bihar'] * df['obc']

        ## Various interactions by distance
    df['treat1_longdist'] = df['treat1'] * df['longdist']
    df['treat2_longdist'] = df['treat2'] * df['longdist']
    df['treat3_longdist'] = df['treat3'] * df['longdist']
    df['treat4_longdist'] = df['treat4'] * df['longdist']
    df['treat5_longdist'] = df['treat5'] * df['longdist']
    df['female_longdist'] = df['female'] * df['longdist']
    df['bihar_longdist'] = df['bihar'] * df['longdist']

        ##DDD Interactions 
        # caste 
    df['treat1_bihar_sc'] = df['treat1'] * df['bihar'] * df['sc']
    df['treat2_bihar_sc'] = df['treat2'] * df['bihar'] * df['sc']
    df['treat3_bihar_sc'] = df['treat3'] * df['bihar'] * df['sc']
    df['treat4_bihar_sc'] = df['treat4'] * df['bihar'] * df['sc']
    df['treat5_bihar_sc'] = df['treat5'] * df['bihar'] * df['sc']

    df['treat1_bihar_st'] = df['treat1'] * df['bihar'] * df['st']
    df['treat2_bihar_st'] = df['treat2'] * df['bihar'] * df['st']
    df['treat3_bihar_st'] = df['treat3'] * df['bihar'] * df['st']
    df['treat4_bihar_st'] = df['treat4'] * df['bihar'] * df['st']
    df['treat5_bihar_st'] = df['treat5'] * df['bihar'] * df['st']

    df['treat1_bihar_obc'] = df['treat1'] * df['bihar'] * df['obc']
    df['treat2_bihar_obc'] = df['treat2'] * df['bihar'] * df['obc']
    df['treat3_bihar_obc'] = df['treat3'] * df['bihar'] * df['obc']
    df['treat4_bihar_obc'] = df['treat4'] * df['bihar'] * df['obc']
    df['treat5_bihar_obc'] = df['treat5'] * df['bihar'] * df['obc']

    df['treat1_female_sc'] = df['treat1'] * df['female'] * df['sc']
    df['treat1_female_sc'] = df['treat2'] * df['female'] * df['sc']
    df['treat3_female_sc'] = df['treat3'] * df['female'] * df['sc']
    df['treat4_female_sc'] = df['treat4'] * df['female'] * df['sc']
    df['treat5_female_sc'] = df['treat5'] * df['female'] * df['sc']

    df['treat1_female_st'] = df['treat1'] * df['female'] * df['st']
    df['treat2_female_st'] = df['treat2'] * df['female'] * df['st']
    df['treat3_female_st'] = df['treat3'] * df['female'] * df['st']
    df['treat4_female_st'] = df['treat4'] * df['female'] * df['st']
    df['treat5_female_st'] = df['treat5'] * df['female'] * df['st']

    df['treat1_female_obc'] = df['treat1'] * df['female'] * df['obc']
    df['treat2_female_obc'] = df['treat2'] * df['female'] * df['obc']
    df['treat3_female_obc'] = df['treat3'] * df['female'] * df['obc']
    df['treat4_female_obc'] = df['treat4'] * df['female'] * df['obc']
    df['treat5_female_obc'] = df['treat5'] * df['female'] * df['obc']

    df['female_bihar_sc'] = df['female'] * df['bihar'] * df['sc']
    df['female_bihar_st'] = df['female'] * df['bihar'] * df['st']
    df['female_bihar_obc'] = df['female'] * df['bihar'] * df['obc']

        # DDDD Interactions 
    df['treat1_female_bihar_longdist'] = df['treat1'] * df['female'] * df['bihar'] * df['longdist']
    df['treat2_female_bihar_longdist'] = df['treat2'] * df['female'] * df['bihar'] * df['longdist']
    df['treat3_female_bihar_longdist'] = df['treat3'] * df['female'] * df['bihar'] * df['longdist']
    df['treat4_female_bihar_longdist'] = df['treat4'] * df['female'] * df['bihar'] * df['longdist']
    df['treat5_female_bihar_longdist'] = df['treat5'] * df['female'] * df['bihar'] * df['longdist']

        # DDD by caste 
    df['treat1_female_bihar_sc'] = df['treat1'] * df['female'] * df['bihar'] * df['sc']
    df['treat2_female_bihar_sc'] = df['treat2'] * df['female'] * df['bihar'] * df['sc']
    df['treat3_female_bihar_sc'] = df['treat3'] * df['female'] * df['bihar'] * df['sc']
    df['treat4_female_bihar_sc'] = df['treat4'] * df['female'] * df['bihar'] * df['sc']
    df['treat5_female_bihar_sc'] = df['treat5'] * df['female'] * df['bihar'] * df['sc']

    df['treat1_female_bihar_st'] = df['treat1'] * df['female'] * df['bihar'] * df['st']
    df['treat2_female_bihar_st'] = df['treat2'] * df['female'] * df['bihar'] * df['st']
    df['treat3_female_bihar_st'] = df['treat3'] * df['female'] * df['bihar'] * df['st']
    df['treat4_female_bihar_st'] = df['treat4'] * df['female'] * df['bihar'] * df['st']
    df['treat5_female_bihar_st'] = df['treat5'] * df['female'] * df['bihar'] * df['st']

    df['treat1_female_bihar_obc'] = df['treat1'] * df['female'] * df['bihar'] * df['obc']
    df['treat2_female_bihar_obc'] = df['treat2'] * df['female'] * df['bihar'] * df['obc']
    df['treat3_female_bihar_obc'] = df['treat3'] * df['female'] * df['bihar'] * df['obc']
    df['treat4_female_bihar_obc'] = df['treat4'] * df['female'] * df['bihar'] * df['obc']
    df['treat5_female_bihar_obc'] = df['treat5'] * df['female'] * df['bihar'] * df['obc']

        #DDD by Treatment Age, State and Gender 
    df['treat1_female_bihar'] = df['treat1'] * df['female'] * df['bihar']
    df['treat2_female_bihar'] = df['treat2'] * df['female'] * df['bihar']
    df['treat3_female_bihar'] = df['treat3'] * df['female'] * df['bihar']
    df['treat4_female_bihar'] = df['treat4'] * df['female'] * df['bihar']
    df['treat5_female_bihar'] = df['treat5'] * df['female'] * df['bihar']

    df['treat1_female'] = df['treat1'] * df['female'] 
    df['treat2_female'] = df['treat2'] * df['female'] 
    df['treat3_female'] = df['treat3'] * df['female'] 
    df['treat4_female'] = df['treat4'] * df['female'] 
    df['treat5_female'] = df['treat5'] * df['female'] 

    df['treat1_bihar'] = df['treat1'] * df['bihar'] 
    df['treat2_bihar'] = df['treat2'] * df['bihar'] 
    df['treat3_bihar'] = df['treat3'] * df['bihar'] 
    df['treat4_bihar'] = df['treat4'] * df['bihar'] 
    df['treat5_bihar'] = df['treat5'] * df['bihar'] 

    df['female_bihar'] = df['female'] * df['bihar'] 

        # Distance

    df['treat1_bihar_longdist'] = df['treat1'] * df['bihar'] * df['longdist']
    df['treat2_bihar_longdist'] = df['treat2'] * df['bihar'] * df['longdist']
    df['treat3_bihar_longdist'] = df['treat3'] * df['bihar'] * df['longdist']
    df['treat4_bihar_longdist'] = df['treat4'] * df['bihar'] * df['longdist']
    df['treat5_bihar_longdist'] = df['treat5'] * df['bihar'] * df['longdist']

    df['treat1_female_longdist'] = df['treat1'] * df['female'] * df['longdist']
    df['treat2_female_longdist'] = df['treat2'] * df['female'] * df['longdist']
    df['treat3_female_longdist'] = df['treat3'] * df['female'] * df['longdist']
    df['treat4_female_longdist'] = df['treat4'] * df['female'] * df['longdist']
    df['treat5_female_longdist'] = df['treat5'] * df['female'] * df['longdist']

    df['female_bihar_longdist'] = df['female'] * df['bihar'] * df['longdist']

        # Creating Asset Index using PCA - Land, Poverty Status, Access to Radio/TV, and Electricity
    
    pca_asset_cols = df.loc[:, ['land', 'bpl', 'media', 'electricity']]
    df['pca_asset'] = PCA(n_components=1).fit_transform(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(pca_asset_cols)).T[0]
    # Creating Socio-Economic-Status Index using PCA - Caste, Household Head School, Land, Poverty Status, Access to TV/Radio, and Electricity 
    pcs_socioeconomic_cols = df.loc[:,['sc', 'st', 'obc', 'hindu', 'muslim', 'hhheadschool', 'land', 'bpl', 'media', 'electricity']].copy()
    # pcs_socioeconomic_notna = pcs_socioeconomic_cols.dropna(axis=0, how="any")



    df['pca_ses'] = PCA(n_components=1).fit_transform(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(pcs_socioeconomic_cols)).T[0]

        ## With Asset Index defined by "pca_asset" 
    #     #  Generating Interactions for Asset Index 
    df['treat1_female_bihar_pca_asset'] = df['treat1'] * df['female'] * df['bihar'] * df['pca_asset']
    df['treat1_female_pca_asset'] = df['treat2'] * df['female'] * df['pca_asset']
    df['female_bihar_pca_asset'] = df['female'] * df['bihar'] * df['pca_asset']
    df['treat1_bihar_pca_asset'] = df['treat1'] * df['bihar'] * df['pca_asset']
    df['treat1_pca_asset'] = df['treat1'] * df['pca_asset'] 
    df['female_pca_asset'] = df['female'] * df['pca_asset']
    df['bihar_pca_asset'] = df['bihar'] * df['pca_asset']

    #     ## With Asset Index defined by "pca_ses"
    #     # Generating Interactions for SES Index
    df['treat1_female_bihar_pca_ses'] = df['treat1'] * df['female'] * df['bihar'] * df['pca_ses']
    df['treat1_female_pca_ses'] = df['treat2'] * df['female'] * df['pca_ses']
    df['female_bihar_pca_ses'] = df['female'] * df['bihar'] * df['pca_ses']
    df['treat1_bihar_pca_ses'] = df['treat1'] * df['bihar'] * df['pca_ses']
    df['treat1_pca_ses'] = df['treat1'] * df['pca_ses'] 
    df['female_pca_ses'] = df['female'] * df['pca_ses']
    df['bihar_pca_ses'] = df['bihar'] * df['pca_ses']

    #     ##  Interactions for Muslims versus High Caste
    df['treat1_female_bihar_muslim'] = df['treat1'] * df['female'] * df['bihar'] * df['muslim']
    df['treat1_female_muslim'] = df['treat2'] * df['female'] * df['muslim']
    df['female_bihar_muslim'] = df['female'] * df['bihar'] * df['muslim']
    df['treat1_bihar_muslim'] = df['treat1'] * df['bihar'] * df['muslim']
    df['treat1_muslim'] = df['treat1'] * df['muslim'] 
    df['female_muslim'] = df['female'] * df['muslim']
    df['bihar_muslim'] = df['bihar'] * df['muslim']

    df = df[
        [
            'state',
            'dist',
            'vpsu',
            'village',
            'age',
            'hhwt',
            'hhheadmale',
            'hhheadschool',
            'currgrade',
            'enrollment_secschool',
            'enrollment_middleschool',
            'female',
            'bihar',
            'sc',
            'st',
            'obc',
            'highcaste',
            'hindu',
            'muslim',
            'electricity',
            'media',
            'land',
            'bpl',
            'secschool',
            'secondarydist',
            'longdist',
            'middle',
            'primary',
            'postoff',
            'bank',
            'towndist',
            'hqdist',
            'railwaydist',
            'busdist',
            'lcurrpop',
            'distborder',
            'treat1',
            'treat2',
            'treat3',
            'treat4',
            'treat5',
            'treat1_sc',
            'treat1_st',
            'treat1_obc',
            'female_sc',
            'female_st',
            'female_obc',
            'bihar_sc',
            'bihar_st',
            'bihar_obc',
            'treat1_longdist',
            'treat2_longdist',
            'treat3_longdist',
            'treat4_longdist',
            'female_longdist',
            'bihar_longdist',
            'treat1_bihar_sc',
            'treat1_bihar_st',
            'treat1_bihar_obc',
            'treat1_female_sc',
            'treat1_female_st',
            'treat1_female_obc',
            'female_bihar_sc',
            'female_bihar_st',
            'female_bihar_obc',
            'treat1_female_bihar_longdist',
            'treat2_female_bihar_longdist',
            'treat3_female_bihar_longdist',
            'treat4_female_bihar_longdist',
            'treat1_female_bihar_sc',
            'treat1_female_bihar_st',
            'treat1_female_bihar_obc',
            'treat1_female_bihar',
            'treat2_female_bihar',
            'treat3_female_bihar',
            'treat4_female_bihar',
            'treat5_female_bihar',
            'treat1_female',
            'treat2_female',
            'treat3_female',
            'treat4_female',
            'treat5_female',
            'treat1_bihar',
            'treat2_bihar',
            'treat3_bihar',
            'treat4_bihar',
            'treat5_bihar',
            'female_bihar',
            'treat1_bihar_longdist',
            'treat2_bihar_longdist',
            'treat3_bihar_longdist',
            'treat4_bihar_longdist',
            'treat1_female_longdist',
            'treat2_female_longdist',
            'treat3_female_longdist',
            'treat4_female_longdist',
            'female_bihar_longdist',
            'pca_asset',
            'pca_ses',
            'treat1_female_bihar_pca_asset',
            'treat1_female_pca_asset',
            'female_bihar_pca_asset',
            'treat1_bihar_pca_asset',
            'treat1_pca_asset',
            'female_pca_asset',
            'bihar_pca_asset',
            'treat1_female_bihar_pca_ses',
            'treat1_female_pca_ses',
            'female_bihar_pca_ses',
            'treat1_bihar_pca_ses',
            'treat1_pca_ses',
            'female_pca_ses',
            'bihar_pca_ses',
            'treat1_female_bihar_muslim',
            'treat1_female_muslim',
            'female_bihar_muslim',
            'treat1_bihar_muslim',
            'treat1_muslim',
            'female_muslim',
            'bihar_muslim'
        ]
    ] 

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


