# -*- coding: utf-8 -*-

# %%import
import os
import re
import glob
import pickle
import pyodbc
import pandas as pd
from datetime import datetime
#import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

import cx_Oracle
from dateutil.relativedelta import relativedelta

from pandas.tseries.offsets import MonthEnd

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 35)
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option('display.expand_frame_repr', False)
#pyodbc.drivers()
mypath="/data/corporateusername/data/lapse/"

from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import KMeans

from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import OneHotEncoder

# %%loading functions
def load_magquesp3():
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/decisionenginep3.csv', sep='\t')
    _t.columns = _t.columns.str.lower()
    _t.created_date = pd.to_datetime(_t.created_date)
    _t.rename(columns={'client_id':'lifeseq'}, inplace=1)
    _t['key'] = _t.proposal_number.astype(int).astype(str) + _t.lifeseq.astype(int).astype(str)
    return _t

def load_nbcus():
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/nbcus.csv', sep='\t')
    _t.columns = _t.columns.str.lower()
    _t.dob = pd.to_datetime(_t.dob, errors = 'coerce')
    return _t

def load_nbprop():
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/nbprop.csv', sep='\t')
    _t.columns = _t.columns.str.lower()
    cols = ['po_id','proposal_date','updated_date','proposal_id','proposal_num','policy_num',
                                'agent_code','frequence','is_po_la','office_code','basic_component_id',
                                'total_premium','ape']
    _t = _t[cols]
    _t.proposal_date = pd.to_datetime(_t.proposal_date)
    _t.proposal_num = _t.proposal_num.astype(str).str.zfill(9).astype(str)
    return _t

def load_nbpo():
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/nbpo.csv', sep='\t')
    _t.columns = _t.columns.str.lower()
    return _t

def load_nbla():
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/nbla.csv', sep='\t')
    _t.columns = _t.columns.str.lower()
    _t.dropna(subset=['life_seq'],inplace=True)
    _t.life_seq = _t.life_seq.astype(int)
    
    return _t

def load_claim():
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/claim.csv',sep='\t')
    _t.columns = _t.columns.str.lower()
    _t.columns = ['polno', 'agentnumber', 'la_num', 'po_num', 'claimstatus', 'claimtype',
       'month_diff', 'accident', 'faceamount', 'totalamount','eventdate']
    
    _t.eventdate = pd.to_datetime(_t.eventdate)
    _t.dropna(subset=['polno','eventdate'], inplace=True)
    
    _p = pd.read_csv(cf.DATA_PATH + '/feature/policy_info/policy_info.csv', sep='\t')
    _p.columns = _p.columns.str.lower()
    _p.issue_date = pd.to_datetime(_p.issue_date)
    
    _t = pd.merge(_t, _p[['policyid','issue_date']], how='left',
                  left_on='polno',right_on='policyid')
    _t['cal_month_diff'] = ((_t.eventdate - _t.issue_date)/np.timedelta64(1, 'M')).apply(np.ceil)
    _t.month_diff = np.where(_t.month_diff.isnull()==False,_t.month_diff,
                             _t.cal_month_diff)
    _t.dropna(subset=['month_diff'], inplace=True)
    
    _t['claim12'] = np.where((_t.month_diff>=0)&(_t.month_diff<=12),1,0)
    _t['claim24'] = np.where((_t.month_diff>=0)&(_t.month_diff<=24),1,0)
    _t['ds_claimtype'] = np.where(_t.claimtype.str.contains('|'.join(['MED', 'CIL'])), 'medci','life')
    _t['label1'] = np.where((_t.claim24==1)&(_t.claimtype.str.contains('|'.join(['MED', 'CIL']))), 1,0)
    _t['label2'] = np.where((_t.claim24==1)&
                            (_t.claimtype.str.contains('|'.join(['MED', 'CIL'])))&
                            (_t.claimtype.str.contains('TPD')==False), 1,0)
    _t['label3'] = np.where((_t.claim24==1)&(_t.claimtype=='DTH '), 1,0)
    _t['label4'] = np.where((_t.claim24==1)&(_t.claimtype!='DTH '), 1,0)
    
    return _t

def load_nbcomp():
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/nbcomp.csv', sep='\t')
    _t.columns = _t.columns.str.lower()
    _t = _t[['component_id','component_code','sum_assured','la_info_id',
             'component_uw_decision','component_status','nb_component_status',
             'policy_term','premium_term']]
    
    return _t

def load_me():
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/me.csv', sep='\t')
    _t.columns = _t.columns.str.lower()
    
    _t.me_it_date = pd.to_datetime(_t.me_it_date)
    return _t


    
# %%building features df functions
def make_train():
    #proposal base
    prop = load_nbprop()
    prop = prop[(prop.updated_date.isnull()==False)]
    prop.sort_values('proposal_id',inplace=True)
    prop = prop.drop_duplicates('proposal_num',keep='last')
    
    nbla = load_nbla()
    df = pd.merge(prop,nbla, on='proposal_id',how='left')
    df.dropna(subset=['life_seq'],inplace=True)
    
    nbcus = load_nbcus()
    nbpo = load_nbpo()
    nbcus['in_nbpo'] = nbcus.customer_id.isin(nbpo.customer_id).astype(int)
    nbcus['is_po'] = nbcus.client_code.isin(nbcus[nbcus.in_nbpo==1]['client_code']).astype(int)
    nbcus.drop(columns=['in_nbpo'],inplace=True)
    
    df = df.merge(nbcus, on='customer_id', how='left')

    df['key'] = df.proposal_num.astype(str) + df.life_seq.astype(int).astype(str)
    df['ym'] = df['proposal_date'].dt.strftime('%Y%m').astype(int)
    return df


def add_sumassured(df):
    _t = df[['key','proposal_num','la_info_id','customer_id','life_seq','po_id','basic_component_id']]
    
    comp = load_nbcomp()
    _t = _t.merge(comp[['la_info_id','sum_assured']],how='left', on='la_info_id')
    
    _t = _t.rename(columns={'sum_assured':'riderSA'})
    _t = _t.groupby(['key','basic_component_id','life_seq'])['riderSA'].sum().reset_index()
    
    
    _t = _t.merge(comp[['component_id','sum_assured']], how='left'
                  , left_on='basic_component_id',right_on = 'component_id')
    _t = _t.rename(columns={'sum_assured':'basicSA'})
    _t.basicSA = np.where(_t.life_seq==1,_t.basicSA,0)
    
    
    _t['totalSA'] = _t.riderSA + _t.basicSA
    
    df = pd.merge(df, _t[['key','totalSA']], on='key',how='left')
    
    return df

def add_decisionengine_alcohol(df, decisionenginep3):
    mn = decisionenginep3
    mn = mn[mn.key.isin(df.key)]
    
    consumpdict = {'1-2':1.5,'3-4':3.5,'5-6':5.5,'7-10':8.5,'>10':20}
    qs = ['ConsumptionAmount','ConsumptionDaysPerWeek']
    searchstr = '|'.join(qs)
    mn = mn[mn.question_locator.str.contains(searchstr)]
    _t = pd.merge(pd.DataFrame(mn['key'].unique(),columns=['key'])
                  , mn[mn.question_locator.str.contains('Week')][['key','question_locator','answer']])
    _t = pd.merge(_t, mn[mn.question_locator.str.contains('Amount')][['key','question_locator','answer']],on='key',how='left')
    _t['adj_ca'] = _t.answer_y.map(consumpdict)
    _t['daily_alcohol_consumption'] = _t.answer_x.astype(int)*_t.adj_ca
    
    df = df.merge(_t[['key','daily_alcohol_consumption']],how='left',on='key')
    return df

def add_decisionengine_tobacco(df, decisionenginep3):
    mn = decisionenginep3
    mn = mn[mn.key.isin(df.key)]
    
    mn = mn[mn.question_locator.str.contains('Tobacco')]
    _t = pd.merge(pd.DataFrame(mn['key'].unique(),columns=['key'])
                  , mn[mn.question_locator.str.contains('TobaccoUse')][['key','question_code']])
    
    _t = pd.merge(_t, mn[mn.question_locator.str.contains('CigarettesPerDay')][['key','answer']],on='key',how='left')
    
    _t.columns = ['key','l12m_smoked','avg_daily_cig']
    df = df.merge(_t,how='left',on='key')
    return df

def add_decisionengine_impairments(df, decisionenginep3):
    mn = decisionenginep3
    mn = mn[mn.key.isin(df.key)]
    
    mn = mn[(mn.key.isin(mn[mn.question_locator.str.contains('mpairmentsGroup2')]['key'])==False)&
   (mn.question_locator.str.contains('.type'))][['key','question_locator','answer']]
    mn
    _t = mn.groupby('key')['answer'].count().reset_index()
    _t.columns = ['key','impairments_count']
    df = df.merge(_t,how='left',on='key')
    df.impairments_count.fillna(0, inplace=True)
    
    return df

def add_bodybuild(df):
    
    df = df[df.dob.dt.year > 1899]
    df['attended_age'] = ((df.proposal_date - df.dob).dt.days/365).apply(np.floor)
    #df['is_juvenile'] = np.where[df.attended_age<=2]
    df['bmi'] = df['weight']/(df['height']/100)**2
    
    return df


def add_agsale_polbad(df):
    p = pd.read_csv(cf.DATA_PATH + '/feature/policy_info/policy_info.csv', sep = '\t')
    p = p[['issue_date','agentid_agsale','agsale_pol_bad']]
    p['ym'] = pd.to_datetime(p.issue_date).dt.strftime('%Y%m')
    p = p.sort_values('issue_date')
    p = p.drop_duplicates(subset=['agentid_agsale','ym'],keep='last')
    p.rename(columns={'agentid_agsale':'agent_code'},inplace=True)
    p = p[['agent_code','ym','agsale_pol_bad']]
    
    df['ym'] = df.proposal_date.dt.strftime('%Y%m')
    
    df = df.merge(p, how='left',on=['agent_code','ym'])
    
    return df

def gen_ag_risk(df):
    claim_tbl = load_claim()
    claim_tbl.dropna(subset=['eventdate'],inplace=True)
    claim_tbl['ym'] = claim_tbl.eventdate.dt.strftime('%Y%m').astype(int)
    
    df = df[['agent_code', 'ym']]
    df.ym = df.ym.astype(int)
    df.sort_values('ym', inplace=True)
    ymlist = list(df.ym.unique())
    
    for i in ymlist:
        _c = claim_tbl[claim_tbl.ym==i]
        _c = _c.groupby('agentnumber')['la_num'].count().reset_index()
        _c.columns = ['agent_code','claims']
        _c.to_csv(cf.DATA_PATH + '/feature/agsale_risk_score_monthly/agsale_risk_{}.csv'.format(i), index=False, sep='\t')
        
    return

def add_ag_risk(df):
    _df = df[['ym', 'agent_code']]
    _df.sort_values('ym', inplace=True)
    l = []
    for i in list(_df.ym.unique()):
        _df2 = _df[_df.ym==i]
        _s = pd.read_csv(cf.DATA_PATH + '/feature/agsale_risk_score_monthly/agsale_risk_{}.csv'.format(i), sep='\t')
        _s.columns=['agent_code','agsale_risk_score']
        _df2 = _df2.merge(_s, how='left', on='agent_code')
        l.append(_df2)
        
    _df = pd.concat(l)
    _df.drop_duplicates(inplace=True)
    df = pd.merge(df, _df, on=['ym', 'agent_code'], how='left')
    return df

def create_master_component(df):
    df = df[['key','frequence','totalSA','la_info_id','basic_component_id']]
    p = pd.read_csv(cf.DATA_PATH + '/feature/product/prodindex.csv')
    p = p[p.contracttype.isnull()==False]
    
    pmktg = pd.read_csv(cf.DATA_PATH + '/_temp/crtable_map_v2.csv')
    pmktg.rename(columns={'Unnamed: 3':'pmkt_component_type'}, inplace=True)
    pmktg = pmktg[['CRTABLE', 'pmkt_component_type']]
    
    comp = load_nbcomp()
    _t1 = df.merge(comp[['la_info_id','component_code','sum_assured','premium_term']], on='la_info_id')
    _t2 = df.merge(comp[['component_id','sum_assured','component_code','premium_term']],
                   left_on='basic_component_id',right_on = 'component_id')
   
    _t1.drop(columns=['totalSA','basic_component_id'],inplace=True)
    _t2.drop(columns=['totalSA','basic_component_id','component_id'],inplace=True)
    _t = pd.concat([_t1, _t2])
    _t.key = _t.key.astype(int)
    _t = _t.sort_values('key')
    _t.drop_duplicates(inplace=True)
    
    _t = pd.merge(_t, p[['contracttype', 'productcode']], how='left', left_on='component_code',right_on='contracttype')
    _t.component_code = np.where(_t.productcode.isnull(), _t.component_code, _t.productcode)    
    _t = pd.merge(_t, pmktg, left_on = 'component_code', right_on='CRTABLE',how='left')
    _t.dropna(subset={'component_code'}, inplace=True)
    return _t

def add_uab(df):
    xls = pd.ExcelFile(cf.DATA_PATH + '/_temp/UAB Calculation_Tool.xlsx', engine='openpyxl')
    p = pd.read_csv(cf.DATA_PATH + '/_temp/prodindex.csv')
    factor_1 = pd.read_excel(xls, 'Factor_g1')
    factor_1 = factor_1.iloc[:,:3].dropna()
    factor_2 = pd.read_excel(xls, 'Factor_g2').dropna()
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import PolynomialFeatures
    '''
    Add more benefit term to Factor1
    '''
    X = factor_1['Benefit term']
    y1 = factor_1.Factor_POW7
    y2 = factor_1.Factor_others
    poly_features = PolynomialFeatures(degree=3)
    # transform the features to higher degree features.
    X_train_poly = poly_features.fit_transform(X.values.reshape(-1, 1))
    lg_model1 = LinearRegression()
    lg_model1.fit(X_train_poly, y1)
    lg_model2 = LinearRegression()
    lg_model2.fit(X_train_poly, y2)
    X_new = pd.Series(list(range(48,78,1))).values.reshape(-1, 1)
    Factor_POW7_new = lg_model1.predict(poly_features.fit_transform(X_new))
    Factor_others_new = lg_model2.predict(poly_features.fit_transform(X_new))
    
    factor_1 = pd.concat([factor_1,pd.DataFrame({"Benefit term":pd.Series(list(range(48,78,1))),
                  "Factor_POW7":Factor_POW7_new,
                  "Factor_others":Factor_others_new})])
    
    _f1 = pd.merge(p[['contracttype','productcode']], factor_2, left_on='contracttype', right_on='Code')
    _f1 = _f1[['productcode','Factor_2']]
    _f1.columns = ['Code', 'Factor_2']
    _f2 = pd.merge(p[['contracttype','productcode']], factor_2, left_on='productcode', right_on='Code')
    _f2 = _f2[['contracttype','Factor_2']]
    _f2.columns = ['Code', 'Factor_2']
    factor_2 = pd.concat([factor_2, _f1, _f2])
    factor_2.drop_duplicates(inplace=True)
    '''
    tính UAB theo công thức trong UAB Calculation tool
    '''
    _df = df[['key','frequence','totalSA','la_info_id','basic_component_id']]
    
    _t = create_master_component(_df)
    
    df_uab = _t.copy()
    
    # factor define is special component code or not
    df_uab['SPECIAL_COMPONENT_CODE'] = np.where(df_uab.component_code.str.contains(r'(OPW[0-9])|(UCW[0-9])|(DSR[0-9])')
                            ,df_uab.premium_term
                            ,0   )
    df_uab = df_uab.merge(factor_1,how='left', left_on=['SPECIAL_COMPONENT_CODE'],right_on=['Benefit term'])
    df_uab = df_uab.merge(factor_2,how='left', left_on=['component_code'],right_on=['Code'])
    df_uab['group1'] = np.where(df_uab.component_code=='OPW7'
                            ,df_uab.Factor_POW7
                            ,df_uab.Factor_others)#['Benefit term'])
    #  calculate manually UAB
    df_uab['Manually_UAB'] = np.where(df_uab.component_code.str.contains(r'(OPW[0-9])|(UCW[0-9])|(DSR[0-9])')
                            ,df_uab.group1*df_uab.frequence*df_uab.sum_assured
                            ,df_uab.Factor_2*df_uab.sum_assured   )
    #df_uab.fillna(0,inplace=True)
    #df_uab
    df_uab = df_uab.groupby('key')['Manually_UAB'].sum().reset_index()
    df = pd.merge(df, df_uab, how='left', on='key')
    return df



def create_check(test):    
    result = pd.concat([pd.DataFrame(model.predict_proba(test.iloc[:, _a:]),
                         columns=["nonrisk_pct", "risk_pct"]), test.iloc[:, :_a]], axis=1)
    result = result.sort_values("risk_pct").reset_index(drop=True)
    #result.loc[result["lapse"] == 0, "CBILLAMT"] = 0
    #result = pd.merge(result, train[["policyid", "ptrneff", "channel"]],
     #                  on=["policyid", "PTRNEFF"], how="left")

    #result = pd.merge(result, df[["policyid", "PTRNEFF", "have_history"]],
    #                   on=["policyid", "PTRNEFF"], how="left")
    #result = result[result["have_history"] == False]
    #result = result.reset_index(drop=True)

    _bins = list(np.linspace(0, len(result) -1, 11).astype(int))
    _groups = np.arange(1, 11, 1)
    result["decile"] = pd.cut(result.index, bins=_bins, labels=_groups,
                              include_lowest=True)

    check = result.groupby("decile")["label1"].value_counts().unstack().reset_index()
    check["total"] = check[[0, 1]].sum(axis=1)

    check['decile_riskrate'] = check[1]/check['total']
    #_r = result.groupby("decile").agg({"risk_pct":[np.min, np.max]}).reset_index(drop=1)
    _r = result.groupby("decile",as_index=0).agg({"risk_pct":[np.min, np.max]}).reset_index(drop=1)
    _r.columns = _r.columns.droplevel()
    _r.columns = ['decile','proba_min','proba_max']
    check = check.merge(_r, on='decile')
    check = check.sort_values("decile", ascending=False)
    #check_ape = result.groupby("sample")["CBILLAMT"].sum().reset_index()
    #check = pd.merge(check, check_ape, on="sample")
    check["cs_total"] = check["total"].cumsum()
    check["cs_risk"] = check[1].cumsum()
    #check["cs_ape"] = check["CBILLAMT"].cumsum()
    check["pct_total"] = check["cs_total"] / check["total"].sum()
    check["pct_risk_covered"] = check["cs_risk"] / check[1].sum()
    #check["pct_ape"] = check["cs_ape"] / check["CBILLAMT"].sum()
    check["cs_risk_rate"] = check["cs_risk"] / check["cs_total"]
    check["shift_total"] = check.total.sum() - check.cs_total.shift(1)
    check["shift_total"].fillna(0,inplace=True)
    check["shift_risk"] = check[1].sum() - check.cs_risk.shift(1)
    check["shift_risk"].fillna(0,inplace=True)
    _x = check[1].sum()/check["total"].sum()
    check["risk_if_xpress"] = check["shift_risk"].div(check["shift_total"])
    check.risk_if_xpress.fillna(_x,inplace=True)
    check.drop(columns=['cs_total','cs_risk','pct_total'
                        ,'shift_total','shift_risk'],inplace=True)
    check = check.reset_index(drop=1)
    check = pd.concat([check,
                        pd.DataFrame(['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%']).rename(columns={0:'target_xpress_rate'})],
                      axis=1)
    check = check[['decile',0,1,'total','cs_risk_rate','target_xpress_rate','risk_if_xpress',
                    'pct_risk_covered','decile_riskrate','proba_min','proba_max']]
    
    return check

    
def old_decisionengine_outcome():

    magpol = pd.read_csv(cf.DATA_PATH + '/_temp/magpol.csv', sep='\t')
    magpol.underwriten_date = pd.to_datetime(magpol.underwriten_date)
    
    lanum_cols = ['proposal_num','underwriten_date'] + [i for i in magpol.columns if 'client_num' in i]
    magpol.sort_values('underwriten_date', inplace=True)
    magpol.drop_duplicates(subset=['proposal_num', 'decision_code'], keep='last',inplace=True)
    
    mcore = magpol[lanum_cols].drop_duplicates(subset=lanum_cols).melt(id_vars=['proposal_num',
                                                                                'underwriten_date'])
    mcore = mcore[mcore.value!=' ']
    
    mcore['key'] = mcore.proposal_num.astype(int).astype(str) + mcore.variable.str[-1:].astype(str)
    mcore = mcore.sort_values('key')
    
    mcodes = magpol[magpol['decision_data'].isin([np.nan])==False][['proposal_num','decision_code','decision_data']]
    mcodes = mcodes[['proposal_num','decision_code','decision_data']]
    mcodes['key'] = mcodes.proposal_num.astype(int).astype(str) + mcodes.decision_code.str[:1].astype(str)
    _m = mcodes.groupby(['proposal_num','key'])['decision_data'].apply(','.join).reset_index()
    
    _m.columns = ['proposal_num','key','old_decisionengine_result']
    _m.key = _m.key.astype(str)
    
    mcore = pd.merge(mcore, _m[['key','old_decisionengine_result']], on='key')
    mcore.drop(columns='variable',inplace=True)
    mcore.value = mcore.value.astype(str).str.zfill(8)
    mcore.old_decisionengine_result = mcore.old_decisionengine_result.str.strip()
    mcore.columns = ['proposal_num', 'date', 'clientcode', 'key',
       'old_decisionengine_result']
    mcore['old_decisionengine_xpress'] = np.where(mcore.old_decisionengine_result.str.contains('Standard'),1,0)
    mcore['old_decisionengine_refer'] = np.where(mcore.old_decisionengine_result.str.contains('Refer'),1,0)
    mcore['old_decisionengine_decline'] = np.where(mcore.old_decisionengine_result.str.contains('Decline'),1,0)

    return mcore

def old_uw_policy_decision():
    uwdec = pd.read_csv(cf.DATA_PATH + '/raw/atmupf/atmupf.csv', sep='\t')
    uwdec.columns = uwdec.columns.str.lower()
    uwdec.datime = pd.to_datetime(uwdec.datime)
    uwdec['old_decisionengine_uwdec'] = np.where(uwdec.xpressdcs.isin(['RAT', 'EAL', 'RHS', 'REX', 'RER', 'RCI']),1,0)
    
    return uwdec

def load_decisionengine_p3_outcomes():
    
    decisionengine_la1 = pd.read_csv(cf.DATA_PATH + '/_temp/rawdecisionengine.csv', sep='\t')
    
    decisionengine_la1.columns = decisionengine_la1.columns.str.lower()
    decisionengine_la1.created_date = pd.to_datetime(decisionengine_la1.created_date)
    decisionengine_la1['key'] = decisionengine_la1['proposal_number'].astype(int).astype(str) + \
        decisionengine_la1['client_id'].astype(int).astype(str)
    decisionengine_la1['no_code'] = np.where((decisionengine_la1.supporting_code=='No supporting code')&\
                                 (decisionengine_la1.requirement_code.isnull())&\
                                     (decisionengine_la1.me_code.isnull())&
                                     #(decisionengine_la1.decisionengine_grouped_outcome=='Nonxpress')&\
                                         (decisionengine_la1.is_code.isnull())&\
                                             decisionengine_la1.adjustment.isnull(),1,0)                                                                                 

   
    return decisionengine_la1

def add_productline_risk_score(mdf):
    #build product component string on CLAIM side
    list_of_files = glob.glob(cf.DATA_PATH + '/raw/covrpf_snapshot/*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)
    covr = pd.read_csv(latest_file, sep='\t')
    covr.columns = covr.columns.str.lower()
    covr = covr[['policyid','statcode','life','crtable']]
    
    life = pd.read_csv(cf.DATA_PATH + '/raw/lifepf/lifepf.csv', sep='\t')
    life.columns = life.columns.str.lower()
    
    covr = covr.merge(life, on=['policyid','life'])
    covr = covr.groupby('lifcnum')['crtable'].apply(','.join).reset_index()
    covr.sort_values('crtable',inplace=True)
    
    claim = load_claim()
    claim = claim.merge(covr, how='left',left_on='la_num',right_on='lifcnum')
    
    #build product component string on PROPOSAL side
    _t = create_master_component(mdf)
    _t.sort_values('component_code',inplace=True)
    cp = _t.groupby('key')['component_code'].apply(','.join).reset_index()
    mdf.key = mdf.key.astype(int)
    mdf = mdf.merge(cp, how='left',on='key')
    
    #add risk score
    claim['ym'] = claim.eventdate.dt.strftime('%Y%m')
    dfl = []
    mdf.ym = mdf.ym.astype(int)
    claim['ym'] = claim['ym'].astype(int)
    for i in mdf.ym.unique():
        _df = mdf[mdf.ym==i]
        _cl = claim[claim['ym']<i]
        _cl = _cl.groupby('crtable')['label1'].sum().reset_index()
        _cl.columns=['crtable', 'productline_risk_score']
        _df = _df.merge(_cl, left_on='component_code', right_on='crtable',how='left')
        dfl.append(_df)
    
    mdf = pd.concat(dfl)
    mdf.productline_risk_score = mdf.productline_risk_score.fillna(0)
    
    return mdf

def add_job_risk_score(mdf):
    client = pd.read_csv(cf.DATA_PATH + '/feature/client_info/client_info.csv', sep='\t')
    client.columns=client.columns.str.lower()
    client = client[['clntnum','cleaned_occpcode']]
    client = client[client.clntnum.astype(str).str.isnumeric()]
    client['clntnum'] = client['clntnum'].astype(int)
    
    claim = load_claim()
    claim = claim.merge(client, how='left',left_on = 'la_num',right_on='clntnum')
    
    mdf = mdf.merge(client[['clntnum','cleaned_occpcode']], how='left', left_on='client_code',
                    right_on = 'clntnum')
    mdf["cleaned_occpcode"] = mdf["cleaned_occpcode"].fillna(0)
    
    dfl = []
    claim['ym'] = claim.eventdate.dt.strftime('%Y%m').astype(int)
    for i in mdf.ym.unique():
        _df = mdf[mdf.ym==i]
    #    _js = claim[(claim['ym']<i)&(claim.occpcode==' ')]
        _js = claim[(claim['ym']<i)]
        _js = _js.groupby('cleaned_occpcode')['label1'].sum().reset_index()
        _js.columns=['cleaned_occpcode', 'job_historical_riskscore']
        _df = _df.merge(_js, on='cleaned_occpcode',how='left')
        dfl.append(_df)
    
    mdf = pd.concat(dfl)
    return mdf


def add_label(df):
    #nbla = load_nbla()
    #nbla = nbla[nbla.proposal_id.isin(df.proposal_id)]
    cl = load_claim()
    _m1 = load_decisionengine_p3_outcomes()
    _m1 = _m1[['key', 'final_decision']]
    _m1.columns = ['key', 'p3_decisionengine_decision']
    om = old_decisionengine_outcome()
    om = om[['key', 'old_decisionengine_result','old_decisionengine_xpress', 'old_decisionengine_refer', 'old_decisionengine_decline']]
    mp = create_master_component(df)
    
    df['earlyclaim'] = df['client_code'].isin(cl[(cl.month_diff<=24)&(cl.month_diff>=0)]['la_num']).astype(int)
    df['uwrisk']     = df.uw_decision.isin(['CA','PP','PP1','DC','DC1','PP2','PP3','ME/IS']).astype(int)
    #df['label'] = np.where((df.earlyclaim + df.uwrisk)>0,1,0)
    _t = pd.read_csv(cf.DATA_PATH + '/raw/nb_proposal/me.csv', sep='\t')
    _t.columns = _t.columns.str.lower()
    
    df['me_requested'] = np.where(df.la_info_id.isin(_t[_t.status!='Inactive']\
                                                                 ['la_info_id']),1,0)
        
    _mp = mp.drop_duplicates(subset=['key','pmkt_component_type'])
    _mp = _mp.dropna(subset=['pmkt_component_type'])
    _mp = _mp.groupby('key')['pmkt_component_type'].apply(','.join).reset_index()
    _mp['ds_ps_ismedci'] = np.where(_mp.pmkt_component_type
                                    .str.contains('|'.join(['CI', 'Health'])),1,0)
    
    cl.rename(columns={'la_num':'client_code'}, inplace=True)
    lb = cl.groupby('client_code')['label1','label2','label3','label4'].sum().reset_index()
    lb[['label1','label2','label3','label4']] = (lb[['label1','label2','label3','label4']]>0).astype(int)
    
    df = pd.merge(df, _m1, how='left', on='key')
    df = pd.merge(df, om, how='left', on='key')
    df = pd.merge(df, _mp, how='left', on='key')
    df = pd.merge(df, lb, how='left',on='client_code')
    
    
    return df

def dummy_encode(data, column, load=False, result_dir=None):
    if load:
        enc = pickle.load(open(
            result_dir + "/enc_{}.pkl".format(column), "rb"))
    else:
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(data[[column]])
    
    oh = pd.DataFrame(enc.transform(data[[column]]).toarray())
    oh = oh.add_prefix("{}_".format(column))
    
    data = pd.concat([data, oh], axis=1)
    data = data.drop(column, axis=1)
    return data, enc



if '__main__' == __name__:
# still have some leftover processing need to be refactor
# %%BUILD TRAIN
    mdf = make_train()
    
    mdf = add_sumassured(mdf)
    mq3 = load_magquesp3()
    mdf = add_decisionengine_alcohol(mdf, mq3)
    mdf = add_decisionengine_tobacco(mdf, mq3)
    mdf = add_decisionengine_impairments(mdf, mq3)
    mdf = add_bodybuild(mdf)
    mdf = add_label(mdf)
    mdf = add_agsale_polbad(mdf)
    mdf = add_ag_risk(mdf)
    mdf = add_uab(mdf)
    mdf = add_productline_risk_score(mdf)
    mdf = add_job_risk_score(mdf)
    # %%PROCESSING
    tcol = ['proposal_date','uw_decision','key','label1',
            'frequence', 'total_premium', 'ape', 'life_seq',
           'height', 'weight', 'average_income',
           #'uw_decision', 'uw_decision_basic',
           'gender',
           'married_status', 'is_po', 'ym',
           'totalSA', 'daily_alcohol_consumption', 'l12m_smoked', 'avg_daily_cig',
           'impairments_count', 'attended_age', 'bmi', 'ds_ps_ismedci', 
           'agsale_pol_bad', 'agsale_risk_score', 'Manually_UAB',
           'productline_risk_score','job_historical_riskscore',
           #'exposure', 'sector_risk'
           ]
    
    df = mdf[tcol]
    #df.info(null_counts=1)
    
    df.married_status = np.where(mdf.married_status.isin(['',' ','U','DD']),'unknown',mdf.married_status)
    df.married_status.fillna('unknown',inplace=True)
    
    df[['daily_alcohol_consumption','l12m_smoked','avg_daily_cig']] = df[['daily_alcohol_consumption','l12m_smoked','avg_daily_cig']].fillna(0)
    df.l12m_smoked = np.where(df.l12m_smoked=='Yes', 1,0)
    df.avg_daily_cig = df.avg_daily_cig.astype(int)
    
    #mdf.columns
    #df = pd.concat([df, mdf[['agsale_pol_bad']]],axis=1)
    df[['label1','productline_risk_score','job_historical_riskscore'
        ,'exposure', 'sector_risk']] = df[['label1','productline_risk_score','job_historical_riskscore',
           'exposure', 'sector_risk']].fillna(0)
    df.agsale_pol_bad.fillna(0, inplace=True)
    df.average_income.fillna(0, inplace=True)
    df.agsale_risk_score.fillna(0, inplace=True)
    
    df.uw_decision.fillna('xpress', inplace=True)
    #df['channel'] = np.where((df.agent_code >= 60000000)&(df.agent_code <))
    
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    
    
    fdf=df.copy()
    
    # %%DUMMY ENCODING
    dcols = ['frequence','is_po','gender','married_status','ym']
    for _col in dcols:
        fdf[_col] = fdf[_col].astype(str)
        fdf, _ = dummy_encode(fdf, _col)
    
    fdf.info(null_counts=1)
    
    # %%out features df
    fdf.to_csv(cf.DATA_PATH + '/_temp/smartuw_features.csv', sep='\t')
    
    # %%TRAIN & TEST SET VARIATIONS
    #==================================================#
    #1 YEAR LABEL
    fdf = pd.read_csv(cf.DATA_PATH + '/_temp/smartuw_features.csv', sep='\t')
    # fdf.proposal_date = pd.to_datetime(fdf.proposal_date)
    # mdf = pd.read_csv(cf.DATA_PATH + '/_temp/smartuw_df2.csv',sep='\t')
    # fdf = fdf.merge(mdf[['key','client_code']],how='left',on='key')
    # claim = load_claim()
    # fdf['label1'] = np.where(fdf.client_code.isin(\
    #                                               claim[(claim.claim12==1)&(claim.claimtype.str.contains('|'.join(['MED', 'CIL'])))]['la_num'])\
    #                          ,1,0)
    # fdf.drop(columns='client_code',inplace=True)
    # fdf.dropna(inplace=True)
        
    # train =fdf[(fdf.proposal_date < datetime(2020,1,1))&(fdf.proposal_date >= datetime(2017,1,1))] 
    # train.reset_index(inplace=True,drop=True)       
    # test = fdf[(fdf.proposal_date >= datetime(2020,1,1))&(fdf.proposal_date < datetime(2020,6,1))]
    # test.reset_index(inplace=True, drop=True)
    #==================================================#
    #ORIGINAL TRAIN TEST 201901 - 201905
    
    #fdf.label.value_counts(normalize=1)
    fdf.proposal_date = pd.to_datetime(fdf.proposal_date)
    
    train =fdf[(fdf.proposal_date<datetime(2019,1,1))&(fdf.proposal_date>=datetime(2017,1,1))] #(648406, 25)
    train.reset_index(inplace=True,drop=True)       
    test = fdf[(fdf.proposal_date>=datetime(2019,1,1))&(fdf.proposal_date<datetime(2019,6,1))] #(343504, 25)
    test.reset_index(inplace=True, drop=True)
    

    
   
    # %%MODEL FITTING
    _a = train.columns.get_loc('label1')+1
    model = RandomForestClassifier(min_samples_leaf=5, n_estimators=300, n_jobs=3)
    
    X = train.iloc[:,_a:]
    
    Y = train['label1']
    
    model.fit(X, Y)
    
    with open(cf.DATA_PATH + '/model/model_smartuw/smart_uw.pkl', 'wb') as f:
        pickle.dump(model, f)
    #fdf.columns
    # %%load ORIGINAL trained model

    model = pickle.load(open(cf.DATA_PATH + '/model/model_smartuw/smart_uw.pkl', 'rb'))
    
    
    # %%DECILES
    cc = create_check(test)
    
    
    # %%Cross validation scores
    from sklearn.model_selection import cross_val_score as cvs
    scores = cvs(model, X, Y, cv=10)
    scores_f1 = cvs(model, X, Y, cv=10, scoring='f1')
    scores
    scores
    # Out[224]: 
    # array([0.84410543, 0.84392971, 0.84407348, 0.84455272, 0.84432659,
    #        0.8437515 , 0.84389527, 0.8436237 , 0.84370357, 0.84373552])
    
    scores_f1
    scores.mean()
    
    r2 = result[['risk_pct','nonrisk_pct','label2']]
    r2['agar'] = test['ag_allrisk']
    r2[r2.agar<100].agar.hist()
    sns.boxplot(data=r2[r2.agar<=30],y='agar',x='label2')
    sns.boxplot(data=r2[r2.agar>100],y='agar',x='label2')
    r2.groupby('label2')['risk_pct'].describe()
    
    
    # %%OOT confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix(test['label2'], model.predict(test.iloc[:,_a:]), normalize='true')
    confusion_matrix(test['label2'], model.predict(test.iloc[:,_a:]), normalize='pred')
    
    confusion_matrix(test['label2'], model.predict(test.iloc[:,_a:]))
    
    unique, counts = np.unique(model.predict(test.iloc[:,_a:]), return_counts=True)
    dict(zip(unique, counts))
    
    # %%feature importantce
    importance = model.feature_importances_.tolist()
    importance = pd.DataFrame({"feature": train.iloc[:,_a:].columns.tolist(),
                         "importance":importance})
    importance = importance.sort_values("importance", ascending=False).head(20)
    plt.figure(figsize=(7, 7))
    plt.gcf().set_facecolor("white")
    sns.barplot(y=importance["feature"], x=importance["importance"])
    plt.yticks(size=9)
    
