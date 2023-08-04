# -*- coding: utf-8 -*-


#import packages from shared internal library
from company.core import config as cf
from company.core import file_utils
from company.core.db import mssql

#import from external libraries
import pandas as pd
from datetime import datetime
from datetime import date
import numpy as np
import pyodbc
from dateutil.relativedelta import relativedelta
import sys
import pyodbc
import cx_Oracle

#%%declare columns
covr_snapshot_cols = ["a.policyid","...","b.DATIME"]

covr_update_cols = ["policyid","...","DATIME"]

#%% functions

#oracle server connection
def oracle_conn():
    dsn_tns = cx_Oracle.makedsn('10.xxx.xxx.100', '1521', service_name='repprd')
    conn = cx_Oracle.connect(user=r'analyticuser08', password='xxx', dsn=dsn_tns)
    
    return conn

#get master contract and contract components
def get_latest_covr():
    query = """
     select {}
     from ...
    """.format(", ".join(covr_snapshot_cols))
    pf = pd.read_sql(query,con = oracle_conn())
    #pf = pd.read_sql(query.replace('\n','').strip(), con=conn)
    
    month = (datetime.now() - relativedelta(months = 1)).strftime("%Y%m")
    pf.to_csv(cf.DATA_PATH + '/raw/covrpf_snapshot/covrpf_{}.csv'\
              .format(month), sep='\t',index=False)
    
#
def gen_product_info():
    month = (datetime.now() - relativedelta(months=1)).strftime("%Y%m")
    cov = pd.read_csv(cf.DATA_PATH + '/raw/.../covrpf_{}.csv'.format(month), sep='\t')
    pindex = pd.read_csv(cf.DATA_PATH + '/feature/product_info/prodindex.csv')
    
    print("redacted")
   
    cov.to_csv(cf.DATA_PATH + '/feature/product_info/product_info_{}.csv'.format(month), sep='\t', index=False)   

def get_update_covr(start_at, end_at):
    while start_at <= end_at:
        _end = start_at + relativedelta(months=1)
        query = """
        select {}
        from ...
        where DATIME >= TO_DATE('{}', 'yyyy-MM-dd') and DATIME < TO_DATE('{}', 'yyyy-MM-dd')
        """.format(", ".join(covr_update_cols), start_at.strftime('%Y-%m-%d'),
        _end.strftime('%Y-%m-%d'))
        
        df = pd.read_sql(query,con = oracle_conn())
        df.to_csv(cf.DATA_PATH + "/raw/covrpf/covr_{}{:02d}.csv"\
                 .format(start_at.year,start_at.month),index = False, sep = "\t")
        print(start_at)
        start_at += relativedelta(months=1)

def gen_product_history_monthly_v2(st, en):
    while st <= en:
        df = pd.read_csv(cf.DATA_PATH + "/.../covr_{}{:02d}.csv"\
                    .format(st.year,st.month), sep = "\t")

       print("redacted")
        
        rider_dict = {"ADD": "accident_r", "CIR": "ci_r", "HCR": "health_r",
                "HSR": "health_r", "TLR": "term_r"}
        df["rider_type"].replace(rider_dict, inplace=True)
        
        print("redacted")
        
        _tempsa = df.groupby(["ym","policyid"]).agg({"SUMINS":"sum"}).reset_index()
        rider_count = df.groupby(["ym","policyid","rider_type"])["rider_flag"].sum().unstack().reset_index()
        
        print("redacted")
       
        df = rider_count.merge(_tempsa, how="left", on=["policyid","ym"])
        df["rider_flag"] = df.iloc[:,-7:-1].sum(axis=1)
        
        print("redacted")

        df.to_csv(cf.DATA_PATH + '/feature/product_history/product_history_monthly_{}{:02d}.csv'\
                    .format(st.year,st.month), sep = "\t",index=False)

        print(st)
        st = st + relativedelta(months=1)



#%%main
if __name__ == "__main__":
    get_latest_covr()
    gen_product_info()
    
    start_at = datetime.strptime(sys.argv[1], "%Y%m")
    end_at = datetime.strptime(sys.argv[2], "%Y%m")
    get_update_covr(start_at, end_at)
    gen_product_history_monthly_v2(start_at, end_at)