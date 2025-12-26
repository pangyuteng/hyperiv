import pandas as pd
import numpy as np
from data_util import find_closest_elements

import pandas_market_calendars as mcal
import datetime

nyse = mcal.get_calendar('NYSE')
TOTAL_SECONDS_ONE_YEAR = 365*24*60*60 # total seconds

def get_market_open_close(day_stamp,no_tzinfo=True):
    early = nyse.schedule(start_date=day_stamp, end_date=day_stamp)
    if len(early) == 0:
        raise LookupError("market not open today!")
    market_open = list(early.to_dict()['market_open'].values())[0]
    market_close = list(early.to_dict()['market_close'].values())[0]
    if no_tzinfo:
        return market_open.replace(tzinfo=None),market_close.replace(tzinfo=None)
    else:
        return market_open,market_close

def get_expiry_tstamp(expiry):
    if not isinstance(expiry,str):
        return np.nan
    expiry = datetime.datetime.strptime(expiry,"%Y-%m-%d")
    _,expiry_tstamp = get_market_open_close(expiry)
    return expiry_tstamp.replace(tzinfo=None)

def get_annualized_time_to_expiration(row,expiry_mapper):
    if isinstance(row.expiry,str):
        expiry = row.expiry
    else:
        expiry = row.expiry.strftime("%Y-%m-%d")
    expiry_tstamp = expiry_mapper[expiry]
    sec_to_expiration = (expiry_tstamp-row.tstamp).total_seconds()
    atte = sec_to_expiration/TOTAL_SECONDS_ONE_YEAR
    return atte


cols=['date', 'forward_price', 'tau', 'risk_free_rate', 'is_call', 'strike_price', 'option_price', 'log_moneyness', 'implied_volatility', 'delta', 'time_to_maturity']

def gen_data(dstamp,zero_day_only=True):
    ticker = "SPXW"
    pq_file = f"/mnt/hd1/data/uw-options-cache/SPX/{dstamp}.parquet.gzip"
    df = pd.read_parquet(pq_file)
    df['tstamp_min'] = df.tstamp_sec.apply(lambda x:x.replace(second=0))
    assert(ticker == list(df.underlying_symbol.unique())[0])
    print(df.shape)
    if zero_day_only:
        df = df[df.expiry == dstamp]
    print(df.shape)

    expiry_mapper = {x:get_expiry_tstamp(x) for x in df.expiry.unique()}
    df['date']=df.tstamp_min
    df['forward_price']=df.underlying_price
    df['tau']=df.apply(lambda x: get_annualized_time_to_expiration(x,expiry_mapper),axis=1)
    df['risk_free_rate']=1e-7
    df['is_call']=df.option_type.apply(lambda x: 1.0 if x == 'call' else -1.0)
    df['strike_price']=df.strike
    df['option_price']=df.price
    df['log_moneyness']= np.log(df.underlying_price/df.strike)
    # df.implied_volatility
    # df.delta
    df['time_to_maturity']=((df.tau*TOTAL_SECONDS_ONE_YEAR)/(60*60*24)) #???
    #df['time_to_maturity']=((df.tau*TOTAL_SECONDS_ONE_YEAR)/(60*60*24)).astype(int) #???
    
    df = df[(df.tau>0)&(df.log_moneyness.notnull())]
    df = df[cols]
    df['is_ref'] = (np.random.rand(len(df)) > 0.8).astype(int)
    # https://quant.stackexchange.com/questions/43596/what-is-forward-moneyness-and-how-to-calculate-it
    if False:
        target_ttms = [0]
        target_deltas = [0.5, 0.25, -0.25]
        df_tmp = df.groupby('date').apply(find_closest_elements, 'time_to_maturity', target_ttms, include_groups=False)
        reference_options = df_tmp.groupby(['date','time_to_maturity']).apply(find_closest_elements, 'delta', target_deltas, include_groups=False)
        df['is_ref'] = 0
        df.loc[reference_options.index.get_level_values(-1), 'is_ref'] = 1

    df = df.dropna()
    return df

mylist = []
for dstamp in ["2025-12-22","2025-12-23","2025-12-24"]:
    tdf = gen_data(dstamp)
    mylist.append(tdf)

df = pd.concat(mylist)
df = df.reset_index()
df.to_hdf(path_or_buf='spx_w_ref.h5', key='df', complevel=9, complib='blosc')