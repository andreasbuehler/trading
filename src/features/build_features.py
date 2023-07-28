import os
import sys
import pandas as pd
import numpy as np
import argparse
from pandarallel import pandarallel
from dataclasses import dataclass

from talib import MA_Type
import talib

import time
import timeit
from functools import wraps

from src.data.data import Data

# Generic print options
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class Conditions():
    '''
    ...[x1, x2, ..., xN] -> specific values
    ...(x1, x2) -> range
    ..._p -> previous 

    close > ema_5min_p[5,7,9,15,100,200]
    ema_5min[5,7,9,15,100,200] > ema_5min[5,7,9,15,100,200]
    ema_4h[5,7,9,15,100,200] > ema_4h[5,7,9,15,100,200] and ema_4h_p[5,7,9,15,100,200] > ema_4h_p[5,7,9,15,100,200]
    pmar < x(0.9920,1.0000)
    %D_1h > %D_1h_p
    '''
    def __init__(self, data):
        self.data = data
        self.conditions = pd.DataFrame(index=data.index)
        self.numConditions = 0
        self.conditionList = []
        self.conditionNamesList = []

    def finalize(self):
        self.conditions = pd.concat([self.conditions, *self.conditionList], axis=1)
        self.conditionNames = pd.DataFrame(self.conditionNamesList)

    def getShiftFromTimeframe(self, timeframe='5min'):
        if timeframe == '5min':
            return 1
        if timeframe == '15min':
            return 1 + 3
        elif timeframe == '1h':
            return 1 + 12
        elif timeframe == '4h':
            return 1 + 48
        elif timeframe == '1d':
            return 1 + 288
        else:
            raise Exception("Invalid timeframe!")

    def biggerThanClose(self, indicator):
        condition = self.data[indicator] > self.data.close
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} > close'.format(indicator)
        self.conditionNamesList.append(conditionName)

    def smallerThanClose(self, indicator):
        condition = self.data[indicator] < self.data.close
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} < close'.format(indicator)
        self.conditionNamesList.append(conditionName)

    def biggerThan(self, indicator1, indicator2):
        condition = self.data[indicator1] > self.data[indicator2]
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} > {}'.format(indicator1, indicator2)
        self.conditionNamesList.append(conditionName)

    def smallerThan(self, indicator1, indicator2):
        condition = self.data[indicator1] < self.data[indicator2]
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} < {}'.format(indicator1, indicator2)
        self.conditionNamesList.append(conditionName)

    def biggerThanX(self, indicator, x):
        condition = self.data[indicator] > x
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} > {}'.format(indicator, x)
        self.conditionNamesList.append(conditionName)

    def smallerThanX(self, indicator, x):
        condition = self.data[indicator] < x
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} < {}'.format(indicator, x)
        self.conditionNamesList.append(conditionName)

    def biggerThanPrevious(self, indicator):
        # We have to adjust based on the timeframe for the shift to be correct here!
        timeframe = indicator.split('_')[-1]
        shift = self.getShiftFromTimeframe(timeframe)
        condition = self.data[indicator] > self.data[indicator].shift(periods=-shift, fill_value=0)
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} > {}[-1]'.format(indicator, indicator)
        self.conditionNamesList.append(conditionName)

    def smallerThanPrevious(self, indicator):
        timeframe = indicator.split('_')[-1]
        shift = self.getShiftFromTimeframe(timeframe)
        condition = self.data[indicator] < self.data[indicator].shift(periods=-shift, fill_value=0)
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} < {}[-1]'.format(indicator, indicator)
        self.conditionNamesList.append(conditionName)

    def crossing(self, indicator1, indicator2):
        timeframe = indicator1.split('_')[-1]
        timeframe2 = indicator2.split('_')[-1]
        shift = self.getShiftFromTimeframe(timeframe)
        shift2 = self.getShiftFromTimeframe(timeframe2)

        if shift != shift2:
            raise Exception("Cannot cross different timeframes!")

        condition = (self.data[indicator1] > self.data[indicator2]) & (self.data[indicator1].shift(periods=-shift, fill_value=0) < self.data[indicator2].shift(periods=-shift, fill_value=0))
        condition.rename(str(self.numConditions), inplace=True)
        self.conditionList.append(condition)
        self.numConditions += 1

        conditionName = '{} > {} and {}[-1] < {}[-1]'.format(indicator1, indicator2, indicator1, indicator2)
        self.conditionNamesList.append(conditionName)


class Indicators():
    '''
    https://ta-lib.github.io/ta-lib-python/doc_index.html
    '''
    def __init__(self, data):
        self.data = data
        self.data5min = data
        self.data15min = data.resample('15m', closed='left', label='right').apply('last')
        self.data1h = data.resample('1h', closed='left', label='right').apply('last')
        self.data4h = data.resample('4h', closed='left', label='right').apply('last')
        self.data1d = data.resample('1d', closed='left', label='right').apply('last')
        self.indicatorList = []

    def finalize(self):
        self.data = pd.concat([self.data, *self.indicatorList], axis=1)
        self.data.fillna(method='ffill', inplace=True)

    def getTimeframe(self, timeframe='5min'):
        if timeframe == '5min':
            return self.data5min
        if timeframe == '15min':
            return self.data15min
        elif timeframe == '1h':
            return  self.data1h
        elif timeframe == '4h':
            return self.data4h
        elif timeframe == '1d':
            return self.data1d
        else:
            raise Exception("Invalid timeframe!")

    @timeit
    def _computeVWMA(self, data, timeperiod=20):
        # ta.sma(source * volume, length) / ta.sma(volume, length) from here: https://www.tradingcode.net/tradingview/volume-weighted-average/
        vwma = talib.SMA(data.close * data.volume, timeperiod=timeperiod) / talib.SMA(data.volume, timeperiod=timeperiod)
        return vwma

    @timeit
    def _computePMAR(self, data, timeperiod=20):
        # Source: https://de.tradingview.com/script/QK6EciNv-Price-Moving-Average-Ratio-Percentile/
        vwma = self._computeVWMA(data, timeperiod=timeperiod)
        pmar = data.close / vwma
        return pmar

    @timeit
    def addPMAR(self, period=20, timeframe='5min'):
        data = self.getTimeframe(timeframe)
        pmar = self._computePMAR(data, timeperiod=period)
        # We filter the PMAR!
        indicator = talib.SMA(pmar, timeperiod=period)
        indicator.rename('pmar{}_{}'.format(period, timeframe), inplace=True)
        # Old code:
        # self.data = pd.merge(self.data, indicator, how='left', on='date')
        # self.data.fillna(method='ffill', inplace=True)
        self.indicatorList.append(indicator)
        return indicator.name

    @timeit
    def addPMARP(self, period=20, lookback=50, timeframe='5min'):
        data = self.getTimeframe(timeframe)
        abs_pmar = abs(self._computePMAR(data, timeperiod=period))
        computation = lambda x: np.sum(x[-1] > x) / len(x) * 100
        indicator = abs_pmar.rolling(window=lookback+1).parallel_apply(computation).rename('pmarp{}_{}_{}'.format(period, lookback, timeframe))
        # We filter the PMARP!
        indicator = talib.SMA(indicator, timeperiod=period)
        indicator.rename('pmarp{}_{}_{}'.format(period, lookback, timeframe), inplace=True)
        self.indicatorList.append(indicator)
        return indicator.name

    @timeit
    def addMACD(self, fastperiod=12, slowperiod=26, signalperiod=9, timeframe='5min'):
        # https://ta-lib.github.io/ta-lib-python/func_groups/momentum_indicators.html
        # macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        data = self.getTimeframe(timeframe)
        macd, macdsignal, _ = talib.MACDEXT(data.close, fastperiod, MA_Type.EMA, slowperiod, MA_Type.EMA, signalperiod, MA_Type.EMA)
        macd.rename('MACD{}_{}_{}'.format(fastperiod, slowperiod, timeframe), inplace=True)
        macdsignal.rename('MACDSignal{}_{}_{}'.format(fastperiod, slowperiod, timeframe), inplace=True)
        self.indicatorList.append(macd)
        self.indicatorList.append(macdsignal)
        return macd.name, macdsignal.name

    @timeit
    def computeBBW(self, data, timeperiod=20, nbdevup=2, nbdevdn=2, timeframe='5min'):
        upperband, _, lowerband = talib.BBANDS(data.close, timeperiod, nbdevup, nbdevdn, MA_Type.SMA)
        return upperband - lowerband

    @timeit
    def addBBWP(self, period=13, lookback=252, periodMA1=5, timeframe='5min'):
        # https://www.tradingview.com/script/tqitSsyG-Bollinger-Band-Width-Percentile/
        data = self.getTimeframe(timeframe)
        bbw = self.computeBBW(data, timeperiod=period, nbdevup=2, nbdevdn=2)
        computation = lambda x: np.sum(x[-1] > x) / len(x) * 100
        bbwp = bbw.rolling(window=lookback+1).parallel_apply(computation).rename('bbwp{}_{}_{}'.format(period, lookback, timeframe))
        # Filter bbwp
        bbwpMA1 = talib.SMA(bbwp, timeperiod=periodMA1).rename('bbwpMA1{}_{}_{}_{}'.format(period, lookback, periodMA1, timeframe))
        self.indicatorList.append(bbwp)
        self.indicatorList.append(bbwpMA1)
        return bbwp.name, bbwpMA1.name

    @timeit
    def addBB(self, timeperiod=20, nbdevup=2, nbdevdn=2, timeframe='5min'):
        # https://ta-lib.github.io/ta-lib-python/func_groups/overlap_studies.html
        # upperband, middleband, lowerband = BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        data = self.getTimeframe(timeframe)
        upperband, middleband, lowerband = talib.BBANDS(data.close, timeperiod, nbdevup, nbdevdn, MA_Type.SMA)
        upperband.rename('BBUp{}_{}_{}_{}'.format(timeperiod, nbdevup, nbdevdn, timeframe), inplace=True)
        middleband.rename('BBMid{}_{}_{}_{}'.format(timeperiod, nbdevup, nbdevdn, timeframe), inplace=True)
        lowerband.rename('BBLow{}_{}_{}_{}'.format(timeperiod, nbdevup, nbdevdn, timeframe), inplace=True)
        self.indicatorList.append(upperband)
        self.indicatorList.append(middleband)
        self.indicatorList.append(lowerband)
        return upperband.name, middleband.name, lowerband.name

    @timeit
    def addRSI(self, period, timeframe='5min'):
        data = self.getTimeframe(timeframe)
        indicator = talib.RSI(data.close, timeperiod=period).rename('rsi{}_{}'.format(period, timeframe))
        self.indicatorList.append(indicator)
        return indicator.name

    @timeit
    def addRSIEMA(self, periodRSI=14, periodEMA=50, timeframe='5min'):
        # Use SMA
        data = self.getTimeframe(timeframe)
        rsi = talib.RSI(data.close, timeperiod=periodRSI)
        indicator = talib.SMA(rsi, timeperiod=periodEMA).rename('rsisma{}_{}_{}'.format(periodRSI, periodEMA, timeframe))
        self.indicatorList.append(indicator)
        return indicator.name

    @timeit
    def addStochastic(self, fastk_period=14, slowk_period=3, slowd_period=6, timeframe='5min'):
        # slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        data = self.getTimeframe(timeframe)
        slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'], fastk_period, slowk_period, MA_Type.SMA, slowd_period, MA_Type.SMA)
        slowk.rename('%K{}_{}_{}'.format(fastk_period, slowk_period, timeframe), inplace=True)
        slowd.rename('%D{}_{}_{}_{}'.format(fastk_period, slowk_period, slowd_period, timeframe), inplace=True)
        self.indicatorList.append(slowk)
        self.indicatorList.append(slowd)
        return slowk.name, slowd.name

    @timeit
    def addEMA(self, period, timeframe='5min'):
        data = self.getTimeframe(timeframe)
        indicator = talib.EMA(data.close, timeperiod=period).rename('ema{}_{}'.format(period, timeframe))
        self.indicatorList.append(indicator)
        return indicator.name

    @timeit
    def addHLOC(self, timeframe='5min'):
        data = self.getTimeframe(timeframe)
        indicator = pd.Series((data.high + data.low + data.close + data.open) / 4).rename('hloc_{}'.format(timeframe))
        self.indicatorList.append(indicator)
        return indicator.name


    def finalizeIndicators(self):
        # call fillna at the end
        pass


def addIndicators(sourceData):
    indicators = Indicators(sourceData)

    # HLOC
    timeframes = ['5min', '15min', '1h', '4h', '1d']
    for timeframe in timeframes:
        indicators.addHLOC(timeframe=timeframe)

    # EMA
    timeframes = ['5min', '15min', '1h', '4h', '1d']
    periods = [7, 9, 21, 55, 100, 200]
    for timeframe in timeframes:
        for period in periods:
            indicators.addEMA(period=period, timeframe=timeframe)

    # PMAR
    timeframes = ['5min']
    periods = range(7, 50, 5)
    for timeframe in timeframes:
        for period in periods:
            indicators.addPMAR(period=period, timeframe=timeframe)

    # PMARP
    timeframes = ['5min']
    periods = range(7, 50, 20)
    for timeframe in timeframes:
        for period in periods:
            # LOOKBACK FIXED TO 100 FOR PERFORMANCE!
            indicators.addPMARP(period=period, lookback=100, timeframe=timeframe)

    # RSI
    timeframes = ['5min', '15min', '1h', '4h', '1d']
    periods = [14]
    for timeframe in timeframes:
        for period in periods:
            indicators.addRSI(period=period, timeframe=timeframe)
            indicators.addRSIEMA(periodRSI=period, periodEMA=period, timeframe=timeframe)

    # Stochastic
    timeframes = ['5min', '15min', '1h', '4h', '1d']
    for timeframe in timeframes:
        indicators.addStochastic(fastk_period=14, slowk_period=3, slowd_period=6, timeframe=timeframe)

    # MACD
    timeframes = ['15min', '1h', '4h', '1d']
    for timeframe in timeframes:
        indicators.addMACD(fastperiod=12, slowperiod=26, signalperiod=9, timeframe=timeframe)

    # BBWP
    timeframes = ['15min', '1h', '4h', '1d']
    for timeframe in timeframes:
        indicators.addBBWP(period=13, lookback=252, periodMA1=5, timeframe=timeframe)

    indicators.finalize()

    print("Indicators added, num of columns: {}".format(len(indicators.data.columns)))

    return indicators.data


def computeConditions(data):
    conditions = Conditions(data)

    # HLOC
    hloc_indicators = [s for s in data.columns.to_list() if 'hloc' in s]
    for indicator in hloc_indicators:
        conditions.biggerThanPrevious(indicator)
        conditions.smallerThanPrevious(indicator)

    # EMA
    ema_indicators = [s for s in data.columns.to_list() if 'ema' in s]
    for indicator in ema_indicators:
        conditions.biggerThanClose(indicator)
        conditions.smallerThanClose(indicator)
        conditions.biggerThanPrevious(indicator)
        conditions.smallerThanPrevious(indicator)

        for indicator2 in ema_indicators:
            conditions.biggerThan(indicator, indicator2)
            conditions.smallerThan(indicator, indicator2)

    # RSI
    rsi_indicators = [s for s in data.columns.to_list() if 'rsi' in s and not 'rsisma' in s]
    x_smaller_range = range(15, 60, 5)
    x_larger_range = range(35, 90, 5)
    for indicator in rsi_indicators:
        conditions.biggerThanPrevious(indicator)
        conditions.smallerThanPrevious(indicator)

        for x_smaller in x_smaller_range:
            conditions.smallerThanX(indicator, x_smaller)

        for x_larger in x_larger_range:
            conditions.biggerThanX(indicator, x_larger)

    # RSIEMA
    rsiema_indicators = [s for s in data.columns.to_list() if 'rsiema' in s]
    for indicator in rsiema_indicators:
        conditions.biggerThanPrevious(indicator)
        conditions.smallerThanPrevious(indicator)

    # PMAR
    indicators = [s for s in data.columns.to_list() if 'pmar' in s and not 'pmarp' in s]

    x_smaller_range = np.arange(0.9920, 1.0000, 0.002)
    x_larger_range = np.arange(0.9970, 1.0100, 0.002)
    for indicator in indicators:

        for x_smaller in x_smaller_range:
            conditions.smallerThanX(indicator, x_smaller)

        for x_larger in x_larger_range:
            conditions.biggerThanX(indicator, x_larger)

    # PMARP
    indicators = [s for s in data.columns.to_list() if 'pmarp' in s]
 
    x_smaller_range = range(1, 70, 10)
    x_larger_range = range(30, 99, 10)
    for indicator in indicators:

        for x_smaller in x_smaller_range:
            conditions.smallerThanX(indicator, x_smaller)

        for x_larger in x_larger_range:
            conditions.biggerThanX(indicator, x_larger)


    # Stochastic
    indicatorsD = [s for s in data.columns.to_list() if '%D' in s]
    indicatorsK = [s for s in data.columns.to_list() if '%K' in s]
 
    x_smaller_range = range(10, 40, 5)
    x_larger_range = range(60, 95, 5)
    for indicator in indicatorsD:

        for x_smaller in x_smaller_range:
            conditions.smallerThanX(indicator, x_smaller)

        for x_larger in x_larger_range:
            conditions.biggerThanX(indicator, x_larger)

        conditions.biggerThanPrevious(indicator)
        conditions.smallerThanPrevious(indicator)

        for indicatorK in indicatorsK:
            conditions.biggerThan(indicator, indicatorK)
            conditions.smallerThan(indicator, indicatorK)

    # MACD
    indicatorsMACD = [s for s in data.columns.to_list() if 'MACD' in s and not 'MACDSignal' in s]
    indicatorsMACDSignal = [s for s in data.columns.to_list() if 'MACDSignal' in s]
    for indicator in indicatorsMACD:
        conditions.biggerThanX(indicator, 0)
        conditions.smallerThanX(indicator, 0)

        for indicatorSignal in indicatorsMACDSignal:
            conditions.biggerThan(indicator, indicatorSignal)
            conditions.smallerThan(indicator, indicatorSignal)

    # BBWP
    indicatorsBBWP = [s for s in data.columns.to_list() if 'bbwp' in s and not 'bbwpMA1' in s]
    x_range = np.arange(0.7, 0.99, 0.05)
    for indicator in indicatorsBBWP:

        conditions.biggerThanPrevious(indicator)
        conditions.smallerThanPrevious(indicator)

        for x in x_range:
            conditions.biggerThanX(indicator, x)
            conditions.smallerThanX(indicator, x)

    conditions.finalize()

    return conditions.conditions, conditions.conditionNames


def compute_labels_from_data(data, take_profit=1.05, stop_loss=0.98, ticks_until_profit_required=10):
    '''
    For each tick assign 0 (do nothing) or 1 (buy)
    For each tick look at the future development. If the price goes to > take_profit before going < stop loss, assign 1. 0 otherwise.
    '''
    labels = pd.DataFrame(index=data.index)
    labels['gt'] = False
    
    def compute(x):
        profit = (x.where(x[0] * take_profit < x).first_valid_index())
        loss = (x.where(x[0] * stop_loss > x).first_valid_index())
        if profit == None :
            return False
        elif loss == None:
            return True
        else:
            return bool(profit < loss)

    # https://stackoverflow.com/questions/22820292/how-to-use-pandas-rolling-functions-on-a-forward-looking-basis
    labels = data.close.shift(-ticks_until_profit_required).rolling(ticks_until_profit_required, min_periods=1).parallel_apply(compute).astype('bool')

    print("Labels computed, number of rows: {}".format(len(labels)))

    return labels


def buildFeaturesFromIntermediateData(data):
    conditions, conditionNames = computeConditions(data)
    conditions['gt'] = data['gt']
    return conditions, conditionNames


def buildIntermediateFeaturesFromRawData(sourceData):
    labels = compute_labels_from_data(sourceData, 1.02, 0.99, 75)
    sourceData['gt'] = labels
    data = addIndicators(sourceData)
    return data


@dataclass
class DataConfig():
    interim_data_object_path: str=os.path.join('data/interim', 'data.pkl')
    processed_data_object_path: str=os.path.join('data/processed', 'conditions.pkl')
    processed_data_csv_path: str=os.path.join('data/processed', 'conditionsNames.csv')


def parseArgs():
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('dataPath', type=str, help='/path/to/rawData.csv')

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArgs()

    pandarallel.initialize(progress_bar=True, verbose=2)

    dataConfig = DataConfig()

    # Get raw data.
    data = Data()
    data.loadRawDataDodo(args.dataPath)

    # Compute intermediate features (gt, indicators).
    dataIntermediate = buildIntermediateFeaturesFromRawData(data.data)

    # Compute conditions.
    conditions, conditionNames = buildFeaturesFromIntermediateData(dataIntermediate)

    # Dump the data.
    data.dumpDataPickle(dataIntermediate, dataConfig.interim_data_object_path)
    data.dumpDataPickle(conditions, dataConfig.processed_data_object_path)
    data.dumpDataCSV(conditionNames, dataConfig.processed_data_csv_path)

