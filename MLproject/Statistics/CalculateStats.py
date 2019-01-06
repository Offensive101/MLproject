'''
Created on Nov 9, 2018

@author: mofir
'''

from __future__ import division

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from bokeh.layouts import column

class StatParam(object):
    false_negative_ratio      = 0
    false_positive_ratio      = 0
    missing_buy_in_up_ratio   = 0
    false_buy_ratio_in_down   = 0
    mean_gain                 = 0
    mean_error                = 0
    std                       = 0
    mean_potential_gain       = 0

    def __init__(self,
                 false_negative_ratio,
                 false_positive_ratio,
                 missing_buy_ratio_in_up,
                 false_buy_ratio_in_down,
                 mean_gain,
                 mean_error,
                 std,
                 mean_potential_gain):
        self.false_negative_ratio    = false_negative_ratio
        self.false_positive_ratio    = false_positive_ratio
        self.missing_buy_in_up_ratio = missing_buy_ratio_in_up
        self.false_buy_in_down_ratio = false_buy_ratio_in_down
        self.mean_gain               = mean_gain
        self.mean_error              = mean_error
        self.std                     = std
        self.mean_potential_gain     = mean_potential_gain

def GetClassStatList():
    all_parameters = dir(StatParam)
    stat_parameters = [attr for attr in all_parameters if not callable(getattr(StatParam, attr)) and not attr.startswith("__")]
    return stat_parameters

def FalseNegativeRatio(real_value,buy_vector):
    # rate of FN compared to all Positive days
    last_value = real_value[0:-1]
    real_value_shifted = real_value[1:]
    days_delta_vec = (real_value_shifted - last_value)
    pos_days_vec = (days_delta_vec > 0)
    FN_vec = (1-buy_vector) * pos_days_vec
    return FN_vec.sum()/pos_days_vec.sum()

def FalsePositiveRatio(real_value,buy_vector):
    # rate of FP compared to all Negative
    last_value = real_value[0:-1]
    real_value_shifted = real_value[1:]
    days_delta_vec = (real_value_shifted - last_value)
    neg_days_vec = (days_delta_vec < 0)
    FP_vec = buy_vector * neg_days_vec
    return FP_vec.sum()/neg_days_vec.sum()

def MissingBuyInUpRatio(real_value,buy_vector):
    # decision not to buy although we saw an up slope the day before

    last_value = real_value[:-2]
    next_value = real_value[1:-1]

    delta_price_between_prev_day_vector = (last_value - next_value) #if positive - its NEG slope: yesterday was bigger than today
    positive_delta_vector = abs(delta_price_between_prev_day_vector * (delta_price_between_prev_day_vector<0))
    buy_vector = buy_vector[1:]
    non_buy_vector = 1-buy_vector
    missing_buy_vector = non_buy_vector * (positive_delta_vector > 0)

    #present the value in precentage out of total buys:
    missing_buy_ratio = missing_buy_vector.sum()/non_buy_vector.sum()
    return missing_buy_ratio

def FalseBuyInDownRatio(real_value,buy_vector):
    # decision to buy although we saw a down slope the day before
    last_value = real_value[:-2]
    next_value = real_value[1:-1]

    delta_price_between_prev_day_vector = last_value - next_value #if positive - its NEG slope: yesterday was bigger than today
    negative_delta_vector = delta_price_between_prev_day_vector * (delta_price_between_prev_day_vector>0)

    buy_vector = buy_vector[1:]

    false_buy_vector  = buy_vector * (negative_delta_vector > 0)
    false_buy_ratio   = false_buy_vector.sum()/buy_vector.sum()
    return false_buy_ratio

def MeanGain(real_value,buy_vector):
    #how much gain in precentage we have achieved
    last_value = real_value[0:-1]
    real_value_shifted = real_value[1:]
    days_delta_ratio = ((real_value_shifted - last_value)/last_value)
    gain_vector = buy_vector * days_delta_ratio
    mean_gain = gain_vector.sum()/len(days_delta_ratio)
    return mean_gain

def MeanPotentialGain(real_value):
    #how much gain in precentage we could have achieved
    last_value = real_value[0:-1]
    real_value_shifted = real_value[1:]
    days_delta_ratio = ((real_value_shifted - last_value)/last_value)
    potaential_gain = days_delta_ratio * (days_delta_ratio > 0) #take only days the stock increased
    mean_potaential_gain = potaential_gain.sum()/len(days_delta_ratio)
    return mean_potaential_gain

def MeanError(real_value,predictad_value):
    #between predicted and actual stock value
    values_diff = (predictad_value - real_value)
    error = 100 * abs(values_diff)/real_value
    return error.mean()

def CalculateAllStatistics(real_value,predictad_value,buy_vector,plot_buy_decisions = False): #should include in [0] the previous day
    df_prediction_statistics = pd.DataFrame(columns=GetClassStatList())

    df_prediction_statistics.at[0,'false_negative_ratio']      = FalseNegativeRatio(real_value,buy_vector)
    df_prediction_statistics.at[0,'false_positive_ratio']      = FalsePositiveRatio(real_value,buy_vector)
    df_prediction_statistics.at[0,'missing_buy_in_up_ratio']   = MissingBuyInUpRatio(real_value,buy_vector)
    df_prediction_statistics.at[0,'false_buy_ratio_in_down']   = FalseBuyInDownRatio(real_value,buy_vector)
    df_prediction_statistics.at[0,'mean_gain']                 = MeanGain(real_value,buy_vector)
    df_prediction_statistics.at[0,'mean_potential_gain']       = MeanPotentialGain(real_value)
    df_prediction_statistics.at[0,'mean_error']                = MeanError(real_value,predictad_value)
    df_prediction_statistics.at[0,'std']                       = 0

    #print(df_prediction_statistics)

    #try to understand where the buy decision is good and when is missing

    last_value = real_value[0:-1]
    real_value_shifted = real_value[1:]
    predictad_value_shifted = predictad_value[1:]

    #buy_vector = predictad_value_shifted > last_value

    if plot_buy_decisions:
        plt.plot(buy_vector * real_value_shifted,'o',color='green')
        plt.plot((1-buy_vector) * real_value_shifted,'o',color='red')
        plt.plot(real_value_shifted,color='blue')
        plt.ylabel('Price and decision (green)')
        plt.xlabel('time line')
    #plt.show()

    return df_prediction_statistics