'''
Created on Nov 9, 2018

@author: mofir
'''

from __future__ import division

import pandas as pd
import numpy as np

from PriceBasedPrediction.PrepareData import ConfidenceAreas

import matplotlib
from matplotlib import pyplot as plt

from bokeh.layouts import column

class StatParam(object):
    false_negative_ratio      = 0
    false_positive_ratio      = 0
    true_negative_ratio       = 0
    true_positive_ratio       = 0
    #missing_buy_in_up_ratio   = 0
    #false_buy_ratio_in_down   = 0
    AvgGainEXp                 = 0
    AvgGainEXp_opt             = 0
    mean_gain                  = 0
    mean_error                 = 0
    #std                       = 0
    #mean_potential_gain       = 0

    def __init__(self,
                 false_negative_ratio,
                 false_positive_ratio,
                 true_negative_ratio,
                 true_positive_ratio,
                 #missing_buy_ratio_in_up,
                 #false_buy_ratio_in_down,
                 mean_gain,
                 mean_error
                 #std,
                 #mean_potential_gain
                 ):
        self.false_negative_ratio   = false_negative_ratio
        self.false_positive_ratio   = false_positive_ratio
        self.true_negative_ratio    = true_negative_ratio
        self.true_positive_ratio    = true_positive_ratio
        #self.missing_buy_in_up_ratio = missing_buy_ratio_in_up
        #self.false_buy_in_down_ratio = false_buy_ratio_in_down
        self.mean_gain               = mean_gain
        self.mean_error              = mean_error
        #self.std                     = std
        #self.mean_potential_gain     = mean_potential_gain

def GetClassStatList():
    all_parameters = dir(StatParam)
    stat_parameters = [attr for attr in all_parameters if not callable(getattr(StatParam, attr)) and not attr.startswith("__")]
    return stat_parameters

def GetBuyVector(y_pred,target_type = 'reg'):
    #TODO - can do here better with using Network to consider bias?
    #print(y_pred)
    if (target_type == 'reg'):
        next_predicted_value = y_pred[1:]
        curr_predicted_value = y_pred[0:-1]
        buy_vector = np.greater(next_predicted_value,curr_predicted_value)
    else:
        buy_vector = []
        for day in y_pred:
            buy_vector.append((day > int(ConfidenceAreas.rise_low)))
    #print(buy_vector)
    return buy_vector

def GetProfit(buy_vector, y_true):
    #TODO - can do here better with using Network to consider bias?
    return MeanGain(y_true,buy_vector)

def FalseNegativeRatio(real_value,buy_vector,target_type):
    # rate of FN compared to all Positive days
    real_buy_vector = GetBuyVector(real_value,target_type)
    FN_vec = (1-buy_vector) * real_buy_vector
    return FN_vec.sum()/real_buy_vector.sum()

def FalsePositiveRatio(real_value,buy_vector,target_type):
    # rate of FP compared to all Negative
    real_buy_vector = GetBuyVector(real_value,target_type)
    neg_days_vec = 1 - real_buy_vector #(days_delta_vec <=0)
    FP_vec = buy_vector * neg_days_vec
    return FP_vec.sum()/neg_days_vec.sum()

def TrueNegativeRatio(real_value,buy_vector,target_type):
    real_buy_vector = GetBuyVector(real_value,target_type)
    neg_days_vec = np.invert(real_buy_vector) #(days_delta_vec <= 0)
    neg_buy_vector = np.invert(buy_vector)
    TN_vec = neg_buy_vector * neg_days_vec
    return TN_vec.sum()/neg_days_vec.sum()

def TruePositiveRatio(real_value,buy_vector,target_type):
    real_buy_vector = GetBuyVector(real_value,target_type)
    TP_vec = buy_vector * real_buy_vector
    return TP_vec.sum()/real_buy_vector.sum()

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
    days_delta_ratio = np.subtract(real_value_shifted,last_value)/last_value
    gain_vector = buy_vector * days_delta_ratio
    mean_gain = gain_vector.sum()/len(days_delta_ratio)

    potential_gain = days_delta_ratio * (days_delta_ratio > 0) #take only days the stock increased
    mean_potential_gain = potential_gain.sum()/len(days_delta_ratio)

    return mean_gain/mean_potential_gain

def AvgGain(real_value,buy_vector):
    #how much gain starting (starting form 1) we have achieved
    today_real_value    = real_value[0:-1]
    tommorow_real_value = real_value[1:]
    opt_buy_vector      = GetBuyVector(real_value)
    gain = 1
    opt_gain = 1

    for ind, close_price in enumerate(today_real_value):
        gain = gain * (tommorow_real_value[ind]/today_real_value[ind]) * buy_vector[ind] + gain * (1 - buy_vector[ind])
        opt_gain = opt_gain * (tommorow_real_value[ind]/today_real_value[ind]) * opt_buy_vector[ind] + opt_gain * (1 - opt_buy_vector[ind])

    return gain

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

def CalculateAllStatistics(real_close_value,real_value,predictad_value,target_type,buy_vector = None,plot_buy_decisions = False): #should include in [0] the previous day
    print('hello from CalculateAllStatistics...')
    #real_value = real_value.reshape(1,-1)[0]
    if buy_vector is None:
        buy_vector = GetBuyVector(predictad_value,target_type)

    df_prediction_statistics = pd.DataFrame(columns=GetClassStatList())

    df_prediction_statistics.at[0,'true_negative_ratio']      = TrueNegativeRatio(real_value,buy_vector,target_type)
    df_prediction_statistics.at[0,'true_positive_ratio']      = TruePositiveRatio(real_value,buy_vector,target_type)
    df_prediction_statistics.at[0,'false_negative_ratio']     = FalseNegativeRatio(real_value,buy_vector,target_type)
    df_prediction_statistics.at[0,'false_positive_ratio']     = FalsePositiveRatio(real_value,buy_vector,target_type)
    #df_prediction_statistics.at[0,'missing_buy_in_up_ratio']   = MissingBuyInUpRatio(real_value,buy_vector)
    #df_prediction_statistics.at[0,'false_buy_ratio_in_down']   = FalseBuyInDownRatio(real_value,buy_vector)
    df_prediction_statistics.at[0,'mean_error']                = MeanError(real_value,predictad_value) if target_type == 'reg' else None
    #df_prediction_statistics.at[0,'std']                       = 0
    df_prediction_statistics.at[0,'mean_gain']      = MeanGain(real_close_value,buy_vector)
    df_prediction_statistics.at[0,'AvgGainEXp']     = AvgGain(real_close_value,buy_vector)
    df_prediction_statistics.at[0,'AvgGainEXp_Opt'] = AvgGain(real_close_value,GetBuyVector(real_value))

    #print(df_prediction_statistics)

    #try to understand where the buy decision is good and when is missing
    last_value = real_value[0:-1]
    real_value_shifted = real_value[1:]
    predictad_value_shifted = predictad_value[1:]

    real_buy_vector = real_value_shifted > last_value

    correct_prediction_vector = np.zeros(real_buy_vector.shape)
    correct_prediction_vector[real_buy_vector==buy_vector] = 1

    good_predict  = correct_prediction_vector * real_value_shifted
    bad_predict   = (1-correct_prediction_vector) * real_value_shifted
    #we don't want to plot zeroes, only on the stock value
    good_predict [ good_predict==0 ] = np.nan
    bad_predict[ bad_predict==0 ] = np.nan

    if plot_buy_decisions:
        plt.plot(good_predict,'o',color='green', label = "buy decision successful")
        plt.plot(bad_predict,'o',color='red', label = "buy decision is bad")
        plt.plot(real_value_shifted,color='blue', label = "stock timeline")
        plt.ylabel('Price and decision (green)')
        plt.xlabel('time line')
        if buy_vector is None:
            plt.title('predicted stock and algorithm action')
        else:
            plt.title('predicted stock and algorithm action with RF classifier combined')
        plt.show(block=False)

    return df_prediction_statistics