import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import random
import math

from copy import copy

#data cleaning function
def clean_df(input_df, cols):
    df = input_df

    # changing datetime column to datetime class
    df['pickup_datetime_hold'] = pd.to_datetime(df['pickup_datetime'])

    # adding minute column
    df['pickup_minute'] = df.apply(lambda x: x.pickup_datetime_hold.minute, axis=1)

    # adding hour column
    df['pickup_hour'] = df.apply(lambda x: x.pickup_datetime_hold.hour, axis=1)

    # adding month column
    df['pickup_month'] = df.apply(lambda x: x.pickup_datetime_hold.month, axis=1)

    # adding day of month column
    df['pickup_day'] = df.apply(lambda x: x.pickup_datetime_hold.day, axis=1)

    # adding day of week column
    df['pickup_weekday'] = df.apply(lambda x: datetime.weekday(x.pickup_datetime_hold), axis=1)

    return (df[df.columns.intersection(cols)])

#converting seconds into hour, minutes, seconds, microseconds
def convert(start_time, add):
    add = add % (24 * 3600)
    hour = add // 3600
    add %= 3600
    minutes = add // 60
    add %= 60

    return (start_time + timedelta(seconds=add, minutes=minutes, hours=hour))

#setup function to only keep columns we want
def setup(x):
    hold = pd.concat([x, x.shift(-1)], axis = 1).dropna()
    hold.columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    return(hold)


# return total time of a set of trips
def get_total_time(x, start_time):
    # setting up intial input df, time, and holding array
    temp = setup(x)
    start_time_hold = start_time
    time_total = [0] * len(temp)

    for i in range(len(temp)):
        hold = set_input(temp, i, start_time_hold, col_list)
        trip_dur = (np.exp(model.predict(xgb.DMatrix(hold))) - 1)[0]
        time_total[i] = trip_dur
        start_time_hold = convert(start_time_hold, trip_dur)

    return (sum(time_total))


# get intial population for GA
def get_init_pop(x, num):
    hold = []
    pop = list(range(x))

    for i in range(num):
        hold.append((random.sample(pop, x)))

    return (hold)


# prepare/rank intial population
def rank(input_pop):
    hold_df = pd.DataFrame(input_pop)

    total_time_hold = [0] * len(input_pop)

    for i in range(len(input_pop)):
        total_time_hold[i] = get_total_time(temp.reindex(input_pop[i]), x)

    # add total time column / fitness
    hold_df['total_time'] = total_time_hold

    # add rank column which gives the highest rank to the combination with the lowest total trip duration
    hold_df['rank'] = len(hold_df) - hold_df['total_time'].rank() + 1

    # add chance column which uses rank to calculate probability of being chosen as a parent
    ##higher fitness means higher chance of breeding
    hold_df['chance'] = hold_df['rank'] * 2 / (len(hold_df) * (len(hold_df) + 1))

    return (hold_df)

#choose a set parents from our population based on the fitness
##parents shouls be equal
def get_parent(input_df):
    pop_size = len(input_df)
    which_parent = np.random.choice(pop_size, pop_size, p = input_df['chance'])
    num_chrom = len(list(set(input_df.columns) - set(['total_time', 'rank', 'chance'])))
    parent = input_df.reindex(which_parent).iloc[:,:num_chrom].to_numpy()
    return(parent)


def cross(parent1, parent2):
    # how many chromosomes we want to keep from each parent
    num_parent = len(parent1)
    num_chrom = len(parent1[0])
    num_one = math.ceil(num_chrom / 2)
    num_zero = num_chrom - math.ceil(num_chrom / 2)  # safer option than using floor

    chrom_filter_source = ([1] * num_one) + ([0] * num_zero)

    # create #create a "filter" for our chromosomes
    chrom_filter = []
    for i in range(num_parent):
        chrom_filter.append((random.sample(chrom_filter_source, num_chrom)))

    # create opposite filter for our second parent
    chrom_filter2 = abs(np.subtract(chrom_filter, 1))

    # need to add 1 because our lowest number right now is 0 if left alone when filtered, we will have multipl 0's
    new_parent1 = np.multiply(np.add(parent1, 1), chrom_filter)
    new_parent2 = np.multiply(np.add(parent2, 1), chrom_filter2)

    # loop through all parents and cross chromosomes
    for i in range(num_parent):
        # only look at couples with matching chromosomes
        if (any(x in new_parent1[i] for x in new_parent2[i])):
            # available chromosomes to choose from to fill gaps
            not_set = list(set(list(range(num_chrom + 1))[1:]) - set(new_parent1[i] + new_parent2[i]))
            # fill gaps
            for idx, j in enumerate(new_parent2[i]):
                if (j in new_parent1[i] and j != 0):
                    insert = random.sample(not_set, 1)[0]
                    new_parent1[i, idx] = insert
                    new_parent2[i, idx] = 0
                    not_set.remove(insert)  # once a chromosome is used remove it from possible choices

    # add them together as a cross
    return (np.add(new_parent1, new_parent2))


def get_next_gen(input_df):
    #choose parents
    p1 = get_parent(input_df)
    p2 = get_parent(input_df)

    #cross
    return (cross(p1, p2))


#this is a supplmentary function which will help with our threshold
def condense (input_df):
    return(input_df.groupby(list(set(input_df.columns) - set(['rank', 'chance']))).sum().reset_index())
