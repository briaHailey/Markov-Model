"""

Name:Bria Weisblat
Date: 03/13/24
Assignment:Project #9 (Question 3)
Due Date: 03/24/24
About this project: This project creates a Markov Model of the recorded PM2.5 Index based
on data from the Changping data set. I broke the PM2.5 readings into four ranges: low(0-50),
moderate(51-100), high(101-150), and very high(151+). I then calculated the transition probabilities
for each of those four ranges and used them to create the Markov Model below.
Assumptions:Assumes proper implementation and function of everything imported.
All work below was performed by Bria Weisblat

"""

import random
import pandas as pd
import numpy as np
from MarkovChain import *
from MChainDraw import *
from statistics import mean

random.seed("aima-python")

file = r"PRSA_Data_Changping_20130301-20170228.csv"
df = pd.read_csv(file)

#Break the pm2.5 data up into four ranges
def label_PM25 (row):
    if row['PM2.5'] < 0:
        return np.nan
    elif 0 <= row['PM2.5'] <= 50:
        return "Low"
    elif 51 <= row['PM2.5'] <= 100:
        return "Moderate"
    elif 101 <= row['PM2.5'] <= 150:
        return "High"
    elif row['PM2.5'] > 150:
        return "Very High"

df['PM2.5_label'] = df.apply (lambda row: label_PM25(row), axis=1)
#print(df['PM2.5_label'])

#Create counts for all 16 transitions
CountLowLow = 0
CountLowModerate = 0
CountLowHigh = 0
CountLowVeryHigh = 0
CountModerateLow = 0
CountModerateModerate = 0
CountModerateHigh = 0
CountModerateVeryHigh = 0
CountHighLow = 0
CountHighModerate = 0
CountHighHigh = 0
CountHighVeryHigh = 0
CountVeryHighLow = 0
CountVeryHighModerate = 0
CountVeryHighHigh = 0
CountVeryHighVeryHigh = 0


#Track the number of each of the 16 types of transitions
indxPM25Label = df.columns.get_loc("PM2.5_label")
for i in range(1,df.shape[0]-1):
    if df.iat[i,indxPM25Label]=='Low':
        if df.iat[i+1,indxPM25Label]=='Low':
            CountLowLow+=1
        elif df.iat[i+1,indxPM25Label]=='Moderate':
            CountLowModerate+=1
        elif df.iat[i+1,indxPM25Label]=='High':
            CountLowHigh+=1
        elif df.iat[i+1,indxPM25Label]=='Very High':
            CountLowVeryHigh+=1
    elif df.iat[i,indxPM25Label]=='Moderate':
        if df.iat[i+1,indxPM25Label]=='Low':
            CountModerateLow+=1
        elif df.iat[i+1,indxPM25Label]=='Moderate':
            CountModerateModerate+=1
        elif df.iat[i+1,indxPM25Label]=='High':
            CountModerateHigh+=1
        elif df.iat[i+1,indxPM25Label]=='Very High':
            CountModerateVeryHigh+=1
    elif df.iat[i,indxPM25Label]=='High':
        if df.iat[i+1,indxPM25Label]=='Low':
            CountHighLow+=1
        elif df.iat[i+1,indxPM25Label]=='Moderate':
            CountHighModerate+=1
        elif df.iat[i+1,indxPM25Label]=='High':
            CountHighHigh+=1
        elif df.iat[i+1,indxPM25Label]=='Very High':
            CountHighVeryHigh+=1
    elif df.iat[i,indxPM25Label]=='Very High':
        if df.iat[i+1,indxPM25Label]=='Low':
            CountVeryHighLow+=1
        elif df.iat[i+1,indxPM25Label]=='Moderate':
            CountVeryHighModerate+=1
        elif df.iat[i+1,indxPM25Label]=='High':
            CountVeryHighHigh+=1
        elif df.iat[i+1,indxPM25Label]=='Very High':
            CountVeryHighVeryHigh+=1

#Calculate the probabilities for each transition
ProbLowLow = CountLowLow / (CountLowLow + CountLowModerate + CountLowHigh + CountLowVeryHigh)
ProbLowModerate = CountLowModerate / (CountLowLow + CountLowModerate + CountLowHigh + CountLowVeryHigh)
ProbLowHigh = CountLowHigh / (CountLowLow + CountLowModerate + CountLowHigh + CountLowVeryHigh)
ProbLowVeryHigh = CountLowVeryHigh / (CountLowLow + CountLowModerate + CountLowHigh + CountLowVeryHigh)
ProbModerateLow = CountModerateLow / (CountModerateLow + CountModerateModerate + CountModerateHigh + CountModerateVeryHigh)
ProbModerateModerate = CountModerateModerate / (CountModerateLow + CountModerateModerate + CountModerateHigh + CountModerateVeryHigh)
ProbModerateHigh = CountModerateHigh / (CountModerateLow + CountModerateModerate + CountModerateHigh + CountModerateVeryHigh)
ProbModerateVeryHigh = CountModerateVeryHigh / (CountModerateLow + CountModerateModerate + CountModerateHigh + CountModerateVeryHigh)
ProbHighLow = CountHighLow / (CountHighLow + CountHighModerate + CountHighHigh + CountHighVeryHigh)
ProbHighModerate = CountHighModerate / (CountHighLow + CountHighModerate + CountHighHigh + CountHighVeryHigh)
ProbHighHigh = CountHighHigh / (CountHighLow + CountHighModerate + CountHighHigh + CountHighVeryHigh)
ProbHighVeryHigh = CountHighVeryHigh / (CountHighLow + CountHighModerate + CountHighHigh + CountHighVeryHigh)
ProbVeryHighLow = CountVeryHighLow / (CountVeryHighLow + CountVeryHighModerate + CountVeryHighHigh + CountVeryHighVeryHigh)
ProbVeryHighModerate = CountVeryHighModerate / (CountVeryHighLow + CountVeryHighModerate + CountVeryHighHigh + CountVeryHighVeryHigh)
ProbVeryHighHigh = CountVeryHighHigh / (CountVeryHighLow + CountVeryHighModerate + CountVeryHighHigh + CountVeryHighVeryHigh)
ProbVeryHighVeryHigh = CountVeryHighVeryHigh / (CountVeryHighLow + CountVeryHighModerate + CountVeryHighHigh + CountVeryHighVeryHigh)

#Create the probability matrix
transition_prob = {'Low': {'Low': ProbLowLow, 'Moderate': ProbLowModerate, 'High': ProbLowHigh, 'VeryHigh': ProbLowVeryHigh},
                   'Moderate': {'Low': ProbModerateLow, 'Moderate': ProbModerateModerate, 'High': ProbModerateHigh, 'VeryHigh': ProbModerateVeryHigh},
                   'High': {'Low': ProbHighLow, 'Moderate': ProbHighModerate, 'High': ProbHighHigh, 'VeryHigh': ProbHighVeryHigh},
                   'VeryHigh': {'Low': ProbVeryHighLow, 'Moderate': ProbVeryHighModerate, 'High': ProbVeryHighHigh, 'VeryHigh': ProbVeryHighVeryHigh}}

PM25_chain = MarkovChain(transition_prob=transition_prob)


# I rounded to four so that all 16 transition arrows appear in the drawing
P = np.array([[round(ProbLowLow,4),round(ProbLowModerate,4),round(ProbLowHigh,4),round(ProbLowVeryHigh,4)],
              [round(ProbModerateLow,4),round(ProbModerateModerate,4),round(ProbModerateHigh,4),round(ProbModerateVeryHigh,4)],
              [round(ProbHighLow,4),round(ProbHighModerate,4),round(ProbHighHigh,4),round(ProbHighVeryHigh,4)],
              [round(ProbVeryHighLow,4),round(ProbVeryHighModerate,4),round(ProbVeryHighHigh,4),round(ProbVeryHighVeryHigh,4)] ]) # Transition matrix

#Assign all four nodes
mc = MarkovChainDraw(P, ['Low','Moderate','High','Very High'])

#Draw the Markov Model
mc.draw()