import pandas as pd

import numpy as np
import random

randomSeed = 1
random.seed(randomSeed)
np.random.seed(randomSeed)

# drive.mount("/drive")
#drive.mount('/content/gdrive')

df = pd.read_csv("board_games.csv")


# Initial cleaning steps within data

df = df.drop(['game_id', 'description', 'artist', 'publisher', 'compilation', 'designer', 'family', 'image', 'max_playtime', 'min_playtime', 'name', 'thumbnail', 'year_published'], axis=1)
df.dtypes

#Change Expaion To Boolean
df["expansion"] = df["expansion"].notnull().astype(int)

#Split Mechanic Collum On Each Type
catSeries = df['mechanic']
catDF = catSeries.str.get_dummies(sep = ",")

#Combine
df_combined = pd.merge(df, catDF, left_index=True, right_index=True, how='inner')

#Split Categoty Collum On Each Type
catSeries = df['category']
catDF = catSeries.str.get_dummies(sep = ",")

#Combine
df_combined = pd.merge(df_combined, catDF, left_index=True, right_index=True, how='inner')


# Drop Arrays
# df = df_combined.drop(['category', 'mechanic'], axis=1)

df = df_combined

# Removing data points with lower then 250 reviews
minReviews = 250
df = df[df["users_rated"] > minReviews]
# df = df.drop(columns=["users_rated"])
df = df.drop(columns=["users_rated"])
# print(df.shape)
# df

# Count the occurrences of each mechanic
df_exploded = df['mechanic'].str.split(',').explode()
mechanic_counts = df_exploded.value_counts()
# Get the top 5 mechanics
top_5_mechanics = mechanic_counts.head(5)

# Count the occurrences of each category
df_exploded = df['category'].str.split(',').explode()
category_count = df_exploded.value_counts()
# Get the top 5 categories
top_5_categories = category_count.head(5)

# # Drop Arrays
df = df.drop(['category', 'mechanic'], axis=1)
# df.dtypes
print(df.shape)
df

# descriptive statistics before cleaning
from prettytable import PrettyTable
import numpy as np

# Columns of interest
columns_of_interest = ['max_players', 'min_players', 'playing_time', 'min_age']

# initilize table
stats_table = PrettyTable()
stats_table.field_names = ['Descriptive Statistics', 'Max Players', 'Min Players', 'Average Play Time', 'Min Age'] # Declare column names

# Calculate the mean for the specified columns
means = round(df[columns_of_interest].mean(), 3)
# add to table
stats_table.add_row(["Mean", means[0], means[1], means[2], means[3]])

# Calculate the median for the specified columns
median = round(df[columns_of_interest].median(), 3)
# add to table
stats_table.add_row(["Median", median[0], median[1], median[2], median[3]])

# Calculate the standard deviation for the specified columns
std_devs = round(df[columns_of_interest].std(),3)
# add to table
stats_table.add_row(["Standard Deviation", std_devs[0], std_devs[1], std_devs[2], std_devs[3]])

# Calculate the quartiles for the specified columns
quartiles = round(df[columns_of_interest].quantile([0.25, 0.5, 0.75]),3)
# add to table
for quartile in [0.25, 0.5, 0.75]:
    quartile_values = quartiles.loc[quartile].tolist()
    name = str(int(quartile * 100)) + "th quartile"
    stats_table.add_row([name, quartile_values[0],  quartile_values[1],  quartile_values[2],  quartile_values[3]])

print(stats_table)
print("\nTop 5 Mechanics:")
print(top_5_mechanics)
print("\nTop 5 Categories:")
print(top_5_categories)

#import matplotlib.pyplot as plt
#import missingno as msno
#msno.matrix(df)
#plt.show()
#df.isna().sum()

# X = df2.iloc[:,:-1] #X
y = df.pop("average_rating")
X = df
X.head()
# y.head()

from sklearn import model_selection

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.20, random_state = 1, shuffle = True)
# X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.10, random_state = 1, shuffle = True)

X_train.head()

from sklearn import svm
from prettytable import PrettyTable
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

# used for finding best parameters

# Parameter tuning with GridSearchCV
#parameters = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
#svc = svm.SVR()
#clf = GridSearchCV(svc, parameters, cv=5)
#clf.fit(X_train, y_train)
#print(clf.best_params_)
# Use the best estimator found
#regr = clf.best_estimator_
#y_pred = regr.predict(X_test)
y_test = y_test.values

regr = svm.SVR(kernel = "rbf", gamma = 0.2)#, C=100,gamma=0.0001)
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)


error = abs(y_test - y_pred)
# print(y_test.values , y_pred)

print(f"Median: {np.median(error)}")
df_describe = pd.DataFrame(error)
df_describe.describe()


# Errors
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)


## Print them to visually see how they match
example_table = PrettyTable() # Initialize Table
example_table.field_names = ["Y Pred", "Y True", "ERROR"] # Declare column names
for i in range(len(y_pred)):
    example_table.add_row([f"{y_pred[i]}",f"{y_test[i]}", error[i]])
    if i == 10:
      break

## See them Side By Side
print(example_table)

# import random

maxes = df.describe().loc["max"]
# maxes = df["playing_time"].median()

#exDist = np.random.default_rng().binomial(n= ((5.44)+4)/2 * val, p= 1/val, size=n)

#MED: 4.0,10.0,2.0,45.0

#'''

# np.random.seed(0)

numGames = 100

bestGame = None
bestRating = 0

totalMechs = 51
totalCats = 83

val = 10
mechDist = np.random.default_rng(seed = randomSeed).binomial(n= 2.274 * val, p= 1/val, size=numGames)
val = 5
cataDist = np.random.default_rng(seed = randomSeed).binomial(n= 2.603 * val, p= 1/val, size=numGames)


val = 10
numberCols = ["max_players", "min_age", "min_players", "playing_time"]
dists = {}
for numberCol in numberCols:
    dists[numberCol] = np.random.default_rng(seed = randomSeed).binomial(n=((df[numberCol].mean())+df[numberCol].median())/2 * val, p= 1/val, size=numGames)


limitAllVars = True
limitNumCats = True or limitAllVars

def generateRandomGame():
    gameDict = {}

    for collumn in df.columns:
        randDistIndex = int(random.random() * numGames)

        if(collumn in numberCols and limitAllVars):
            gameDict[collumn] = [int(dists[collumn][randDistIndex])]
            # print(collumn)
        else:
        # print(collumn)
            gameDict[collumn] = [int(random.random() * maxes[collumn])] #0's Basically
        # print(int(random.random() * maxes[collumn]))

    # print(mechDist[i], cataDist[i])
    if(limitNumCats):
        numMech = mechDist[randDistIndex]
        numCats = cataDist[randDistIndex]
        # print(f"COUNTS: {numMech}, {numCats}")
        mechs = []
        cats = []

        for i in range(numMech):
            mech = int(random.random() * totalMechs)
            while(mech in mechs):
                mech = int(random.random() * totalMechs)
            mechs.append(mech)

        for i in range(numCats):
            cat = int(random.random() * totalCats)
            while(cat in cats):
                cat = int(random.random() * totalCats)
            cats.append(cat)
    else:
        for collumn in df.columns:
            if(collumn in numberCols):
                pass
                # print(collumn)
            else:
                gameDict[collumn] = [int(random.random() * 2)]

    # print(mechs, cats)

    collumns = df.columns


    for i, mech in enumerate(mechs):
        # print(f"MECH: {collumns[mech + 5]}")
        gameDict[collumns[mech + 5]] = 1

    for i, cat in enumerate(cats):
        # print(f"CAT: {collumns[cat + 5 + totalMechs]}")
        gameDict[collumns[cat + 5 + totalMechs]] = 1

    gameDF = pd.DataFrame.from_dict(gameDict)
    # print(gameDF)
    return gameDF



for c in range(numGames):

    gameDF = generateRandomGame()
    # printGame(gameDF)

    gameRating = regr.predict(gameDF)
    print(gameRating)
    if(gameRating > bestRating):
        bestRating = gameRating
        bestGame = gameDF

#'''

# maxes


#Gamma Allowed:
#Unvaried Input = Varied Score (lower) (6.88)
#Varied Games = Constant Score (Low) (6.803 ==)


#Gamma Taken Out
#Unvaried Input = Low Score (6.86)
#Varied Games = Higher Score (7.92)


#1.38, 5.8, 6,4, 6.94, 9.00

def printGame(bestGame):
    gameInfo = bestGame.iloc[:, : 5]
    print(gameInfo)

    mechanics = bestGame.iloc[:, 5 : 5+51]
    mechanics

    categories =  bestGame.iloc[:, 5+51+0: ]
    categories

    listOfMechaincs = np.where(mechanics.values[0] == 1)
    listOfMechaincs = mechanics.columns.values[listOfMechaincs[0]]
    print(listOfMechaincs)


    # print(categories.columns.values, categories.values[0])
    listOfCategories = np.where(categories.values[0] == 1)
    listOfCategories = categories.columns.values[listOfCategories[0]]
    print(listOfCategories)

    rating = regr.predict(bestGame)
    print(f"Rating: {rating}")

printGame(bestGame)
printGame(pd.DataFrame(X_test.iloc[4]).transpose())
print(type(bestGame), type(X_test))

def isReasonableGame(game: pd.DataFrame) -> bool:
    for i, column in enumerate(game.columns):
        # print(i,column)
        row = game.iloc[0]
        # print(row)
        # print(f"GAME: {game[column]} ENDGAME")
        # print(f"ROW: {row[column]} ENDROW")
        if(i == 0):
            #max_players
            if(row[column] > 11 or row[column] < 2):
                return False
        elif(i == 1):
            #min_age
            if(row[column] > 18 or row[column] < 5):
                return False
        elif(i == 2):
            #min_players
            if(row[column] > 5 or row[column] < 1):
                return False
        elif(i == 3):
            #playing_time
            if(row[column] > 300 or row[column] < 10):
                return False
        else:
            #Mechanics and Categories
            if(row[column] != 0 and row[column] != 1):
                return False

    numMechs = game.iloc[0,5:56].sum()
    numCats  = game.iloc[0,56:].sum()

    valid = numMechs > 0 and numMechs < 6 and numCats > 0 and numCats < 6 and game["max_players"][0] > game["min_players"][0]
    return valid



def gameNeighbors(game: pd.DataFrame) -> list[pd.DataFrame]:
    neighbors = []

    for i, column in enumerate(game.columns):
        # print(i,column)
        # row = game.iloc[0]
        # print(row)
        # print(f"GAME: {game[column]} ENDGAME")
        # print(f"ROW: {row[column]} ENDROW")

        addNei = game.copy(deep=True)
        row = addNei.iloc[0]
        row[column] += 1
        # print(row, game.iloc[0])
        if(isReasonableGame(addNei)):
            neighbors.append(addNei)

        subNei = game.copy(deep=True)
        row = subNei.iloc[0]
        row[column] -= 1
        if(isReasonableGame(subNei)):
            neighbors.append(subNei)

    return neighbors

def generatorNeighbors(game: pd.DataFrame):
    for i, column in enumerate(game.columns):
        # print(i,column)
        # row = game.iloc[0]
        # print(row)
        # print(f"GAME: {game[column]} ENDGAME")
        # print(f"ROW: {row[column]} ENDROW")

        addNei = game.copy(deep=True)
        row = addNei.iloc[0]
        row[column] += 1
        # print(row, game.iloc[0])
        if(isReasonableGame(addNei)):
            # neighbors.append(addNei)
            yield addNei

        subNei = game.copy(deep=True)
        row = subNei.iloc[0]
        row[column] -= 1
        if(isReasonableGame(subNei)):
            # neighbors.append(subNei)
            yield subNei


def scoreState(state) -> float:
    return regr.predict(state)

def hillClimbingStep(currentGame: pd.DataFrame):
    currentScore = scoreState(currentGame)
    #print(f"CS: {currentScore}")

    currentBestNeighbor = None

    # neighbors = gameNeighbors(currentGame)
    # for i in range(len(neighbors)):
    #     neighbor = neighbors[i]

    for neighbor in generatorNeighbors(currentGame):

        neighborScore = scoreState(neighbor)
        # print(f"NP: {neighbor}, NS: {neighborScore}")
        # print(f"NS: {neighborScore}, CS: {currentScore}")
        if(neighborScore >= currentScore):
            return neighbor
            #CheckAll
            currentScore = neighborScore
            currentBestNeighbor = neighbor
    # if(currentBestInd == -1):
    if(type(currentBestNeighbor) == type(None)):
        #print("NO MORE STEPS TO TAKE")
        return currentGame #currentScore
    else:
        return currentBestNeighbor

# print(bestGame)
# gameNeighbors(bestGame)
# isReasonableGame(pd.DataFrame(X_test.iloc[4]).transpose())

# gen = generatorNeighbors(bestGame)
# gen = gameNeighbors(bestGame)
# print(next(gen))
# print(next(gen))
# print(gen)

# bestGame["max_players"][0]
# bestGame.iloc[0,5:56].sum()
# bestGame.iloc[0,56:].sum()

restarts = 10
maxScore = -1

finalBestGame = None
finalBestScore = -1

for j in range(restarts):

    currentGame = generateRandomGame()
    # printGame(currentGame)

    steps = 1000

    lastScore = None
    nextGameScore = None
    for i in range(0,steps):
        #print(f"CurrentPath: {currentPath}, Score = {g.pathCost(currentPath)}")
        nextGame = hillClimbingStep(currentGame)

        lastScore = nextGameScore
        nextGameScore = scoreState(nextGame)
        # print(f"Score: {nextGameScore}")
        if(lastScore == nextGameScore):
            break
        # if(i == 1):
            # printGame(nextGame)
        '''
        if(nextGame == currentGame):
            break #Early
        #'''
        currentGame = nextGame

    sc = scoreState(currentGame)
    # print(sc)
    # printGame(currentGame)
    if(sc > finalBestScore):# and sc < 7.75):
        finalBestScore = sc
        finalBestGame = currentGame

    print(f"Game Score At Local Minima: {sc}, MaxScore {finalBestScore}")


print(f"Final Game")
printGame(finalBestGame)
# print(f"Best Score: {finalBestScore}")

#Generator
#For Valid Game
#Max Players > min Players + 1
#Mechanics and Attributes Counts Capped

#Seems to always go to Max and mins
finalBestGame

##I Forgot About Expasion
##Top 50 - 10000 Games in dataset
