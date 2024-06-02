import pandas as pd
import numpy as np
import random
from sklearn import svm
from prettytable import PrettyTable
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import model_selection

def clean(df):
    # Initial cleaning steps within data
    df = df.drop(['game_id', 'description', 'artist', 'publisher', 'compilation', 'designer', 'family', 'image', 'max_playtime', 'min_playtime', 'name', 'thumbnail', 'year_published'], axis=1)
    #df.dtypes
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
    df = df_combined

    # Removing data points with lower then 250 reviews
    minReviews = 250
    df = df[df["users_rated"] > minReviews]
    df = df.drop(columns=["users_rated"])

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

    # Drop Arrays
    df = df.drop(['category', 'mechanic'], axis=1)

    # Print top 5 Mechanics and categories
    print("\nTop 5 Mechanics:")
    print(top_5_mechanics)
    print("\nTop 5 Categories:")
    print(top_5_categories)
    print("\n")

    return df

def describeAnalytics(df):
    # Columns of interest
    columns_of_interest = ['max_players', 'min_players', 'playing_time', 'min_age']

    # initilize table
    stats_table = PrettyTable()
    stats_table.field_names = ['Descriptive Statistics', 'Max Players', 'Min Players', 'Average Play Time', 'Min Age'] # Declare column names

    # Calculate the mean for the specified columns
    means = round(df[columns_of_interest].mean(), 3)
    # add to table
    stats_table.add_row(["Mean", means.iloc[0], means.iloc[1], means.iloc[2], means.iloc[3]])

    # Calculate the median for the specified columns
    median = round(df[columns_of_interest].median(), 3)
    # add to table
    stats_table.add_row(["Median", median.iloc[0], median.iloc[1], median.iloc[2], median.iloc[3]])

    # Calculate the standard deviation for the specified columns
    std_devs = round(df[columns_of_interest].std(),3)
    # add to table
    stats_table.add_row(["Standard Deviation", std_devs.iloc[0], std_devs.iloc[1], std_devs.iloc[2], std_devs.iloc[3]])

    # Calculate the quartiles for the specified columns
    quartiles = round(df[columns_of_interest].quantile([0.25, 0.5, 0.75]),3)
    # add to table
    for quartile in [0.25, 0.5, 0.75]:
        quartile_values = quartiles.loc[quartile].tolist()
        name = str(int(quartile * 100)) + "th quartile"
        stats_table.add_row([name, quartile_values[0],  quartile_values[1],  quartile_values[2],  quartile_values[3]])

    print(stats_table)

def runModel(df):
    y = df.pop("average_rating")
    X = df

    # Split data for testing and training
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.20, random_state = 1, shuffle = True)

    # Parameter tuning with GridSearchCV to find best parameters
    #parameters = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    #svc = svm.SVR()
    #clf = GridSearchCV(svc, parameters, cv=5)
    #clf.fit(X_train, y_train)
    #print(clf.best_params_)
    # Use the best estimator found
    #regr = clf.best_estimator_
    #y_pred = regr.predict(X_test)

    # Train model
    y_test = y_test.values
    regr = svm.SVR(kernel = "rbf", gamma = 0.2)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    # Calculating and printing errors
    error = abs(y_test - y_pred)
    df_describe = pd.DataFrame(error)
    df_describe.describe()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("\nMean Absolute Error:", mae)
    print("Mean Squared Error:", mse)

    # Print them to visually see how they match
    error_table = PrettyTable() # Initialize Table
    error_table.field_names = ["Y Pred", "Y True", "ERROR"] # Declare column names
    for i in range(len(y_pred)):
        error_table.add_row([f"{y_pred[i]}",f"{y_test[i]}", error[i]])
        if i == 10: # Print only 1st 10
            break

    # See them Side By Side
    print(error_table)
    print("\n")
    return regr

def generateRandomGame(df, randomSeed, numGames):
    maxes = df.describe().loc["max"]
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
    gameDict = {}

    for collumn in df.columns:
        randDistIndex = int(random.random() * numGames)

        if(collumn in numberCols and limitAllVars):
            gameDict[collumn] = [int(dists[collumn][randDistIndex])]
        else:
            gameDict[collumn] = [int(random.random() * maxes[collumn])] #0's Basically

    if(limitNumCats):
        numMech = mechDist[randDistIndex]
        numCats = cataDist[randDistIndex]
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
            else:
                gameDict[collumn] = [int(random.random() * 2)]
    collumns = df.columns

    for i, mech in enumerate(mechs):
        gameDict[collumns[mech + 5]] = 1

    for i, cat in enumerate(cats):
        gameDict[collumns[cat + 5 + totalMechs]] = 1

    gameDF = pd.DataFrame.from_dict(gameDict)
    return gameDF

def printGame(bestGame, regr):
    gameInfo = bestGame.iloc[:, : 5]
    print(gameInfo)

    mechanics = bestGame.iloc[:, 5 : 5+51]
    mechanics

    categories =  bestGame.iloc[:, 5+51+0: ]
    categories

    listOfMechaincs = np.where(mechanics.values[0] == 1)
    listOfMechaincs = mechanics.columns.values[listOfMechaincs[0]]
    print(listOfMechaincs)

    listOfCategories = np.where(categories.values[0] == 1)
    listOfCategories = categories.columns.values[listOfCategories[0]]
    print(listOfCategories)

    rating = regr.predict(bestGame)
    print(f"Rating: {rating}")

def isReasonableGame(game: pd.DataFrame) -> bool:
    for i, column in enumerate(game.columns):
        row = game.iloc[0]

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
        addNei = game.copy(deep=True)
        row = addNei.iloc[0]
        row[column] += 1
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
        addNei = game.copy(deep=True)
        row = addNei.iloc[0]
        row[column] += 1
        if(isReasonableGame(addNei)):
            yield addNei

        subNei = game.copy(deep=True)
        row = subNei.iloc[0]
        row[column] -= 1
        if(isReasonableGame(subNei)):
            yield subNei


def scoreState(state, regr) -> float:
    return regr.predict(state)

def hillClimbingStep(currentGame: pd.DataFrame, regr):
    currentScore = scoreState(currentGame, regr)
    currentBestNeighbor = None

    for neighbor in generatorNeighbors(currentGame):

        neighborScore = scoreState(neighbor, regr)
        if(neighborScore >= currentScore):
            return neighbor

    if(type(currentBestNeighbor) == type(None)):
        return currentGame #currentScore
    else:
        return currentBestNeighbor

def main():
    randomSeed = 2
    random.seed(randomSeed)
    np.random.seed(randomSeed)
    df = pd.read_csv("board_games.csv")

    df = clean(df)
    describeAnalytics(df)
    regr = runModel(df)

    numGames = 100
    restarts = 10
    finalBestGame = None
    finalBestScore = -1

    for j in range(restarts):
        randomSeed = j
        currentGame = generateRandomGame(df, randomSeed, numGames)

        steps = 1000
        lastScore = None
        nextGameScore = None
        for i in range(0,steps):
            nextGame = hillClimbingStep(currentGame, regr)

            lastScore = nextGameScore
            nextGameScore = scoreState(nextGame, regr)
            if(lastScore == nextGameScore):
                break
            '''
            if(nextGame == currentGame):
                break #Early
            #'''
            currentGame = nextGame

        sc = scoreState(currentGame, regr)
        if(sc > finalBestScore):# and sc < 7.75):
            finalBestScore = sc
            finalBestGame = currentGame

        print(f"Game Score At Local Minima: {sc}, MaxScore {finalBestScore}")

    # Prints best game created
    print("\nBest Game Created")
    printGame(finalBestGame, regr)

main()
