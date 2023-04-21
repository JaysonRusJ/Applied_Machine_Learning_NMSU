# CS 487 - project PS5
# April 20th, 2023
#

import RandomForestRegressor.RFR_rossmann as RFR_rossmann
import RandomForestRegressor.RFR_walmart as RFR_walmart
import RNN_function as RNN
import FNN_function as FNN
import LinearRegression.linear_reg as lr

# take in user input for model
print("Type digit to select model:")
print("Linear Regression - 1\nRandom Forest Regressor - 2")
model = input("FNN - 3\nRNN - 4\n")

# take in user input for dataset
print("Type digit to selcet dataset:")
dataset = input("Walmart - 1\nRossmann - 2\n")

if (dataset != "1" and dataset != "2"):
    print("Invalid Input")
    exit(0)

# run linear regression
if (model == "1"):
    if (dataset == "1"):
        lr.linear_reg_walamrt()
    elif (dataset == "2"):
        lr.linear_reg_rossmann()
    
# run Random Forest Regressor
elif (model == "2"):

    if (dataset == "1"):
        RFR_walmart.RFR_walmart()
    elif (dataset == "2"):
        RFR_rossmann.RFR_rossmann()

# run CNN
elif (model == "3"):

    FNN.run_function(dataset)

# run RNN
elif (model == "4"):

    RNN.run_function(dataset)

# invalid input
else:
    print("Invalid Input")

