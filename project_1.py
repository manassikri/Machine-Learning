"""
A simple and a very easy way project to understand the linear regression model of machine learning with this basic example.
We have taken two numpy arrays x_values and y_values.In this example we train our model using these arrays(for every vaule of x we have a subsequent value in y).
The linear regression model follows the equation of simple line y=mx+b, were
y= output
x= input
b= bias
m= slope/gradient

Step 1->Import all the necessary libraries
Step 2->Import the dataset, here we made our own for simplicity and used two arrays as stated above.
Step 3->Calculate m and and b using the following formulas
        m = ((x̅ * y̅) - x̅y̅) / ((x̅)²-(x̅²))
        b = y̅ - mx̅

        Here bar above the variable indicates mean or average.

Step4 ->After calculating the values of m and b, we can put the same values in the equation y=mx+b and predict the value of y for the given value of x.

It is not mandatory that with this algorithm every time the values predicted will be right, there is always a possibility that the predicted value is not right or it has some error as compared to the actual value.
Hence we estimate the score or percentage/ probability of the right occurance of the value.

For this we find the R-square value using the following formula.
r^2 = 1 - ((Squared Error of Regression Line) / (Squared Error of y Mean Line))

More thevalue is close to  1 more accurate is our line.

At last we plot the graph to visualize the the points and see the occurance of predicted points.

These arenot theonly steps which are followed for the correct estimation of the predicted values. The error has to be reduced,for that the correct way is to use gradientdescent algorithm which keeps on itrating untill it finds the most appropriate value of m and n forwhich we will getthe least error.
Will upload another detailed example of machine leaning project using gradient descent algorithm.
"""

from statistics import mean #mean is used to calculate the mean value of all the the values in the array
import numpy as np #to form a numpy array
import matplotlib.pyplot as plt #to plot the graph atlast to visualize the points
from matplotlib import style


x_values = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float64) #we form a numpy array for input variable
y_values = np.array([1,4,1,6,4,7,4,6,10,8], dtype=np.float64) #this is the array of output variable

def best_fit_line(x_values,y_values): #this function is used to calculate the values of m and b
    m = (((mean(x_values)*mean(y_values)) - mean(x_values*y_values)) /
         ((mean(x_values)*mean(x_values)) - mean(x_values*x_values)))

    b = mean(y_values) - m*mean(x_values)

    return m, b

m, b = best_fit_line(x_values, y_values) 
print("regression line: " + "y = " + str(round(m,2)) + "x + " + str(round(b,2)) )


x_prediction = 15
y_prediction = (m*x_prediction)+b #finding the output/ predicted value for the input variable
print("predicted coordinate: (" + str(round(x_prediction,2)) + ", " + str(round(y_prediction,2)) + ")")

regression_line = [(m*x)+b for x in x_values] 

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig)) # helper function to return the sum of the distances between the two y values squared

def r_squared_value(ys_orig,ys_line):
    squared_error_regr = squared_error(ys_orig, ys_line) # squared error of regression line
    y_mean_line = [mean(ys_orig) for y in ys_orig] # horizontal line (mean of y values)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line) # squared error of the y mean line
    return 1 - (squared_error_regr/squared_error_y_mean)

r_squared = r_squared_value(y_values, regression_line)
print("r^2 value: " + str(r_squared))

style.use('seaborn')

plt.title('Linear Regression')
plt.scatter(x_values, y_values,color='#5b9dff',label='data')
plt.scatter(x_prediction, y_prediction, color='#fc003f', label="predicted")
plt.plot(x_values, regression_line, color='000000', label='regression line')
plt.legend(loc=4)
plt.savefig("graph.png")
plt.show()
