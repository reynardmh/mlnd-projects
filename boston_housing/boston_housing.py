"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

################################
### ADD EXTRA LIBRARIES HERE ###
################################
from sklearn import metrics
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description="Without any arguments it will run the program to meet the project requirement. \n" +
    "This optional params lets you customize what you want to run, as well as chosing the performance metric you want to choose.")
parser.add_argument('-m', '--metric',
                   default='mean_absolute_error',
                   help='performance metric name. Valid values: mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score')
parser.add_argument('-mdt', '--max-depth-trends', action='store_true',
                   help='Run the program to iterate the GridSearchCV using the metric specified in --metric, and find the max_depth trend for that metric.')
parser.add_argument('-s', '--skip-graph-plots', action='store_true',
                   help='Run only the GridSearchCV and skip all the exploratory graph plottings.')
args = parser.parse_args()

def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    # Please calculate the following values using the Numpy library
    # Size of data (number of houses)?
    print "Number of data points (houses): {0}".format(housing_prices.size)

    # Number of features?
    print "Number of features: {0}".format(housing_features[0].size)

    # Minimum price?
    print "Minimum price: {0}".format(housing_prices.min())

    # Maximum price?
    print "Maximum price: {0}".format(housing_prices.max())

    # Calculate mean price?
    print "Mean price: {0}".format(housing_prices.mean())

    # Calculate median price?
    print "Median price: {0}".format(np.median(housing_prices))

    # Calculate standard deviation?
    print "Standard deviation: {0}".format(housing_prices.std())


def metric_method():
    valid_metrics = ['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2_score', 'explained_variance_score']
    default_metric = 'mean_absolute_error' # the default
    metric = default_metric
    if args.metric in valid_metrics:
        metric = args.metric

    return getattr(metrics, metric)
    # return metrics.mean_squared_error
    # return metrics.mean_absolute_error
    # return metrics.median_absolute_error
    # return metrics.r2_score
    # return metrics.explained_variance_score

# just a helper function to adjust greater_is_better params depending on which metric is used
def greater_is_better():
    m = metric_method().__name__
    return m == 'r2_score' or m == 'explained_variance_score'


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################

    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    return metric_method()(label, prediction)


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=7)

    return X_train, y_train, X_test, y_test


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.linspace(1, len(X_train), 50)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # 1. Find the best performance metric
    # should be the same as your performance_metric procedure
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

    scorer = metrics.make_scorer(metric_method(), greater_is_better=greater_is_better())

    # 2. Use gridearch to fine tune the Decision Tree Regressor and find the best model
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
    reg = grid_search.GridSearchCV(regressor, parameters, scoring=scorer)

    # Fit the learner to the training data
    print "Final Model: "
    print reg.fit(X, y)


    print "\n------ Result ------"
    print "Performance Metric: {0}".format(metric_method().__name__)
    print "Max depth: {0}".format(reg.best_params_['max_depth'])

    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    prediction_y = reg.predict([x])
    print "House: " + str(x)
    print "Prediction: " + str(prediction_y)

    # Check prediction with nearest neighbors y value
    neighbors = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    distances, indices = neighbors.kneighbors([x])
    # print distances
    neighbors_y = []
    for i in indices:
        neighbors_y.append(y[i])
    neighbors_y = np.array(neighbors_y)

    print "\n------ Comparison with nearest neighbors ------"
    print "Nearest neighbors mean price: {0}".format(neighbors_y.mean())
    print "Nearest neighbors std. dev: {0}".format(neighbors_y.std())
    print "Nearest neighbors price range: {0} to {1}".format(neighbors_y.min(), neighbors_y.max())

    # return the max_depth for running the GridSearchCV over many iterations to find the max_depth trends
    return reg.best_params_['max_depth']


def find_max_depth_trends(city_data):
    max_depths = []
    for i in range(0, 100):
        max_depths.append(fit_predict_model(city_data))

    max_depths = np.array(max_depths)
    print max_depths
    print "max depth trend for metric: {0}".format(metric_method().__name__)
    print "max depth mode: {0}".format(stats.mode(max_depths))
    print "max depth mean: {0}".format(max_depths.mean())
    print "max depth range: {0}-{1}".format(max_depths.min(), max_depths.max())
    print "max depth standard deviation: {0}".format(max_depths.std())


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    if args.max_depth_trends:
        find_max_depth_trends(city_data)
    else:
        if not args.skip_graph_plots:
            # Explore the data
            explore_city_data(city_data)

            # Training/Test dataset split
            X_train, y_train, X_test, y_test = split_data(city_data)

            # Learning Curve Graphs
            max_depths = [1,2,3,4,5,6,7,8,9,10]
            for max_depth in max_depths:
                learning_curve(max_depth, X_train, y_train, X_test, y_test)

            # Model Complexity Graph
            model_complexity(X_train, y_train, X_test, y_test)

        # Tune and predict Model
        fit_predict_model(city_data)


if __name__ == "__main__":
    main()
