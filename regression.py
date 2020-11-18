import requests
import pandas as pd
import scipy
import numpy
import sys
from scipy import stats
import matplotlib.pyplot as plt


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    a = pd.read_csv('linreg_train.csv')
    x = a.columns.to_list()[1:]
    x = [float(i) for i in x]
    a = a.values.tolist()
    y = a[0][1:]
    #plt.plot(x, y, 'ro')
    #plt.show()
    slope, intc, a1, a2, a3 = stats.linregress(x, y)
    #print(slope, intc)
    ans = []
    for i in area:
        ans.append(slope*i+intc)
    return ans


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
