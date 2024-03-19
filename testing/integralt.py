"""
This module is dedicated to the generator functions of the testing
phases for the comprehensive analysis of PCI approximations
"""

from pandas import DataFrame
from numpy import arange
from itertools import product as c_product

from testing.metrics import exec_time
from mathematics.pci import PCI


def generate_data(
        function,
        initial_value,
        final_value,
        step
) -> DataFrame:
    """
    Create two datasets, the set X serving as input for the function,
    and the outputs generated by it grouped in set Y.
    Because the sets X and Y cannot be continuous,
    a parameter 'step' is established to discretize the values within the interval.

    :param function: The function to related X set with Y set
    :param initial_value: The initial value of the X range set
    :param final_value: The final value of the Y range set
    :param step: The step between each value in X set

    :return: Dataframe containing two sets
    """
    inputs = [x for x in arange(initial_value, final_value, step)]
    outputs = [function(x) for x in inputs]

    rows = zip(inputs, outputs)

    rtn = DataFrame(rows, columns=["x", "y"])

    return rtn


@exec_time
def uniform_data_range(
        df: DataFrame,
        function,
        offset_range: list,
        rounder_range: list,
        show_progress=True,
        force_train=False):
    """
    Generate predictions for all possible combinations using the
    value ranges of 'offset' and 'rounder,' as well as intermediate values for each dataset.

    Params
    ---------
    :param df: This is a DataFrame that will be used to make tests

    :param function:  This parameter should be a function or an
    executable that returns a real value for the 'x' input.
    This real value will be compared with the approximate value
    to calculate the approximation error.

    :param offset_range: It is a list of values that represents all
    offsets that will be used to train the PCI system for all approximations.

    :param rounder_range: It is a list of values that represents all-rounders
    that will be used to train the PCI system for all aproximations.

    :param show_progress: If set to True, print a loading bar that shows
    the progress of the data approximations. If set to False, this loading bar is not shown

    :param force_train: If set to True force the system to retrain
    for each approximated value (does not use the effective range)
    """

    # If show_progress is True, import sys.stdout to
    # show loading bar
    from sys import stdout

    # Initialize new DataFrame to return it
    rtn_df = DataFrame()

    # Initialize a new list to store the
    # values that will be used as inputs
    # to make approximations with the PCI system
    inputs = list()

    # Extract the values from the 'x'
    # column of the DataFrame
    x_vals = df["x"].to_list()

    # Get the count of input values
    length = len(x_vals)

    # The values with which the PCI system's ability
    # to make predictions will be evaluated will not
    # be the exact values in the 'x' column of the dataset.
    # Instead, intermediate values will be used as it is
    # estimated that these carry a greater margin of error.
    # To create this set of evaluation inputs, all values from
    # the 'x' column of the initial dataset will be considered,
    # and the average of each contiguous set of values
    # will be calculated
    for i, x in enumerate(x_vals):

        # If index is out of index range from values list
        if i + 1 >= length:
            break  # break loop

        inputs.append((x + x_vals[i + 1]) / 2)

    # To avoid using three nested for loops, we use the Cartesian product
    # to calculate all possible combinations for different values of
    # 'offset,' 'rounder,' and 'x'.
    product = list(c_product(offset_range, rounder_range, inputs))

    # iteration count
    iters = len(product)

    print(f"\n\nElements count {iters:,}...\n\n")

    for i, element in enumerate(product):
        # For each possible combination of (offset, rounder, x),
        # we will generate a PCI approximation for the trio of values.

        if show_progress:
            # If the 'show_progress' variable was set
            # to true, display a loading bar

            cur = i / iters
            bar = "=" * int(50 * cur)
            spaces = " " * (50 - len(bar))
            stdout.write(f"\r\tProcess: [{bar}{spaces}] {int(cur * 100)}% - {i:,} of {iters:,}")
            stdout.flush()

        # Get real val evaluating in real function
        real_val = function(element[2])

        # Get approximate value evaluating in PCI system
        psys = PCI(df, offset=element[0], rounder=element[1])
        if not force_train:
            approx_val = psys.predict(element[2])
        else:
            approx_val = psys.force_predict(element[2])
        # Struct to data frame
        rtn_df = rtn_df._append(
            {
                "offset": element[0],
                "rounder": element[1],
                "x": element[2],
                "Real value": real_val,
                "Approx value": approx_val,
                "Dynamic function": f"\"{str(psys.dynamic_sp)}\"",
                "Static function": f"\"{str(psys.static_sp)}\""
            },
            ignore_index=True

        )

    return rtn_df
