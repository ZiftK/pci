# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:46 2023

@author: ZiftK
"""

import mathematics.aop as aop
import mathematics.dfop as dfop

from numpy import array, matrix, linalg, round, delete, arange

from matplotlib import pyplot as plt

from testing.metrics import exec_time

from numpy import array
from numpy import arange

import time


class SolvePackage:
    """
    The Resolution Package is a class designed to encapsulate a range of
    data along with its corresponding limits and values used in the PCI approximation.
    Essentially, it is a class that enables PCI to locate the values of its dynamic or static data.

    Properties
    ---------

    df -> It refers to the feeding DataRange. From this DataRange,
    the data for the approximation will be taken.

    le -> Effective Lower Limit [PCI Method].

    ue -> Effective Upper Limit [PCI Method].

    coef -> Coefficients obtained within the interval defined
    by the effective lower limit and the effective upper limit [PCI Method].

    exp -> Exponents obtained within the interval defined
    by the effective lower limit and the effective upper limit [PCI Method].

    """

    def __init__(self, data: dfop.DataFrame):

        # data range object
        self.__dr = dfop.DataRange(data)

        # effective limits
        self.__le = None  # effective lower limit
        self.__ue = None  # effective upper limit

        # coefficients to save data range solution
        self.__coefficients = None
        # exponentes to save data range solution
        self.__exp = None

    # hd: Aux methods

    def is_inside_ef(self, point, column_name: str):
        """
        If point is inside effective range return true, else return false
        """
        try:  # try check if inside effective range
            # is inside effecitve limits
            c1 = point >= self.dr.get_value(column_name, self.__le)
            c2 = point <= self.dr.get_value(column_name, self.__ue)
            return c1 and c2
        except TypeError:
            # if effective limits are null, set condition to false
            return False
        except KeyError:
            return False

    def extract_ef_df(self):
        """
        Uses the effective lower and upper limits to extract
        the data frame within the effective range.
        """
        return self.dr.extract_df(self.__le, self.__ue)

    # hd: Train methods

    def train(self, point, offset, rounder):
        """
        Train system to selected solve package around specified point
        """

        # The train process take place in four stages:
        # calculating effective limits
        # calculating exponents, 
        # solving coefficients, 
        # and cleaning coefficients

        # Because the system can be trained in the dynamic or static range,
        # and each of these ranges contains its effective range,
        # SolvePackages are used to encapsulate all the variables
        # and functionalities of each super range (static and dynamic).
        # This way, we can independently control in each training phase in which
        # super range the system is being trained.

        self.__calc_ef_limits(point, offset)  # calculating effective limits
        self.__calc_exp()  # calculating exponents
        self.__solve(rounder)  # solving coefficients
        self.__clear()  # cleaning coefficients

    def __calc_ef_limits(self, point, offset):
        """
        Calculate the effective limits for the SolvePackage using the specified point
        """

        # The first step is to locate the pivot within the dataset,
        # which is simply the value within the dataset that is
        # closest to the point you want to approximate
        val = self.__dr.get_near_value("x", point)

        # Since the effective limits correspond to indices within
        # the dataset, the pivot must also be translated into an
        # index to locate it within the dataset
        val_indx = self.__dr.get_index("x", val)

        # The lower limit should not be less than zero, as it is
        # the minimum allowed value for an index. Therefore,
        # if the offset exceeds the range of values established
        # in the dataset, the effective range will be shortened,
        # and the effective lower index will be set to zero by default
        self.__le = max(0, val_indx - offset)

        # The same applies to the upper effective limit, as if the offset
        # from the pivot exceeds the range of values in the dataset,
        # the effective upper limit will be set to the maximum allowed
        self.__ue = min(val_indx + offset, self.__dr.rows_count() - 1)

    def __calc_exp(self):
        """
        Calculate the exponents used in the solution polynomial
        """

        # The exponents are calculated using the length of the effective 
        # range as a parameter. A list of exponents is computed,
        # where each value represents the exponent of a term in the final 
        # polynomial (it represents the dimension of the monomial)

        # The super range contained in the assigned SolvePackage is used,
        # allowing us to obtain the exponents within the specifically
        # defined effective range for the point to be approximated.
        # If the point is within the static range, the static SolvePackage
        # will be used; otherwise, the dynamic one will be used

        self.__exp = [n for n in range(0, len(self.extract_ef_df()))]

    def __solve(self, rounder):
        """
        Approximate the coefficients of the solution polynomial
        """
        # To approximate the coefficients, it is necessary to solve a 
        # matrix (n, n), where 'n' is the number of data points used
        # for the approximation. The first step is to define this matrix
        m = list()

        # We subtract the effective data frame from the SolvePackage to
        # be used and evaluate at each of the 'x' column values in their 
        # respective rows.
        for x in self.extract_ef_df()["x"]:
            m.append(aop.valpow(x, self.__exp))

        # ______ SOLVE ______

        # Define 'm' as a NumPy matrix object
        m = matrix(m)

        # Solve the matrix using the 'y' column of the effective data frame
        # as the expansion vector of the matrix
        self.__coefficients = linalg.solve(m, array(self.extract_ef_df()["y"]))

        # Round each polynomial coefficient using the rounder value
        self.__coefficients = round(self.__coefficients, rounder)

    def __clear(self):
        """
        Delete Monomials with negligible coeficients from solution
        """

        # In this list, the indices of each negligible coefficient
        # will be stored to delete them from the coefficients list
        del_index = list()

        # get index of negligible coeficients
        # iterate throught each round coeficient and get its index
        # for delete to polynomial
        for index, coef in enumerate(self.__coefficients):

            # add index with negligible coeficients
            if coef == 0:
                del_index.append(index)

        # This is done to generate polynomials as small as possible or to reduce noise
        self.__coefficients = delete(self.__coefficients, del_index)
        self.__exp = delete(self.__exp, del_index)

    def update_out_data(self, point, step=0.5):
        """
        Inserts a value outside the original data range,
        offset by a value defined by 'step' towards the approximation point
        """

        # pass the 'step' parameter through the 'abs' function to avoid
        # using negative values (this would result in incorrect 
        # calculations as it would change the direction 
        # of the offset for extrapolation)
        step = abs(step)

        # Within this function, we only need to make one check.
        # To avoid defining too much code within conditionals
        # and with the intention of not repeating the same code too much,
        # we generate three variables that will be set within the 
        # conditionals to generalize the process using them.
        # Thus, the extrapolation will change direction with
        # the change of these variables

        # *last inside value
        # It refers to the nearest value to the desired extrapolation 
        # within the dynamic range 
        in_val = None

        # * insert index
        # It refers to the index where the new data will be inserted
        indx = None

        if point < self.__dr.get_value("x", 0):

            # If the extrapolation point is to the left of the dynamic dataset,
            # we need to change the extrapolation direction. 
            # We achieve this by changing the sign of the 'step' to negative,
            # to decrease the extrapolation value
            step *= -1

            # In order to approximate a value outside the dynamic range, 
            # we need to know the last value within the range in the direction
            # of the extrapolation
            in_val = self.__dr.get_value("x", int(self.__le))

            # To insert the new extrapolated value within the dynamic range,
            # it is necessary to know on which side of the range it will be inserted.
            # We do this by establishing the insertion index. In this case,
            # the index is the effective lower limit, as the data is to the left
            # of the dynamic range
            indx = self.__le

        else:
            # It has already been verified that the point 
            # is not within the dynamic range, so if it is 
            # also not found on the left, the only possible 
            # option is that it is located on the right.

            # In order to approximate a value outside the dynamic range, 
            # we need to know the last value within the range in the direction
            # of the extrapolation
            in_val = self.__dr.get_value("x", self.__ue)

            # To insert the new extrapolated value within the dynamic range,
            # it is necessary to know on which side of the range it will be inserted.
            # We do this by establishing the insertion index. In this case,
            # the index is the effective upper limit, as the data is to the right
            # of the dynamic range
            indx = self.__ue

        # approximate the value outside the dynamic range using the dynamic SolvePackage
        out_val = in_val + step
        extrapol_val = self.apply_pol(out_val)

        # insert value in selected index
        self.__dr.soft_insert({"x": out_val, "y": extrapol_val}, indx)

    def apply_pol(self, point):
        """
        Apply polinomial solution to specyfic point
        """
        # * make prediction
        # pow point to each value in exponents array
        a = aop.valpow(float(point), self.__exp)
        # multiply each value in solve point exponents
        # to each value in solve coefficients
        pdct = aop.amult(self.__coefficients, a)
        # return sum of array
        return sum(pdct)

    # hd: Properties

    @property
    def le(self):
        return self.__le

    @property
    def ue(self):
        return self.__ue

    @property
    def dr(self):
        return self.__dr

    @property
    def coef(self):
        return self.__coefficients

    @property
    def exp(self):
        return self.__exp

        # hd: Object override

    def __str__(self):
        """
        String object representation
        """
        string = ""

        if self.__coefficients is None:
            return string

        for index, coef in enumerate(self.__coefficients):
            string += f"{self.__coefficients[index]}*x^{self.__exp[index]}"
            string += "" if index == len(self.__coefficients) - 1 else "+"

        return string.replace("e", "*10^")


class PCI:
    """
    PCI (Polynomial with Coefficients Indeterminate) is a system
    that approximates real data without apparent correlation through
    a Taylor series, a polynomial. This method requires the tabulation of data.

    Limitations
    -----------

    - Currently, the PCI system only works with single-variable correlation.
    Its performance for multi-variable correlation is planned for the future.

    - The PCI system uses a large amount of memory to approximate values,
    so setting a high offset and consequently a wide effective range can lead
    to memory overflow.

    Properties
    ---------

    data -> The 'data' entered into the PCI system must be in tabular form.
    Currently, a data frame or a path to extract it is supported.
    If another type of data or an incorrect path is entered, an exception will be raised

    offset -> The 'offset' is the value used to control the amplitude of the effective range.
    For practical and technical reasons, the entered value will be doubled to generate the final amplitude.
    **Important** This value will affect the accuracy of the final result.

    rounder -> The 'rounder' value controls how many decimals are considered
    for calculations. Higher decimal precision incurs more computational cost.
    Once the final polynomial is obtained, all coefficients rounded to zero
    will be eliminated using 'rounder' as a parameter.
    **Important** This value will affect the accuracy of the final result.

    """

    def __init__(self, data, **kwargs) -> None:

        # We initialize a SolvePackage for the data related 
        # to the dynamic range and another one for the data
        # related to the static range
        self.__ssp = SolvePackage(data)
        self.__dsp = SolvePackage(data)

        # set default offset
        self.__offset = kwargs.get("offset", 5)
        # set default rounder
        self.__rounder = kwargs.get("rounder", 15)

        # Testing variables
        self.__tst = None
        self.__tst2 = None

    def __train(self, point, solve_package: SolvePackage):
        """

        """
        solve_package.train(point, self.__offset, self.__rounder)

    def force_predict(self, point):
        """
        Train PCI system around point using it as pivot, forcing re-training
        """
        point = round(point, 5)

        # check if point is inside static effective data range or
        # dynamic effective data range

        # check if inside static effective data range
        if self.__ssp.dr.is_inside(point, "x"):

            self.__ssp.train(point, self.__offset, self.__rounder)
            return self.__ssp.apply_pol(point)

        # check if inside dynamic effective data range
        elif self.__dsp.dr.is_inside(point, "x"):
            self.__dsp.train(point, self.__offset, self.__rounder)
            return self.__dsp.apply_pol(point)

        pass

    def predict(self, point, ep_step=0.5):
        """
        Get aprox value from trained system
        """

        point = round(point, 5)

        # check if point is inside static effective data range or
        # dynamic effective data range

        # check if inside static effective data range
        if self.__ssp.is_inside_ef(point, "x"):
            return self.__ssp.apply_pol(point)

        # check if inside dynamic effective data range
        elif self.__dsp.is_inside_ef(point, "x"):
            return self.__dsp.apply_pol(point)

        # it is inside static limits?
        elif self.__ssp.dr.is_inside(point, "x"):

            # train system inside static limits
            self.__train(point, self.__ssp)

            # apply polinomial solution to static solve package
            return self.__ssp.apply_pol(point)

        # It has been previously verified that the point to approximate
        # is outside the dynamic effective range and the
        # static effective range. It has also been confirmed to be
        # outside the static range, so the only available options are
        # that it is within the dynamic range or it is outside all ranges.

        # check if point is inside dynamic range
        in_dynamic = self.__dsp.dr.is_inside(point, "x")

        if in_dynamic:  # * if point is in dynamic range

            # apply polinomial solution to dynamic solve package
            return self.__dsp.apply_pol(point)

        # extrapolation loop
        while not self.__dsp.is_inside_ef(point, "x"):
            self.__dsp.train(point, self.__offset, self.__rounder)

            self.__dsp.update_out_data(point, ep_step)

        return self.__dsp.apply_pol(point)

    def normalize(self, step=None, norm_round=5):
        """
        Fill in the empty spaces in the dataset by interpolating entries in multiples of the
        'step' from the initial value to the final value, generating approximate outputs for each entry

        Params
        -------
        step -> The variable 'step' represents the normalization step. If a 'step'
        of 1 is used, it will cause the dataset to be normalized to multiples of 1 for the inputs.

        norm_round -> It is the number of decimals to round in the normalization approximations
        """
        # TODO: Document normalize function and refactor. Fix normalize bug

        # If the 'step' variable is not specified, 
        # it will be taken as the average difference 
        # between input samples in the original dataset.
        if step is None:
            step = self.__dsp.dr.get_mean_diff("x")

        # cur_val' represents the current interpolated input value.
        # Before the first iteration, we calculate this value by
        # taking the first entry in our dataset and adding the value
        # of 'step'. 'cur_val' is rounded because adding 'step' does
        # not yield an exact value, but a very close, almost infinitesimally close,
        # approximate value. However, when comparing this value with the current 
        # entries in the dataset, a difference is obtained, and therefore,
        # the algorithm enters already established approximations.
        # Rounding solves this issue

        cur_val = round(self.__dsp.dr.get_value("x", 0) + step, norm_round)

        # The new values must be inserted at a specific index, for which we use the variable 'indx'
        indx = 1

        while True:

            # If the value to approximate is outside the ranges
            # of the original dataset, we are not interested 
            # in continuing the approximation, so we break the loop.
            if not self.__dsp.dr.is_inside(cur_val, "x"):
                break

            # We are interested in adding values to the dataset, 
            # but if the 'step' is a multiple of any entry in 
            # the original dataset, we want to omit that data, 
            # as it is already present in the original set

            if not self.__dsp.dr.is_in_column(cur_val, "x"):
                # interpolate new value
                pdct_val = self.predict(cur_val)

                # add relation to data set
                self.__dsp.dr.soft_insert({"x": cur_val, "y": pdct_val}, indx)

            # After each iteration, we add 'step' to the 
            # current value, and round it using the specified rounder.
            # We also increment the position index by one
            cur_val += step
            cur_val = round(cur_val, norm_round)
            indx += 1

        # hd:                    Properties

    @property
    def static_sp(self):
        return self.__ssp

    @property
    def dynamic_sp(self):
        return self.__dsp


class GEnP:

    def __init__(self, df, **kwargs):

        # Output data frame
        self.__df: dfop.DataFrame

        # Groups of data frames
        self.__groups: list = []

        if type(df) == str:
            self.__df = dfop.read_csv(df)
        if type(df) == dfop.DataFrame:
            self.__df = df

    def split(self, *keys):

        # Group by keys
        grouped = self.__df.groupby(*keys)

        # Convert each group into data frame
        self.__groups = [group for _, group in grouped]


if __name__ == "__main__":
    pass
