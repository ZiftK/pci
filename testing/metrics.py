import time


def exec_time(func):
    """
    Print the execution time of function passed as a parameter
    """

    def wrapper(*args, **kwargs):
        # The pre-execution time is obtained using the system clock
        intl = time.time()
        # The return value of the function to be evaluated is stored
        rtn = func(*args, **kwargs)
        # The post-execution time is obtained using the system clock
        fnl = time.time()
        # The execution time is calculated
        tt = fnl - intl
        # Show message
        print(f"\n\nThe function {func.__name__} was executed in {tt:.6f} seconds.")
        # Return value
        return rtn

    # Return value
    return wrapper
