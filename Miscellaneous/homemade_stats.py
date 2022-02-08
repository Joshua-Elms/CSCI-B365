from math import sqrt

def cov(x: list|tuple, y: list|tuple, r: int = 9) -> float: 
    """
    Calculate sample covariance matrix for vectors x and y (must be same length)
    r is the desired numbers of digits after decimal to round to
    """
    try: 
        N = max(len(x), len(y))
        x_mean = sum(x) / N
        y_mean = sum(y) / N
        cov = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(N)]) / (N - 1)
        return round(cov, r)

    except IndexError: 
        raise ValueError("Vectors must be of same length")

    except TypeError:
        raise TypeError("Vectors must only contain ints or floats")


def corr(x: list|tuple, y: list|tuple, r: int = 9) -> float:
    """
    Calculates the correlation between vectors x and y (must be same length)
    Calls cov(), std()
    Output float bounded by [-1, 1] inclusive
    """
    return round(cov(x, y) / (std(x) * std(y)), r)

def var(x: list|tuple, r: int = 9) -> float:
    """
    Calculate population variance matrix for vector x
    Calls cov()
    r is the desired numbers of digits after decimal to round to
    """
    return cov(x, x, r)


def std(x: list|tuple, r: int = 9) -> float: 
    """
    Calculate standard deviation of x
    Calls var()
    r is the desired numbers of digits after decimal to round to
    """
    return round(sqrt(var(x)), r)
