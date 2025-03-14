"""
A collection of simple math operations: Addition, subtraction, multiplication, division, first and second order polynomial equation
"""

def simple_add(a,b):
    """
    Adds two numbers.
    
    Parameters
    ----------
    a : float or int
        The first number.
    b : float or int
        The second number.
    
    Returns
    -------
    float or int
        The sum of `a` and `b`.
    """
    return a+b

def simple_sub(a,b):
    """
    Subtract two numbers.
    
    Parameters
    ----------
    a : float or int
        The first number.
    b : float or int
        The second number.
    
    Returns
    -------
    float or int
        The subtraction of `a` and `b`.
    """
    return a-b

def simple_mult(a,b):
    """
    Multiply two numbers.
    
    Parameters
    ----------
    a : float or int
        The first number.
    b : float or int
        The second number.
    
    Returns
    -------
    float or int
        The multiplication of `a` and `b`.
    """
    return a*b

def simple_div(a,b):
    """
    Diivde two numbers.
    
    Parameters
    ----------
    a : float or int
        The first number.
    b : float or int
        The second number.
    
    Returns
    -------
    float or int
        The division of `a` and `b`.
    """
    return a/b

def poly_first(x, a0, a1):
    """
    Returns the result of a first order polynomial equation
    where a0 and a1 are the coefficients and x the variable
    """
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    """
    Returns the result of a second order order polynomial equation
    where a0, a1 and a2 are the coefficients and x the variable
    """
    return poly_first(x, a0, a1) + a2*(x**2)

# Feel free to expand this list with more interesting mathematical operations...
# .....
