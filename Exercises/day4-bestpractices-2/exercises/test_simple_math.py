# test for simple_math.py 
import simple_math

# the name of the testing function should  
# also start with test_ 
def test_sum(): 
    assert simple_math.simple_add(1,-1) == 0

def test_sub(): 
    assert simple_math.simple_sub(1,-1) == 2

def test_mult(): 
    assert simple_math.simple_mult(10,1) == 10

def test_div(): 
    assert simple_math.simple_div(99,99) == 1


def test_poly_first(): 
    assert simple_math.poly_first(1,2,3) == 5


def test_poly_second(): 
    assert simple_math.poly_second(1,2,3,4) == 9





