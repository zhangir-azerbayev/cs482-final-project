import numpy as np
from codeproject.execution import check_correctness

def programs_to_passed_lst(programs, tests, show=False): 
    to_execute = [program + "\n" + tests for program in programs]

    if show: 
        for x in to_execute: 
            print("#"*20)
            print(x)

    passed_lst = [check_correctness(x, 1)["passed"] for x in to_execute]

    return passed_lst

def pass_at_k(lst, k): 
    """
    lst: Boolean list 
    k: value of pass@k to calculate. 
    """
    n = len(lst)
    c = sum(lst)
    if n - c < k: return 1.0 
    return 1.0 - np.prod(1.0 - k / np.arange(n-c+1, n+1))
