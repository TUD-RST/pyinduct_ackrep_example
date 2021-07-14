"""
Dummy file to do ackrep's work
"""
from problem import ProblemSpecification
from solution import solve

if __name__ == "__main__":
    pc = ProblemSpecification()
    sol = solve(pc)
    res = pc.evaluate_solution(sol)
    assert res.success
