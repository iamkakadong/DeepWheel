from numpy.random import rand


def uniformInit(dim, lb, ub):
	return (rand(dim[0], dim[1]) + (ub + lb) / 2.0) * abs(ub - lb)
