from dragonfly.gp.kernel import MaternKernel
import numpy as np

ELECTROLYTE_KER_DIM = 10
class ElectrolyteKernel(MaternKernel):
  def __init__(self, nu=None, scale=None, dim_bandwidths=None):
    """
    input domain is the cartesian product of multiple domains, where
    each domain could be euclidean or discrete
    """
    super(ElectrolyteKernel, self).__init__(ELECTROLYTE_KER_DIM, nu, scale, dim_bandwidths)

  def _extract_feat(self, X):
    """
    extract feat from each X
    Params:
    - X: (N, 17) where 17 is split into (1,1,1,3,1,1,1,3,3)
    Return:
    - extracted: (N, 10)
    """
    return[[float(x[0]*x[4]), float(x[1]) * x[5], float(x[2]) * x[6],
            float(x[3][0]) * x[7][0], float(x[3][1]) * x[7][1],
            float(x[3][2]) * x[7][2], float(x[3][3]) * x[7][3],
            x[8][0], x[8][1], x[8][2]] for x in X]

  def _child_evaluate(self, X1, X2):
    """
    Params:
    - X1, X2: (N, 17) where 17 is split into (1,1,1,3,1,1,1,3,3)
    """
    X1 = self._extract_feat(X1)
    X2 = self._extract_feat(X2)
    return super(ElectrolyteKernel, self)._child_evaluate(X1, X2)

  def __str__(self):
    return "ElectrolyteKernel: " + super(ElectrolyteKernel, self).__str__()
