class ElectrolyteKernel(Kernel):
  def __init__(self, scalar, bias):
    """
    input domain is the cartesian product of multiple domains, where 
    each domain could be euclidean or discrete
    """
    super().__init__()
    # add hyper parameters here
    self.add_hyperparams(scalar, bias)

  def is_guaranteed_psd(self):
    """
    Whether the new kernel is gauranteed to be PSD
    """
    return False
  
  def _child_evaluate(self, X1, X2):
    """
    X1, X2's first dimension is the existence of the catalyst
    second dimension is the ordering, which only takes effect if first True
    K(x1, x2) = scalar * (x1[0] == x2[0])(x1[1] - x2[1]) +  bias
    """
    if len(X1 == 0) or len(X2 == 0): return np.zeros(len(X1), len(X2))
    X1scalar, X2scalar = X1[:, 0], X2[:, 0]
    X1order, X2order = X1[:, 1:], X2[:, 1:]
    same_setting = (X1scalar == X2scalar.transpose()) # (n, n)
    dist =  
    return np.exp(X1)

  def _child_gradient(self):
    pass

  def __str__(self):
    return "ElectrolyteKernel, Params: {}".format(self.hyperparams)
    
