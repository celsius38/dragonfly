import numpy as np
from argparse import Namespace
from dragonfly.opt import gpb_acquisitions
from dragonfly.opt.gp_bandit import GPBandit, get_all_gp_bandit_args
from dragonfly.utils.reporters import get_reporter
from dragonfly.utils.option_handler import load_options
from dragonfly.exd.exd_utils import get_euclidean_initial_qinfos
from electrolyte_gp import ElectrolyteGP, ElectrolyteGPFitter, electrolyte_gp_args

# Electrolyte GPBandit Class
# ================================================================
class Electrolyte_GPBandit(GPBandit):
  def __init__(self, func_caller, worker_manager, is_mf=False,
               domain_dist_computers=None, options=None, reporter=None):
    if options is None:
      reporter = get_reporter(reporter)
    if is_mf:
      raise NotImplementedError("Not Implemented for Electrolyte")
    else:
      all_args = get_all_electrolyte_gp_bandit_args(additional_args=None)
      options = load_options(all_args, reporter)
    self.domain_dist_computers = domain_dist_computers
    super(Electrolyte_GPBandit, self).__init__(func_caller, worker_manager, is_mf=is_mf,
                                               options=options, reporter=reporter)
  def _set_up_for_acquisition(self):
    """ Set up for acquisition """
    if self.options.acq == 'default':
      acq = "ei-ucb-ttei-ts" # use '-' separated list of options
    else:
      acq = self.options.acq
    self.acqs_to_use = [elem.lower() for elem in acq.split('-')]
    self.acqs_to_use_counter = {key: 0 for key in self.acqs_to_use}
    if self.options.acq_probs == 'uniform':
      self.acq_probs = np.ones(len(self.acqs_to_use)) / float(len(self.acqs_to_use))
    elif self.options.acq_probs == 'adaptive':
      self.acq_uniform_sampling_prob = 0.05
      self.acq_sampling_weights = {key: 1.0 for key in self.acqs_to_use}
      self.acq_probs = self._get_adaptive_ensemble_acq_probs()
    else:
      self.acq_probs = np.array([float(x) for x in self.options.acq_probs.split('-')])
    # normalize to have prob summed up to 1
    self.acq_probs = self.acq_probs / self.acq_probs.sum()
    assert len(self.acq_probs) == len(self.acqs_to_use)

  def _child_opt_method_set_up(self):
    self.options.init_method = self.options.euc_init_method
    self.add_gp = None
    self.req_add_gp = False

  def _get_non_mf_gp_fitter(self, reg_data, use_additive=False):
    options = Namespace(**vars(self.options))
    if use_additive:
      raise NotImplementedError("Additive GP not implemented for Electrolyte")
    return ElectrolyteGPFitter(reg_data[0], reg_data[1],
                               options=options, reporter=self.reporter)
  
  def _determine_next_query(self):
    """ Determine next point for evaluation"""
    curr_acq = self._get_next_acq()
    anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
    select_pt_func = getattr(gpb_acquisitions.asy, curr_acq)
    qinfo_hp_tune_method = self.gp_processor.hp_tune_method
    qinfo = Namespace(curr_acq=curr_acq, hp_tune_method=qinfo_hp_tune_method)
    next_eval_point = select_pt_func(self.gp, anc_data)
    qinfo.point = next_eval_point
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    curr_acq = self._get_next_acq()
    anc_data = self._get_ancillary_data_for_acquisition(curr_acq)
    select_pt_func = getattr(gpb_acquisitions.syn, curr_acq)
    qinfo_hp_tune_method = self.gp_processor.hp_tune_method
    next_batch_of_eval_points = select_pt_func(batch_size, self.add_gp, anc_data)
    qinfos = [Namespace(point=pt,
                        hp_tune_method=qinfo_hp_tune_method,
                        curr_acq=curr_acq) for pt in next_batch_of_eval_points]
    return qinfos
  
  def _get_initial_qinfos(self, num_init_evals):
    return get_euclidean_initial_qinfos(self.options.init_method,
                                        num_init_evals,
                                        self.domain.bounds)
    
  def _domain_specific_acq_opt_set_up(self):
    """ Set up acquisition optimisation for the child class. """
    if self.acq_opt_method.lower() in ['direct']:
      self._set_up_euc_acq_opt_direct()
    elif self.acq_opt_method.lower() in ['pdoo']:
      self._set_up_euc_acq_opt_pdoo()
    elif self.acq_opt_method.lower() == 'rand':
      self._set_up_euc_acq_opt_rand()
    else:
      raise NotImplementedError('Not implemented acquisition optimisation for %s yet.'%( \
                                self.acq_opt_method))

  # Any of these set up methods can be overridden by a child class -------------------
  def _set_up_euc_acq_opt_direct(self):
    """ Sets up optimisation for acquisition using direct/pdoo. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 1 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     1000, 3e4)

  def _set_up_euc_acq_opt_pdoo(self):
    """ Sets up optimisation for acquisition using direct/pdoo. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 2 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     2000, 6e4)

  def _set_up_euc_acq_opt_rand(self):
    """ Sets up optimisation for acquisition using random search. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 10 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)),
                                                     2000, 3e4)

# Electrolyte GP ends ======================================================
def get_all_electrolyte_gp_bandit_args(additional_args=None):
  if additional_args is None:
    additional_args = []
  return get_all_gp_bandit_args(additional_args) + electrolyte_gp_args
