import numpy as np
from dragonfly.utils.ancillary_utils import get_list_of_floats_as_str
from electrolyte_kernel import ElectrolyteKernel, ELECTROLYTE_KER_DIM
from dragonfly.utils.option_handler import load_options
from dragonfly.gp import gp_core, euclidean_gp 
from dragonfly.utils.reporters import get_reporter
from dragonfly.gp.euclidean_gp import get_euclidean_integral_gp_kernel

_DFLT_KERNEL_TYPE = 'electrolyte'
electrolyte_gp_args =   gp_core.mandatory_gp_args + \
                        euclidean_gp.basic_euc_gp_args + \
                        euclidean_gp.matern_gp_args

class ElectrolyteGPFitter(gp_core.GPFitter):
    def __init__(self, X, Y, options=None, reporter=None):
        self.dim = 10
        reporter = get_reporter(reporter)
        print("[{}]options: {}".format(__file__, options))
        if options is None:
            options = load_options(electrolyte_gp_args, "ElectrolyteGPFitter", reporter=reporter)
        super(ElectrolyteGPFitter, self).__init__(X, Y, options, reporter)

    def _child_set_up(self):
        if self.options.kernel_type == "default":
            #self.kernel_type = _DFLT_KERNEL_TYPE
            self.options.kernel_type = _DFLT_KERNEL_TYPE
        if not self.options.kernel_type == "electrolyte":
            raise NotImplementedError("Only electrolyte kernel has been implemented")
        # 1 & 2 mean value and noise variance - done in gp_core.GPFitter
        # set up based on electrolyte kernel
        # scale and bandwidth parameter
        self.scale_log_bounds = [np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)]
        self.param_order.append(["scale", "cts"])
        X_std_norm = np.linalg.norm(self.X, 'fro') + 1e-4
        single_bandwidth_log_bounds = [np.log(0.01 * X_std_norm), np.log(10 * X_std_norm)]
        if self.options.use_same_bandwidth:
            self.bandwidth_log_bounds = [single_bandwidth_log_bounds]
            self.param_order.append(["same_dim_bandwidths", "cts"])
        else:
            self.bandwidth_log_bounds = [single_bandwidth_log_bounds] * self.dim
            for _ in range(self.dim):
                self.param_order.append(["dim_bandwidths", "cts"])
        self.cts_hp_bounds += [self.scale_log_bounds] + self.bandwidth_log_bounds

        # if nu is negative (to be tuned), then test on several half integers
        if self.options.matern_nu < 0:
            self.dscr_hp_vals.append([0.5, 1.5, 2.5])
            self.param_order.append(["nu", "dscr"])
        
    def _child_build_gp(self, mean_func, noise_var, gp_cts_hps, gp_dscr_hps,
                      other_gp_params=None, *args, **kwargs):
        """
        gp_cts_hps contains hyperparameter: 
        scale, bandwidths (*dim if different bandwidths), (nu if provided and positive)
        """
        # build electrolyte kernel
        # first scale
        scale = np.exp(gp_cts_hps[0])
        gp_cts_hps = gp_cts_hps[1:]
        # then bandwidth(s)
        if self.options.use_same_bandwidth:
            ke_dim_bandwidths = [np.exp(gp_cts_hps[0])] * self.dim
            gp_cts_hps = gp_cts_hps[1:]
        else:
            ke_dim_bandwidths = np.exp(gp_cts_hps[:self.dim])
            gp_cts_hps = gp_cts_hps[self.dim:]
        # nu if positive
        if self.options.matern_nu < 0:
            matern_nu = gp_dscr_hps[0] 
            gp_dscr_hps = gp_dscr_hps[1:]
        else:
            matern_nu = self.options.matern_nu
        # build kernel from hyper params
        # now the continuous and discrete hyperparameters would not contain scale, bandwidth(s), (nu)
        electrolyte_kernel = ElectrolyteKernel(nu=matern_nu, scale=scale, dim_bandwidths=ke_dim_bandwidths)
        # generate gp from kernel
        ret_gp = ElectrolyteGP(self.X, self.Y, electrolyte_kernel, mean_func, noise_var, *args, **kwargs)
        return ret_gp, gp_cts_hps, gp_dscr_hps
        
class ElectrolyteGP(gp_core.GP):
    def __init__(self, X, Y, kernel, mean_func, noise_var,
                kernel_hyperparams=None, build_posterior=True, reporter=None):
        
        if isinstance(kernel, str):
            kernel = ElectrolyteKernel( nu=kernel_hyperparams["nu"],
                                        scale=kernel_hyperparams["scale"],
                                        dim_bandwidths=kernel_hyperparams["dim_bandwidths"] )
        super(ElectrolyteGP, self).__init__(X, Y, kernel, mean_func, noise_var, build_posterior, reporter)

    def _child_str(self):
        """ String representation for child GP. """
        ke_str = "electrolyte(nu={:.1f})".format(self.kernel.hyperparams["nu"]) # showing nu
        ke_str = ke_str + get_list_of_floats_as_str(self.kernel.hyperparams["dim_bandwidths"]) # showing dim_bandwidths
        mean_str = "mu(0)={:.3f}".format(self.mean_func([np.zeros(ELECTROLYTE_KER_DIM), ])[0])
        scale_str = "scale={:.3f}".format(self.kernel.hyperparams["scale"])
        return "{}, {}, {}".format(scale_str, ke_str, mean_str)