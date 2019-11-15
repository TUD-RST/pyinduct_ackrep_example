"""
This example implements the introductory backstepping example
4.1 from [KristicEtAl08].

References:

    .. [KristicEtAl08]
        Miroslav Krstic, Andrey Smyshlyaev
        Boundary control of PDEs: a course on backstepping designs
        Society for Industrial and Applied Mathematics
        Advances in Design and Control, 2008

"""
import numpy as np
import pyinduct as pi
from matplotlib import pyplot as plt

from feedback import (AnalyticBacksteppingController,
                      ApproximatedBacksteppingController)
from simulation import ModalApproximation, FEMApproximation


def run():
    # number of eigenfunctions, used for control law approximation
    n_modal = 5

    # number of basis functions, used for system approximation
    n_fem_sim = 20
    n_modal_sim = 10

    # original system parameters
    a2 = 1
    a1 = 0
    a0 = 20
    orig_params = [a2, a1, a0, None, None]

    # target system parameters (controller parameters)
    a0_t = 0
    tar_params = [a2, a1, a0_t, None, None]

    # system/simulation parameters
    z_start = 0
    z_end = 1
    spat_bounds = (z_start, z_end)
    spatial_domain = pi.Domain(bounds=spat_bounds, num=100)

    # derive initial profile
    initial_data = np.random.rand(*spatial_domain.shape)
    initial_profile = pi.Function.from_data(spatial_domain,
                                            initial_data,
                                            domain=spat_bounds)
    # simulation domains
    temp_dom_open = pi.Domain(bounds=(0, .1), num=100)
    temp_dom_closed = pi.Domain(bounds=(0, .5), num=100)

    # scenarios to simulate
    fem_sys = FEMApproximation(orig_params, n_fem_sim, spat_bounds)
    modal_sys = ModalApproximation(orig_params, n_modal_sim, spatial_domain)

    # define dummy feedforward input
    ffwd_input = pi.ConstantTrajectory(np.array([0]))

    # define analytic backstepping controller
    analytic_cont = AnalyticBacksteppingController(spatial_domain,
                                                   orig_params,
                                                   fem_sys)

    # define approximated backstepping controller
    approx_cont_mod = ApproximatedBacksteppingController(orig_params,
                                                         tar_params,
                                                         n_modal,
                                                         spatial_domain,
                                                         modal_sys)
    approx_cont_fem = ApproximatedBacksteppingController(orig_params,
                                                         tar_params,
                                                         n_modal,
                                                         spatial_domain,
                                                         fem_sys)

    scenarios = {
        # "ol_modal": (temp_dom_open, ffwd_input, initial_profile, modal_sys),
        "open_loop_fem": (temp_dom_open, ffwd_input, initial_profile, fem_sys),
        # "cl_analytic_modal": (temp_dom_closed, analytic_cont, initial_profile, modal_sys),
        "cl_analytic_fem": (temp_dom_closed, analytic_cont, initial_profile, fem_sys),
        # "cl_approx_modal": (temp_dom_closed, approx_cont_mod, initial_profile, modal_sys),
        "cl_approx_fem": (temp_dom_closed, approx_cont_fem, initial_profile, fem_sys),
    }

    x_res = []
    u_res = []
    for name, params in scenarios.items():
        print("Simulating '{}'".format(name))
        temp_dom, u, init_p, meta_sys = params
        sys = meta_sys.get_system(u)
        ics = meta_sys.get_initial_state(init_p, u)
        t_sim, q_sim = pi.simulate_state_space(sys, ics, temp_dom)
        ed = meta_sys.get_results(q_sim, u, t_sim, spatial_domain, name)
        x_res.append(ed)
        u_sim = u.get_results(t_sim)
        u_res.append(u_sim)

    # visualization
    plots = []
    plots.append(pi.PgAnimatedPlot(x_res, replay_gain=1e-1, title="animation"))
    plots.append(pi.surface_plot(x_res, title="Surface plots"))

    fig, ax = plt.subplots()
    for x, u in zip(x_res, u_res):
        t_values = x.input_data[0]
        ax.plot(t_values, u, label=x.name)
    ax.legend()
    ax.grid()
    plots.append(fig)

    pi.show()
    pi.tear_down(tuple(), plots)


if __name__ == "__main__":
    np.random.seed(20190911)
    run()
