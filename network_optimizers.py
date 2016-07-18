import theano
import theano.tensor as T
import numpy as np


def adam_loves_theano(inp_list, cost, param_list, cost_belonging=None,
                      alpha=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-7):
    """
    adam: adaptive... momentum???

    Parameters
    ----------
    inp_list: List of Theano variables
        Whatever non-parameter things are needed to do a training step
    cost: Theano variable or list of Theano variables
        Objective function(s) to minimize
    param_list: List of Theano variables
        The variables that are changed for optimization
    [cost_belonging]: {None}
        If cost is a single Theano variable, this is ignored.
        If cost is a list, cost_belonging is a list of length len(param_list) that says
        which cost each parameter belongs to.
            For example, cost_belonging[3] = 1 means that the parameter at param_list[3]
            is associated with the cost at cost[1]
        If cost is a list and cost_belonging is not specified, all parameters will be,
        by default, assocated with the first cost in the cost list
    [alpha]: {0.0001}
        Training parameter: learning rate
    [beta1]: {0.9}
        Training parameter: decay rate for momentum
    [beta2]: {0.999}
        Training parameter: decay rate for velocity
    [epsilon]: {1e-7}
        Training parameter: i dunno.
    
        

    Outputs
    -------
    f_adam_helpers (updates helpers)
    f_adam_train (uses updated helpers to update parameters in param_list)
    adam_param_list: List of the adam parameters (time, momentum, velocity)
    adam_hyperparam_list: List of the adam hyperparameters
    grads: list of the adam gradients
    """
    # Create 2 theano functions that will be called sequentially
    # The first one "updates" the shared variables that go into the calculation of the parameter update
    # The second one combines them into an update

    # Create the first function:
    # (These are going to be useful to precompute and store as a list):
    if type(cost) == list:
        # See if a cost_belonging was supplied
        if cost_belonging != None:
            # It was. Make sure it's valid.
            assert type(cost_belonging) == list
            assert len(cost_belonging) == len(param_list)
            assert max(cost_belonging) <= len(cost)
        else:
            # It wasn't. Use the default.
            cost_belonging = [0 for i in range(len(param_list))]

        # Compute gradients
        grads = [T.grad(cost[c], p) for c, p in zip(cost_belonging, param_list)]
        
    else:
        # Compute gradients with respect to the single objective
        grads = [T.grad(cost, p) for p in param_list]
        
    # Initialize the helper variables, one for each parameter (this will only happen once and doesn't affect updates)
    Ts = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
          for p, g in zip(param_list, grads)]  # t term in adam
    Ms = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
          for p, g in zip(param_list, grads)]  # m term in adam
    Vs = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
          for p, g in zip(param_list, grads)]  # v term in adam

    # Define parameter list of shared variables
    adam_param_list = [Ts, Ms, Vs]

    # Define shared variable of network hyperparameters
    adam_hyperparam_list = theano.shared(
        np.array([alpha, beta1, beta2, epsilon])
        .astype(theano.config.floatX)
    )

    # Define each of their update rules
    up_t = [(T_, T_+1 ) for T_ in Ts]
    up_m = [(M, adam_hyperparam_list[1]*M + (1 - adam_hyperparam_list[1])*g)
            for M, p, g in zip(Ms, param_list, grads)]
    up_v = [(V, adam_hyperparam_list[2]*V + (1 - adam_hyperparam_list[2])*(g**2))
            for V, p, g in zip(Vs, param_list, grads)]

    # Combine this into a full update list
    up_h = up_t + up_m + up_v

    # Create that first function
    f_adam_helpers = theano.function(inp_list, cost, updates=up_h, no_default_updates=True)

    # Create the second function (during training, this is called right after calling the first):
    # Compute, using the updated helper variables, the components of the parameter update equation
    # (updated by the call to f_adam_helpers, which will occurr during training)
    mHat = [m / (1-(adam_hyperparam_list[1]**t)) for m, t in zip(Ms, Ts)]
    vHat = [v / (1-(adam_hyperparam_list[2]**t)) for v, t in zip(Vs, Ts)]
    # Use them to update the parameters
    up_p = [(p, p - (adam_hyperparam_list[0]*mH / (T.sqrt(vH)+ adam_hyperparam_list[3])))
            for p, mH, vH in zip(param_list, mHat, vHat)]
    # Create your training function with this update
    f_adam_train = theano.function(inp_list, cost, updates=up_p, no_default_updates=False)

    return f_adam_helpers, f_adam_train, adam_param_list, adam_hyperparam_list, grads


def adadelta_fears_committment(inp_list, cost, param_list, mask_list, rho=.95, epsilon=1e-6):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    inp_list: List of Theano variables
        Whatever non-parameter things are needed to do a training step
    cost: Theano variable
        Objective fucntion to minimize
    param_list: List of Theano variables
        The variables that are changed for optimization
    [rho]: {0.95}
        Training parameter: decay rate
    [epsilon]: {1e-6}
        Training parameter: i dunno.

    Outputs
    -------
    train_adadelta: A function that takes the inputs in inp_list and runs these two guys (which are created below),
        f_adadelta_helpers (updates helpers)
        f_adadelta_train (uses updated helpers to update parameters in param_list)

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    ### = DESCRIPTION FROM LITERATURE

    # Initialize the helper variables, one for each parameter (this will only happen once and doesn't affect updates)
    grads = [T.grad(cost,p) for p in param_list]
    # Standard gradients: g_t
    zipped_grads = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
                    for p, g in zip(param_list, grads)]
    # Running expectation of squared update: E[ d[x]**2 ]_t
    running_up2 = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
                   for p, g in zip(param_list, grads)]
    # Running expectation of squared gradient: E[g**2]_t
    running_grads2 = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
                      for p, g in zip(param_list, grads)]

    # Initialize parameter list
    adadelta_param_list = [zipped_grads, running_up2, running_grads2]


    ### Compute Gradient: g_t
    # Update rule for shared variables in zipped_grads (they just equal variables in grads)
    zgup = [(zg, T.grad(cost, p)) for zg, p in zip(zipped_grads, param_list)]

    ### Accumulate Gradient: E[g**2]_t = rho * E[g**2]_t-1  +  (1-rho) * (g_t)**2
    # Update rule for shared variables in running_grads2
    rg2up = [(rg2, (rho * rg2 + (1-rho) * (T.grad(cost, p) ** 2))*m + (1-m)*rg2)
             for rg2, m, p in zip(running_grads2, mask_list, param_list)]

    # Function that, when called, applies the two above update rules
    # (during training, this is called, then f_update is)
    f_adadelta_helpers = theano.function(inp_list, cost, updates=zgup+rg2up)


    ### Compute Update: d[x]_t = - [ RMS(d[x])_t-1 / RMS(g)_t ] * g_t
    # Create symbolic variable out of zipped_grads, running_up2, and running_grads2 for each parameter
    updir = [-T.sqrt(ru2 + epsilon) / T.sqrt(rg2 + epsilon) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]

    ### Accumulate Update: E[ d[x]**2 ]_t = rho * E[ d[x]**2 ]_t-1  +  (1-rho) * (d[x]_t)**2
    # Update rule for ru2up (whatever that is)
    ru2up = [(ru2, m*(rho * ru2 + (1-rho) * (ud ** 2)) + (1-m)*ru2)
             for ru2, m, ud in zip(running_up2, mask_list, updir)]

    ### Apply Update: x_t+1 = x_t + d[x]_t
    # Final update rule for parameter, combining all that
    # weight_updates = [m*ud for m, ud in zip(mask_list, updir)]
    param_up = [(p, p + m*ud) for p, m, ud in zip(param_list, mask_list, updir)]
    # param_up = [(p, p + wu) for p, wu in zip(param_list, weight_updates)]

    # Function to actually update the parameters (as well as ru2up)
    f_adadelta_train = theano.function(inp_list, cost, updates=ru2up + param_up)

    # Combine these into a single function using this neat trick that Ari pointed out!
    # def train_adadelta( *args ):
    #     # Update helpers
    #     f_adadelta_helpers( *args )
    #     # Update parameters with updated helpers
    #     return f_adadelta_train( *args )

    return f_adadelta_helpers, f_adadelta_train, adadelta_param_list


def i_hate_SGD(inp_list, cost,param_list, alpha=0.01):
    """
    SGD: but why???

    Parameters
    ----------
    inp_list: List of Theano variables
        Whatever non-parameter things are needed to do a training step
    cost: Theano variable
        Objective fucntion to minimize
    param_list: List of Theano variables
        The variables that are changed for optimization
    [alpha]: {0.001}
        Training parameter: learning rate

    Outputs
    -------
    train_SGD: function
        Uses updated helpers to update parameters in param_list

    """
    # Define shared variable of network hyperparameter (learning rate)
    SGD_alpha = theano.shared(
        np.array([alpha])
        .astype(theano.config.floatX)
    )
    
    # This is so straightforward I should punch you if you don't understand.
    update_rules = [(p, p-T.grad(cost, p)*SGD_alpha.get_value()) for p in param_list]
    train_SGD = theano.function(inp_list, cost, updates=update_rules)
    # Did you get it? Because if not you deserve punches.
    return train_SGD, SGD_alpha


# def rmsprop(lr, tparams, grads, x, mask, y, cost):
#     """
#     A variant of  SGD that scales the step size by running average of the
#     recent step norms.

#     Parameters
#     ----------
#     lr : Theano SharedVariable
#         Initial learning rate
#     tpramas: Theano SharedVariable
#         Model parameters
#     grads: Theano variable
#         Gradients of cost w.r.t to parameres
#     x: Theano variable
#         Model inputs
#     mask: Theano variable
#         Sequence mask
#     y: Theano variable
#         Targets
#     cost: Theano variable
#         Objective fucntion to minimize

#     Notes
#     -----
#     For more information, see [Hint2014]_.

#     .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
#        lecture 6a,
#        http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
#     """

#     zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
#                                   name='%s_grad' % k)
#                     for k, p in tparams.iteritems()]
#     running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
#                                    name='%s_rgrad' % k)
#                      for k, p in tparams.iteritems()]
#     running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
#                                     name='%s_rgrad2' % k)
#                       for k, p in tparams.iteritems()]

#     zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
#     rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
#     rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
#              for rg2, g in zip(running_grads2, grads)]

#     f_grad_shared = theano.function([x, mask, y], cost,
#                                     updates=zgup + rgup + rg2up,
#                                     name='rmsprop_f_grad_shared')

#     updir = [theano.shared(p.get_value() * numpy_floatX(0.),
#                            name='%s_updir' % k)
#              for k, p in tparams.iteritems()]
#     updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
#                  for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
#                                             running_grads2)]
#     param_up = [(p, p + udn[1])
#                 for p, udn in zip(tparams.values(), updir_new)]
#     f_update = theano.function([lr], [], updates=updir_new + param_up,
#                                on_unused_input='ignore',
#                                name='rmsprop_f_update')

#     return f_grad_shared, f_update