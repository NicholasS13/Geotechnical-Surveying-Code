import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import kv as besselk
from scipy.special import gamma
import warnings

def variogramfit(h, gammaexp, a0=None, c0=None, numobs=None, **kwargs):
    """
    Fit a theoretical variogram to an experimental variogram.
    
    Parameters:
    -----------
    h : array-like
        Lag distance of the experimental variogram
    gammaexp : array-like
        Experimental variogram values (gamma)
    a0 : float, optional
        Initial value for range parameter
    c0 : float, optional 
        Initial value for sill variance
    numobs : array-like, optional
        Number of observations per lag distance (used for weight function)
    
    Keyword Arguments:
    -----------------
    model : str, optional
        Variogram model type. Options are:
        Bounded models: 'blinear', 'circular', 'spherical' (default), 'pentaspherical'
        Unbounded models: 'exponential', 'gaussian', 'whittle', 'stable', 'matern'
        
    nugget : float, optional
        Initial value for nugget variance. Default is None (no nugget variance)
        
    plotit : bool, optional
        If True (default), plot experimental and theoretical variogram together
        
    solver : str, optional
        'minimize' (default): Use scipy.optimize.minimize with bounds
        'fmin': Use scipy.optimize.fmin (equivalent to MATLAB's fminsearch)
        
    weightfun : str, optional
        Weighting function for the least squares fit
        'none' (default): No weighting
        'cressie85': m(hi)/gammahat(hi)^2 as weights
        'mcbratney86': m(hi)*gammaexp(hi)/gammahat(hi)^3 as weights
        
    stablealpha : float, optional
        Parameter for the stable model. Default is 1.5.
        
    nu : float, optional
        Shape parameter for the Matern model. Default is 1.
    
    Returns:
    --------
    a : float
        Range parameter of the fitted variogram model
        
    c : float
        Sill parameter of the fitted variogram model
        
    n : float or None
        Nugget parameter (None if nugget is not applied)
        
    S : dict
        Dictionary with additional information about the fit
    
    Example:
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Load or create experimental variogram data
    >>> h = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> gammaexp = np.array([0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0, 1.0])
    >>> # Initial guesses
    >>> a0 = 10  # initial range
    >>> c0 = 1.0  # initial sill
    >>> # Fit variogram model
    >>> a, c, n, S = variogramfit(h, gammaexp, a0, c0, 
    ...                          model='spherical', 
    ...                          nugget=0.1, 
    ...                          plotit=True)
    """
    # Convert inputs to numpy arrays
    h = np.asarray(h).flatten()
    gammaexp = np.asarray(gammaexp).flatten()
    
    # Set default initial values if not provided
    if a0 is None:
        a0 = max(h) * 2/3
    
    if c0 is None:
        c0 = max(gammaexp)
    
    # Set default parameters
    params = {
        'model': 'spherical',
        'nugget': None,
        'plotit': True,
        'solver': 'minimize',
        'stablealpha': 1.5,
        'weightfun': 'none',
        'nu': 1
    }
    
    # Update with provided keyword arguments
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        else:
            warnings.warn(f"Unknown parameter: {key}")
    
    # Check if h and gammaexp are vectors and have the same size
    if h.size != gammaexp.size:
        raise ValueError("h and gammaexp must have the same size")
    
    # Remove nans
    nans = np.isnan(h) | np.isnan(gammaexp)
    if np.any(nans):
        h = h[~nans]
        gammaexp = gammaexp[~nans]
        if numobs is not None:
            numobs = np.asarray(numobs)
            numobs = numobs[~nans]
    
    # Check weight inputs
    if numobs is None:
        params['weightfun'] = 'none'
    
    # Set initial values for optimization
    if params['nugget'] is None:
        b0 = np.array([a0, c0])
        nugget = False
        funnugget = lambda b: 0
    else:
        b0 = np.array([a0, c0, params['nugget']])
        nugget = True
        funnugget = lambda b: b[2]
    
    # Define variogram model functions
    model = params['model'].lower()
    
    # Define model type and function
    if model == 'spherical':
        type = 'bounded'
        func = lambda b, h: b[1] * ((3*h/(2*b[0])) - 0.5*(h/b[0])**3)
    elif model == 'pentaspherical':
        type = 'bounded'
        func = lambda b, h: b[1] * (15*h/(8*b[0]) - 5/4*(h/b[0])**3 + 3/8*(h/b[0])**5)
    elif model == 'blinear':
        type = 'bounded'
        func = lambda b, h: b[1] * (h/b[0])
    elif model == 'circular':
        type = 'bounded'
        def func(b, h):
            # Safe implementation for circular model
            ratio = np.minimum(h/b[0], 1.0)  # Ensure ratio <= 1 for arccos
            return b[1] * (1 - (2/np.pi)*np.arccos(ratio) + 
                           (2*h)/(np.pi*b[0])*np.sqrt(np.maximum(0, 1-ratio**2)))
    elif model == 'exponential':
        type = 'unbounded'
        func = lambda b, h: b[1] * (1 - np.exp(-h/b[0]))
    elif model == 'gaussian':
        type = 'unbounded'
        func = lambda b, h: b[1] * (1 - np.exp(-(h**2)/(b[0]**2)))
    elif model == 'stable':
        type = 'unbounded'
        stablealpha = params['stablealpha']
        func = lambda b, h: b[1] * (1 - np.exp(-(h**stablealpha)/(b[0]**stablealpha)))
    elif model == 'whittle':
        type = 'unbounded'
        def func(b, h):
            # Safe implementation for Whittle model
            mask = h > 0
            result = np.zeros_like(h, dtype=float)
            result[mask] = b[1] * (1 - h[mask]/b[0] * besselk(1, h[mask]/b[0]))
            return result
    elif model == 'matern':
        type = 'unbounded'
        nu = params['nu']
        def func(b, h):
            # Safe implementation for Matern model
            mask = h > 0
            result = np.zeros_like(h, dtype=float)
            result[mask] = b[1] * (1 - (1/((2**(nu-1))*gamma(nu))) * 
                                 (h[mask]/b[0])**nu * besselk(nu, h[mask]/b[0]))
            return result
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Check for zero distances which can cause issues with certain models
    if model in ['whittle', 'matern']:
        izero = h == 0
        flagzerodistances = np.any(izero)
    else:
        flagzerodistances = False
    
    # Adjust range parameter for unbounded models
    if type == 'unbounded':
        b0[0] = b0[0] / 3
    
    # Set up bounds for the optimization
    if params['solver'] == 'minimize':
        if nugget:
            bounds = [(0, None), (0, max(gammaexp)), (0, max(gammaexp))]
        else:
            bounds = [(0, None), (0, max(gammaexp))]
    
    # Define the variogram fitting function
    def variofun(b, h_values):
        if type == 'bounded':
            I = h_values <= b[0]
            gammahat = np.zeros_like(h_values)
            gammahat[I] = funnugget(b) + func(b, h_values[I])
            gammahat[~I] = funnugget(b) + b[1]
        else:  # unbounded
            gammahat = funnugget(b) + func(b, h_values)
            if flagzerodistances:
                gammahat[izero] = funnugget(b)
        return gammahat
    
    # Define weighting functions
    def weights_function(b, h_values):
        gamma_hat = variofun(b, h_values)
        
        if params['weightfun'] == 'cressie85':
            # Avoid division by zero
            eps = np.finfo(float).eps
            gamma_hat = np.maximum(gamma_hat, eps)
            w = numobs / gamma_hat**2
            return w / np.sum(w)
        elif params['weightfun'] == 'mcbratney86':
            # Avoid division by zero
            eps = np.finfo(float).eps
            gamma_hat = np.maximum(gamma_hat, eps)
            w = numobs * gammaexp / gamma_hat**3
            return w / np.sum(w)
        else:  # 'none'
            return np.ones_like(h_values)
    
    # Define objective function: weighted least square
    def objective_function(b):
        return np.sum(((variofun(b, h) - gammaexp)**2) * weights_function(b, h))
    
    # Perform optimization
    if params['solver'] == 'minimize':
        result = minimize(objective_function, b0, bounds=bounds, method='L-BFGS-B')
        b = result.x
        fval = result.fun
        exitflag = result.success
        output = {
            'algorithm': 'L-BFGS-B',
            'funcCount': result.nfev,
            'iterations': result.nit,
            'message': result.message
        }
    else:  # 'fmin'
        from scipy.optimize import fmin
        result = fmin(objective_function, b0, full_output=True, disp=False)
        b = result[0]
        fval = result[1]
        exitflag = 1  # fmin doesn't return success flag
        output = {
            'algorithm': 'Nelder-Mead simplex',
            'funcCount': result[2],
            'iterations': result[3],
            'message': 'Optimization terminated successfully' if not result[4] else 'Warning: Maximum number of function evaluations has been exceeded.'
        }
    
    # Extract optimized parameters
    a = b[0]  # range
    c = b[1]  # sill
    n = b[2] if nugget else None  # nugget
    
    # Create output structure
    S = {
        'model': model,
        'func': func,
        'type': type,
        'range': a,
        'sill': c,
        'nugget': n,
        'h': h,
        'gamma': gammaexp,
        'gammahat': variofun(b, h),
    }
    
    # Add model-specific parameters
    if model == 'matern':
        S['nu'] = params['nu']
    elif model == 'stable':
        S['stablealpha'] = params['stablealpha']
    
    # Calculate residuals and R-squared
    S['residuals'] = gammaexp - S['gammahat']
    
    # Calculate covariance and R-squared
    cov_matrix = np.cov(S['gammahat'], gammaexp)
    if cov_matrix.size > 1:  # Check if we have a 2x2 matrix
        S['Rs'] = (cov_matrix[0, 1]**2) / (np.var(S['gammahat']) * np.var(gammaexp))
    else:
        S['Rs'] = 1.0  # Perfect fit if we have only one point
    
    S['weights'] = weights_function(b, h)
    S['weightfun'] = params['weightfun']
    S['exitflag'] = exitflag
    S['algorithm'] = output['algorithm']
    S['funcCount'] = output['funcCount']
    S['iterations'] = output['iterations']
    S['message'] = output['message']
    
    # Plot results if requested
    if params['plotit']:
        plt.figure()
        plt.plot(h, gammaexp, 'rs', markersize=10)
        
        if type == 'bounded':
            # Plot within range
            h_plot = np.linspace(0, b[0], 100)
            gamma_plot = funnugget(b) + np.array([func(b, hi) for hi in h_plot])
            plt.plot(h_plot, gamma_plot, 'b-')
            
            # Plot beyond range (constant sill)
            h_plot2 = np.linspace(b[0], max(h), 100)
            gamma_plot2 = np.full_like(h_plot2, funnugget(b) + b[1])
            plt.plot(h_plot2, gamma_plot2, 'b-')
        else:  # unbounded
            h_plot = np.linspace(0, max(h), 200)
            gamma_plot = funnugget(b) + func(b, h_plot)
            plt.plot(h_plot, gamma_plot, 'b-')
        
        plt.xlim(0, max(h))
        plt.ylim(0, max(gammaexp)*1.1)
        plt.xlabel('lag distance h')
        plt.ylabel('γ(h)')
        plt.title(f'Variogram Fit ({model} model)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return a, c, n, S