import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

def kriging(vstruct, x, y, z, xi, yi, chunksize=100):
    """
    Interpolation with ordinary kriging in two dimensions.
    
    Parameters:
    -----------
    vstruct : dict
        Structure with variogram information as returned by variogramfit (4th output)
    x, y : array-like
        Coordinates of observations
    z : array-like
        Values of observations
    xi, yi : array-like
        Coordinates of locations for predictions
    chunksize : int, optional
        Number of elements in zi that are processed at one time.
        The default is 100, but this depends on available memory and len(x).
    
    Returns:
    --------
    zi : array-like
        Kriging predictions
    s2zi : array-like, optional
        Kriging variance
    
    Example:
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib import cm
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> 
    >>> # Create random field with autocorrelation
    >>> x_grid = np.arange(0, 501)
    >>> y_grid = np.arange(0, 501)
    >>> X, Y = np.meshgrid(x_grid, y_grid)
    >>> Z = np.random.randn(*X.shape)
    >>> 
    >>> # Apply Gaussian filter to create spatial correlation
    >>> from scipy.ndimage import gaussian_filter
    >>> Z = gaussian_filter(Z, sigma=8)
    >>> 
    >>> # Sample the field
    >>> n = 500
    >>> x = np.random.rand(n) * 500
    >>> y = np.random.rand(n) * 500
    >>> 
    >>> # Interpolate at random sample locations
    >>> from scipy.interpolate import interpn
    >>> points = (y_grid, x_grid)
    >>> z = interpn(points, Z, np.vstack((y, x)).T, method='linear')
    >>> 
    >>> # Calculate sample variogram
    >>> from variogram import variogram
    >>> v = variogram(np.column_stack((x, y)), z, maxdist=100, plotit=False)
    >>> 
    >>> # Fit a variogram model
    >>> from variogramfit import variogramfit
    >>> a, c, n, vstruct = variogramfit(v['distance'], v['val'], model='stable')
    >>> 
    >>> # Perform kriging interpolation
    >>> Zhat, Zvar = kriging(vstruct, x, y, z, X, Y)
    >>> 
    >>> # Plot results
    >>> fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    >>> 
    >>> # Original field with sample points
    >>> im1 = axs[0, 0].imshow(Z, extent=[0, 500, 0, 500], origin='lower')
    >>> axs[0, 0].plot(x, y, '.k')
    >>> axs[0, 0].set_title('Random field with sampling locations')
    >>> plt.colorbar(im1, ax=axs[0, 0])
    >>> 
    >>> # Kriging predictions
    >>> im2 = axs[1, 0].imshow(Zhat, extent=[0, 500, 0, 500], origin='lower')
    >>> axs[1, 0].set_title('Kriging predictions')
    >>> plt.colorbar(im2, ax=axs[1, 0])
    >>> 
    >>> # Kriging variance
    >>> im3 = axs[1, 1].contourf(X, Y, Zvar, cmap='viridis')
    >>> axs[1, 1].set_title('Kriging variance')
    >>> plt.colorbar(im3, ax=axs[1, 1])
    >>> 
    >>> plt.tight_layout()
    >>> plt.show()
    """
    # Size of input arguments
    sizest = np.shape(xi)
    numest = np.size(xi)
    numobs = np.size(x)
    
    # Force column vectors
    xi = np.array(xi).flatten()
    yi = np.array(yi).flatten()
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    z = np.array(z).flatten()
    
    # Check if we have the correct variogram structure
    if not isinstance(vstruct, dict) or 'func' not in vstruct:
        raise ValueError("vstruct must be a dictionary with a 'func' field as returned by variogramfit")
    
    # Variogram model checks
    if vstruct['model'].lower() in ['whittle', 'matern']:
        raise NotImplementedError('whittle and matern models are not supported yet')
    
    # Distance matrix of locations with known values
    # Use broadcasting to calculate distances more efficiently
    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y
    Dx = np.hypot(dx, dy)
    
    # For bounded variogram models, set distances > range to range
    if vstruct['type'] == 'bounded':
        Dx = np.minimum(Dx, vstruct['range'])
    
    # Calculate matrix with variogram values
    A = vstruct['func']([vstruct['range'], vstruct['sill']], Dx)
    
    # Add nugget effect if present
    if vstruct['nugget'] is not None:
        A = A + vstruct['nugget']
    
    # Expand matrix to account for the condition that weights must sum to one
    A = np.vstack([np.hstack([A, np.ones((numobs, 1))]),
                  np.hstack([np.ones((1, numobs)), np.zeros((1, 1))])])
    
    # Use pseudo-inverse for solving the equations (handles ill-conditioned matrices)
    A = linalg.pinv(A)
    
    # Expand z with zero (Lagrange multiplier)
    z = np.append(z, 0)
    
    # Allocate output arrays
    zi = np.full(numest, np.nan)
    
    # Check if we need to calculate kriging variance
    if 2 > 1:  # This mimics MATLAB's nargout == 2
        s2zi = np.full(numest, np.nan)
        krigvariance = True
    else:
        krigvariance = False
    
    # Calculate number of loops needed
    nrloops = int(np.ceil(numest / chunksize))
    
    # Process in chunks
    for r in range(1, nrloops + 1):
        # Build chunks
        if r < nrloops:
            IX = slice((r-1)*chunksize, r*chunksize)
            current_chunk_size = chunksize
        else:
            IX = slice((r-1)*chunksize, numest)
            current_chunk_size = numest - (r-1)*chunksize
        
        # Current chunk of target coordinates
        xi_chunk = xi[IX]
        yi_chunk = yi[IX]
        
        # Build distance matrix between observations and target points
        dx = x[:, np.newaxis] - xi_chunk
        dy = y[:, np.newaxis] - yi_chunk
        b = np.hypot(dx, dy)
        
        # Set maximum distances to the range for bounded variograms
        if vstruct['type'] == 'bounded':
            b = np.minimum(vstruct['range'], b)
        
        # Calculate variogram values
        b = vstruct['func']([vstruct['range'], vstruct['sill']], b)
        
        # Add nugget effect if present
        if vstruct['nugget'] is not None:
            b = b + vstruct['nugget']
        
        # Expand b with ones
        b = np.vstack([b, np.ones((1, current_chunk_size))])
        
        # Solve system
        lambda_values = A @ b
        
        # Estimate zi
        zi[IX] = lambda_values.T @ z
        
        # Calculate kriging variance
        if krigvariance:
            s2zi[IX] = np.sum(b * lambda_values, axis=0)
    
    # Reshape outputs to match input shapes
    zi = zi.reshape(sizest)
    
    if krigvariance:
        s2zi = s2zi.reshape(sizest)
        return zi, s2zi
    else:
        return zi