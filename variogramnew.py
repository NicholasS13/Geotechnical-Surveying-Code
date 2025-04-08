import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

def variogram(x, y, **kwargs):
    """
    Isotropic and anisotropic experimental (semi-)variogram
    
    Parameters:
    -----------
    x : array-like
        Coordinates. Each row is a location in a n-dimensional space
        (e.g. [x y elevation])
    y : array-like
        Values at the locations in x.
    
    Keyword Arguments:
    -----------------
    nrbins : int, optional
        Number of bins the distance should be grouped into (default=20)
    maxdist : float, optional
        Maximum distance for variogram calculation 
        (default = maximum distance in dataset / 2)
    type : str, optional
        'gamma' returns the variogram value (default)
        'cloud1' returns the binned variogram cloud
        'cloud2' returns the variogram cloud
    plotit : bool, optional
        True -> plot variogram
        False -> don't plot (default)
    subsample : int, optional
        Number of randomly drawn points if large datasets are used.
        Positive integer (e.g. 3000) or np.inf (default) = no subsampling
    anisotropy : bool, optional
        False (default), True (works only in two dimensions)
    thetastep : float, optional
        If anisotropy is set to true, specifying thetastep allows 
        you to set the angle width (default 30°)
    
    Returns:
    --------
    dict
        A dictionary with distance and gamma vectors
    
    Example:
    --------
    Generate a random field with periodic variation in x direction
    
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.random.rand(1000) * 4 - 2
    >>> y = np.random.rand(1000) * 4 - 2
    >>> z = 3 * np.sin(x * 15) + np.random.randn(len(x))
    >>> coords = np.column_stack((x, y))
    >>> 
    >>> plt.figure(figsize=(10, 8))
    >>> 
    >>> plt.subplot(2, 2, 1)
    >>> plt.scatter(x, y, c=z, s=4, cmap='viridis')
    >>> plt.colorbar()
    >>> plt.ylabel('y')
    >>> plt.xlabel('x')
    >>> plt.title('data (coloring according to z-value)')
    >>> 
    >>> plt.subplot(2, 2, 2)
    >>> plt.hist(z, 20)
    >>> plt.ylabel('frequency')
    >>> plt.xlabel('z')
    >>> plt.title('histogram of z-values')
    >>> 
    >>> plt.subplot(2, 2, 3)
    >>> d = variogram(coords, z, plotit=True, nrbins=50)
    >>> plt.title('Isotropic variogram')
    >>> 
    >>> plt.subplot(2, 2, 4)
    >>> d2 = variogram(coords, z, plotit=True, nrbins=50, anisotropy=True)
    >>> plt.title('Anisotropic variogram')
    >>> 
    >>> plt.tight_layout()
    >>> plt.show()
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y).flatten()
    
    # Error checking
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    
    # Check for nans
    II = ~(np.isnan(x).any(axis=1) | np.isnan(y))
    x = x[II]
    y = y[II]
    
    # Extent of dataset
    minx = np.min(x, axis=0)
    maxx = np.max(x, axis=0)
    maxd = np.sqrt(np.sum((maxx - minx) ** 2))
    nrdims = x.shape[1]
    
    # Parse parameters with defaults
    params = {
        'nrbins': 20,
        'maxdist': maxd / 2,
        'type': 'gamma',
        'plotit': False,
        'anisotropy': False,
        'thetastep': 30,
        'subsample': np.inf
    }
    
    # Update with user-provided parameters
    for key, value in kwargs.items():
        if key in params:
            params[key] = value
        else:
            warnings.warn(f"Unknown parameter: {key}")
    
    # Validate parameter values
    if params['type'] not in ['gamma', 'cloud1', 'cloud2']:
        params['type'] = 'gamma'  # Default to gamma if invalid type
    
    if params['maxdist'] > maxd:
        warnings.warn(f"Maximum distance exceeds maximum distance in the dataset. maxdist was decreased to {maxd}")
        params['maxdist'] = maxd
    
    if params['anisotropy'] and nrdims != 2:
        params['anisotropy'] = False
        warnings.warn("Anisotropy is only supported for 2D data")
    
    # Take only a subset of the data
    if not np.isinf(params['subsample']) and len(y) > params['subsample']:
        IX = np.random.choice(len(y), size=int(params['subsample']), replace=False)
        x = x[IX]
        y = y[IX]
    
    # Calculate bin tolerance
    tol = params['maxdist'] / params['nrbins']
    
    # Calculate distance matrix
    iid = distmat(x, params['maxdist'])
    
    # If no pairs are within maxdist, return empty result
    if iid.size == 0:
        return {'distance': np.array([]), 'val': np.array([])}
    
    # Calculate squared difference between values of coordinate pairs
    lam = (y[iid[:, 0]] - y[iid[:, 1]]) ** 2
    
    # Initialize result dictionary
    S = {}
    
    # Handle anisotropy
    if params['anisotropy']:
        thetastep_rad = params['thetastep'] / 180 * np.pi
        nrthetaedges = int(np.floor(180 / params['thetastep'])) + 1
        
        # Calculate angles (Matching MATLAB's atan2 behavior)
        # In MATLAB: atan2(x(iid(:,2),1)-x(iid(:,1),1), x(iid(:,2),2)-x(iid(:,1),2))
        dx = x[iid[:, 1], 0] - x[iid[:, 0], 0]  # x difference
        dy = x[iid[:, 1], 1] - x[iid[:, 0], 1]  # y difference
        theta = np.arctan2(dx, dy)  # Using same order as MATLAB
        
        # Only the semicircle is necessary for the directions
        mask = theta < 0
        theta[mask] += np.pi
        mask = theta >= np.pi - thetastep_rad/2
        theta[mask] = 0
        
        # Create a vector with edges for binning of theta
        thetaedges = np.linspace(-thetastep_rad/2, np.pi-thetastep_rad/2, nrthetaedges)
        
        # Bin theta using numpy's digitize
        ixtheta = np.digitize(theta, thetaedges) - 1
        
        # Bin centers
        thetacents = thetaedges[:-1] + thetastep_rad/2
        thetacents = np.append(thetacents, np.pi)
    
    # Calculate variogram
    if params['type'] in ['default', 'gamma']:
        # Variogram function
        def fvar(x):
            return 1 / (2 * len(x)) * np.sum(x)
        
        # Distance bins
        edges = np.linspace(0, params['maxdist'], params['nrbins'] + 1)
        edges[-1] = np.inf
        
        # Bin distances
        ixedge = np.digitize(iid[:, 2], edges) - 1
        
        if params['anisotropy']:
            # Initialize arrays
            val = np.full((len(edges)-1, len(thetaedges)), np.nan)
            num = np.full((len(edges)-1, len(thetaedges)), np.nan)
            
            # Accumulate values
            for i in range(len(edges)-1):
                for j in range(len(thetaedges)):
                    mask = (ixedge == i) & (ixtheta == j)
                    if np.any(mask):
                        val[i, j] = fvar(lam[mask])
                        num[i, j] = np.sum(mask)
            
            # Copy values from first to last column (cyclic)
            val[:, -1] = val[:, 0]
            num[:, -1] = num[:, 0]
            
            S['val'] = val
            S['theta'] = thetacents
            S['num'] = num
        else:
            # Initialize arrays
            val = np.full(len(edges)-1, np.nan)
            num = np.full(len(edges)-1, np.nan)
            
            # Accumulate values
            for i in range(len(edges)-1):
                mask = ixedge == i
                if np.any(mask):
                    val[i] = fvar(lam[mask])
                    num[i] = np.sum(mask)
            
            S['val'] = val
            S['num'] = num
        
        S['distance'] = edges[:-1] + tol/2
        
    elif params['type'] == 'cloud1':
        edges = np.linspace(0, params['maxdist'], params['nrbins'] + 1)
        edges[-1] = np.inf
        
        # Bin distances
        ixedge = np.digitize(iid[:, 2], edges) - 1
        
        S['distance'] = edges[ixedge] + tol/2
        S['val'] = lam
        
        if params['anisotropy']:
            S['theta'] = thetacents[ixtheta]
            
    elif params['type'] == 'cloud2':
        S['distance'] = iid[:, 2]
        S['val'] = lam
        
        if params['anisotropy']:
            S['theta'] = thetacents[ixtheta]
    
    # Create plot if desired
    if params['plotit']:
        if params['type'] in ['default', 'gamma']:
            marker = 'o--'
        else:
            marker = '.'
        
        if not params['anisotropy']:
            plt.plot(S['distance'], S['val'], marker)
            plt.axis([0, params['maxdist'], 0, np.nanmax(S['val']) * 1.1])
            plt.xlabel('h')
            plt.ylabel('γ (h)')
            plt.title('(Semi-)Variogram')
        else:
            # Create polar grid and convert to cartesian coordinates
            theta_mesh, dist_mesh = np.meshgrid(S['theta'], S['distance'])
            X = dist_mesh * np.cos(theta_mesh)
            Y = dist_mesh * np.sin(theta_mesh)
            
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, S['val'], cmap='viridis')
            ax.set_xlabel('h y-direction')
            ax.set_ylabel('h x-direction')
            ax.set_zlabel('γ (h)')
            ax.set_title('directional variogram')
            plt.colorbar(surf)
    
    return S


def distmat(X, dmax):
    """
    Constrained distance function
    
    Parameters:
    -----------
    X : array-like
        Coordinates for distance calculation
    dmax : float
        Maximum distance to consider
    
    Returns:
    --------
    array
        Array with [row_indices, column_indices, distances]
    """
    X = np.asarray(X)
    n = X.shape[0]
    nrdim = X.shape[1]
    
    if n < 1000:
        # For smaller datasets, calculate all pairwise distances at once
        i, j = np.triu_indices(n, 1)  # Upper triangular indices (excluding diagonal)
        
        if nrdim == 1:
            # For 1D, just take absolute difference
            d = np.abs(X[i, 0] - X[j, 0])
        elif nrdim == 2:
            # For 2D, use hypot (more stable than manual calculation)
            d = np.hypot(X[i, 0] - X[j, 0], X[i, 1] - X[j, 1])
        else:
            # For higher dimensions
            d = np.sqrt(np.sum((X[i] - X[j])**2, axis=1))
        
        mask = d <= dmax
        return np.column_stack((i[mask], j[mask], d[mask]))
    else:
        # For larger datasets, handle in chunks to avoid memory issues
        result = []
        for i in range(n):
            j = np.arange(i+1, n)
            
            if j.size == 0:  # Skip if no pairs to compare
                continue
                
            if nrdim == 1:
                d = np.abs(X[i, 0] - X[j, 0])
            elif nrdim == 2:
                d = np.hypot(X[i, 0] - X[j, 0], X[i, 1] - X[j, 1])
            else:
                # Using vectorized operations for higher dimensions
                d = np.sqrt(np.sum((X[i] - X[j])**2, axis=1))
            
            mask = d <= dmax
            if np.any(mask):
                result.append(np.column_stack((np.full(np.sum(mask), i), j[mask], d[mask])))
        
        if result:
            return np.vstack(result)
        else:
            return np.empty((0, 3))