import numpy as np

def center_data(A):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    #
    # OUTPUT:
    # X    [NxM] numpy centered data matrix (N samples, M features)
    m = np.mean(A, axis=0)
    X = A - m
    return X

def compute_covariance_matrix(A):
    # INPUT:
    # A    [NxM] centered numpy data matrix (N samples, M features)
    #
    # OUTPUT:
    # C    [MxM] numpy covariance matrix (M features, M features)
    #
    # Do not apply centering here. We assume that A is centered before this function is called.
    C = np.cov(np.transpose(A))
    return C

def compute_eigenvalue_eigenvectors(A):
    # INPUT:
    # A    [DxD] numpy matrix
    #
    # OUTPUT:
    # eigval    [D] numpy vector of eigenvalues
    # eigvec    [DxD] numpy array of eigenvectors
    eigval, eigvec = np.linalg.eig(A)
    # Numerical roundoff can lead to (tiny) imaginary parts. We correct that here.
    eigval = eigval.real
    eigvec = eigvec.real
    return eigval, eigvec

def sort_eigenvalue_eigenvectors(eigval, eigvec):
    # INPUT:
    # eigval    [D] numpy vector of eigenvalues
    # eigvec    [DxD] numpy array of eigenvectors
    #
    # OUTPUT:
    # sorted_eigval    [D] numpy vector of eigenvalues
    # sorted_eigvec    [DxD] numpy array of eigenvectors
    indices = np.argsort(eigval)
    indices = indices[::-1]
    sorted_eigval = eigval[indices]
    sorted_eigvec = eigvec[:,indices]
    return sorted_eigval, sorted_eigvec

def pca(A,m):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    # m    integer number denoting the number of learned features (m <= M)
    #
    # OUTPUT:
    # pca_eigvec    [Mxm] numpy matrix containing the eigenvectors (M dimensions, m eigenvectors)
    # P             [Nxm] numpy PCA data matrix (N samples, m features)
    
    A = center_data(A)
    C = compute_covariance_matrix(A)
    eigval, eigvec = compute_eigenvalue_eigenvectors(C)
    eigval, eigvec = sort_eigenvalue_eigenvectors(eigval, eigvec)
    
    if m > 0:
        eigvec = eigvec[:,:m]

    for i in range(np.shape(eigvec)[1]):
        eigvec[:,i] / np.linalg.norm(eigvec[:,i]) * np.sqrt(eigval[i])

    pca_eigvec = eigvec
    
    P = np.dot(np.transpose(eigvec), np.transpose(A))

    return pca_eigvec, P.T

def encode_decode_pca(A,m):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    # m    integer number denoting the number of learned features (m <= M)
    #
    # OUTPUT:
    # Ahat [NxM] numpy PCA reconstructed data matrix (N samples, M features)
    me = np.mean(A, axis=0)
    A = center_data(A)
    C = compute_covariance_matrix(A)
    eigval, eigvec = compute_eigenvalue_eigenvectors(C)
    eigval, eigvec = sort_eigenvalue_eigenvectors(eigval, eigvec)
    if m > 0:
        eigvec = eigvec[:,:m]
    for i in range(np.shape(eigvec)[1]):
        eigvec[:,i] / np.linalg.norm(eigvec[:,i]) * np.sqrt(eigval[i])
    P = np.dot(np.transpose(eigvec), np.transpose(A))
    Ahat = np.dot(P[:,:m].T, eigvec.T)
    Ahat += me  
    return Ahat