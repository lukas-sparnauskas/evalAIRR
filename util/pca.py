import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

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
    eigval, eigvec = np.linalg.eigh(A)
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
    
    print('[LOG] PCA: Centering data')
    A = center_data(A)
    print('[LOG] PCA: Computing covariance matrix')
    C = compute_covariance_matrix(A)
    print('[LOG] PCA: Computing PCA matrix and eigenvectors')
    eigval, eigvec = compute_eigenvalue_eigenvectors(C)
    print('[LOG] PCA: Sorting PCA matrix and eigenvectors')
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

def export_pca_2d_comparison(data_real, data_sim):
    eigvec_R, pca_R = pca(data_real, 2)
    eigvec_S, pca_S = pca(data_sim, 2)

    eigvec_R_x = np.linspace(min(pca_R[:, 0]), max(pca_R[:, 0]), 1000)
    eigvec_R_y = eigvec_R[0][1] / eigvec_R[0][0] * eigvec_R_x

    eigvec_S_x = np.linspace(min(pca_S[:, 0]), max(pca_S[:, 0]), 1000)
    eigvec_S_y = eigvec_S[0][1] / eigvec_S[0][0] * eigvec_S_x

    f,(ax1, ax2) = plt.subplots(1, 2)
    f.set_size_inches(10, 5)
    f.suptitle('PCA comparison in two dimensions')

    ax1.scatter(pca_R[:, 0], pca_R[:, 1])
    ax1.plot(eigvec_R_x, eigvec_R_y, c='#1b24a8')
    ax1.set_title('Real dataset')

    ax2.scatter(pca_S[:, 0], pca_S[:, 1], c='red')
    ax2.plot(eigvec_S_x, eigvec_S_y, c='#781010')
    # ax2.set_ylabel('')
    # ax2.set_yticklabels([])
    # ax2.set_yticks([])
    ax2.set_title('Simulated dataset')
    
    xbound = (min(ax1.get_xbound()[0], ax2.get_xbound()[0]), max(ax1.get_xbound()[1], ax2.get_xbound()[1]))
    ybound = (min(ax1.get_ybound()[0], ax2.get_ybound()[0]), max(ax1.get_ybound()[1], ax2.get_ybound()[1]))

    ax1.set_xbound(xbound)
    ax1.set_ybound(ybound)
    ax2.set_xbound(xbound)
    ax2.set_ybound(ybound)

    plt.show()