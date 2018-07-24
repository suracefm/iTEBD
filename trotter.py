import numpy as np
from scipy.linalg import expm



def ST_step(delta, twosite, onesite=None):
    """ SUZUKI-TROTTER EVOLUTION: single step
    Construct evolution operator for two sites
    Inputs:
        delta (real for GS or imaginary for evolution)
        list of triplets (coupling, op1, op2) -> acting on two sites
        list of pairs (coupling, op1) -> acting on one site
    Output:
        U[j'j,k',k]

        old vector
        j'|_____|k'
        |____U____|
         j|     |k
         new vector
    """
    dim_loc = np.shape(twosite[0][1])[0]
    H = np.zeros((dim_loc**2, dim_loc**2))
    for J, op1, op2 in twosite:
        assert np.shape(op1)==(dim_loc, dim_loc)
        assert np.shape(op2)==(dim_loc, dim_loc)
        H+=np.real(J*np.kron(op1,op2))
    if onesite is not None:
        for h, op in onesite:
            assert np.shape(op)==(dim_loc, dim_loc)
            H+=np.real((h/2)*np.kron(op, np.eye(dim_loc)))
            H+=np.real((h/2)*np.kron(np.eye(dim_loc), op))
    U = expm(-delta*H)
    return np.reshape(U, (dim_loc, dim_loc, dim_loc, dim_loc)) #j', k', j, k

def iTEDB(dim, chi, GammaA, LambdaA, GammaB, LambdaB, U):
    """ iTEDB: single step
    Performs one step of iTEBD, keeping the smallest chi eigenvalues of the SVD
    Inputs:
        dim (dimension of real indices)
        chi (dimension of virtual indices)
        GammaA (matrix dim*chi*chi)
        LambdaA (vector chi)
        GammaB (matrix dim*chi*chi)
        LambdaB (vector chi)
        U (matrix dim*dim*dim*dim)
    Outputs:
        new GammaA (matrix dim*chi*chi)
        new LambdaA (vector chi)
        new GammaB (matrix dim*chi*chi)
        discarded weights
        norm squared (before truncating and normalizing)-> for
                imaginary time evolution is equal to e^(2*delta*Energy)
    """
    old_theta = np.einsum('a,lab,b,mbc,c->lamc', LambdaB, GammaA,\
                LambdaA, GammaB, LambdaB)
    theta = np.einsum('lmjk,lamc->jakc', U, old_theta)
    theta = np.reshape(theta, (dim*chi, dim*chi))
    X, Y, Z = np.linalg.svd(theta)
    norm_sq = np.sum(Y**2)

    # Truncation
    dw = np.sum(Y[chi:])
    new_LambdaA = Y[:chi]/np.sqrt(np.sum(Y[:chi]**2))
    new_GammaA = np.einsum('a,lab->lab', 1/LambdaB,\
        np.reshape(X[:, :chi], (dim, chi, chi)))
    new_GammaB = np.einsum('c,bkc->kbc', 1/LambdaB,\
        np.reshape(Z[:chi], (chi, dim, chi)))

    return new_GammaA, new_LambdaA, new_GammaB, dw, norm_sq
