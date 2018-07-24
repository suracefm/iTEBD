import numpy as np
import os
from trotter import *

folder='phi0'
dim = 3
for g in np.linspace(-1,0, 21):
    for theta in np.linspace(0, np.pi/3, 21):
        print(g, theta)
        g = float("{0:.5f}".format(g))
        J = float("{0:.5f}".format(-1-g)); theta = float("{0:.5f}".format(theta)); phi = float("{0:.5f}".format(0));
        alpha = np.array([np.exp(1j*theta),np.exp(-1j*theta)])
        beta = np.array([np.exp(1j*phi),np.exp(1j*phi)])
        chi = 10; delta = 0.01; N = 500


        sigma = np.diag(np.exp(1j*2*np.pi*np.arange(dim)/dim))
        tau = np.roll(np.eye(dim), -1, axis = -1)
        interactions = list(zip(J*alpha, [sigma**m for m in range(1,dim)], [sigma**(dim-m) for m in range(1,dim)]))
        transv_field = list(zip(g*beta, [tau**m for m in range(1,dim)]))
        U=ST_step(delta, interactions, transv_field)

        GA0 = np.random.rand(dim,chi,chi)
        LA0 = np.random.rand(chi)
        GB0 = np.random.rand(dim,chi,chi)
        LB0 = np.random.rand(chi)

        res={}
        res_names = ["Energy", "SigmaA", "SigmaB"]

        GA = GA0; LA = LA0; GB = GB0; LB=LB0
        discarded_weights = []
        for step in range(N):
            GA, LA, GB, dw, norm_sq = iTEDB(dim, chi, GA, LA, GB, LB, U)
            E_sim =-np.log(norm_sq)/delta/2
            discarded_weights.append(dw)
            GB, LB, GA, dw, norm_sq = iTEDB(dim, chi, GB, LB, GA, LA, U)
            E_sim =-np.log(norm_sq)/delta/2
            discarded_weights.append(dw)
        #res["Energy"] = E_sim
        #res["SigmaA"] = np.einsum('sab,tab,st,b,a->', GA, np.conj(GA), sigma, LA**2, LB**2)
        #res["SigmaB"] = np.einsum('sab,tab,st,b,a->', GB, np.conj(GB), sigma, LB**2, LA**2)

        #for key in res_names: print(key, res[key])

        File_code = '{}J_{}g_{}theta_{}phi_{}Chi_{}delta_{}N'.format(J, g, theta, phi, chi, delta,N)
        os.system('mkdir -p {}/data_'.format(folder)+File_code)

        np.savetxt('./{}/data_{}/Energy.dat'.format(folder,File_code), [E_sim], header=File_code)
        np.savetxt('./{}/data_{}/GammaA.dat'.format(folder,File_code), GA.flatten(), header=File_code)
        np.savetxt('./{}/data_{}/GammaB.dat'.format(folder, File_code), GB.flatten(), header=File_code)
        np.savetxt('./{}/data_{}/LambdaA.dat'.format(folder, File_code), LA, header=File_code)
        np.savetxt('./{}/data_{}/LambdaB.dat'.format(folder, File_code), LB, header=File_code)
        np.savetxt('./{}/data_{}/DiscardedWeights.dat'.format(folder,File_code), discarded_weights, header=File_code)

        #Outfile1 = './clock3/data_{}/Results.dat'.format(File_code)
        #Outfile2 = './clock3/data_{}/EntanglementSpectrum.dat'.format(File_code)
        # Outfile3 = './clock3/data_{}/DiscardedWeights.dat'.format(File_code)
        #
        # with open(Outfile1, 'w') as f1:
        #     print('# ', File_code, file=f1)
        #     print('# ', *res_names, file=f1)
        #     print(*[res[key] for key in res_names], file=f1)
        # np.savetxt(Outfile2, LB**2, header=File_code)
        # np.savetxt(Outfile3, discarded_weights, header=File_code)
