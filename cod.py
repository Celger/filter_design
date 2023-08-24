#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Code to Design non-Gaussian SWIR Multispectral Filters based on the ECOSTRESS Library

This code was used in the paper "Design of non-Gaussian Multispectral SWIR Filters for Assessment of ECOSTRESS Library"
@author: Germano de Souza Fonseca <germanosfonseca@yahoo.com.br>
@date: 2023.05.30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA


if __name__ == "__main__":

    # dataset file
    dataset = 'data/reflet_amostras.csv'
    # illuminant file
    ill = 'data/ASTMG173.csv'
    # define the seed. Used to reproduce a result
    rs = 2
    # load the SSR dataset
    samples = pd.read_csv( dataset, header = None )
    # load the illuminant
    L = pd.read_csv( ill, header = None, delimiter = ' ', comment = '#', usecols = [ 2 ] )
    L = L[740:1541].to_numpy().squeeze()

    # Covariance matrix of the SSR dataset
    R = samples.to_numpy().T.dot(samples.to_numpy())/samples.shape[1]

    # splitting the dataset
    training = samples.sample(frac=0.8, random_state= rs )
    test = samples.drop(training.index)
    samples = training

    # instantiate the pca
    pca = PCA()
    # Caculate the principal component analysis
    pca.fit(samples)
    A = pca.components_.T
    V = np.diag( L ).dot( A )

    def gauss( parameters, filters = 3 ):
        '''Function to evaluate the filter transmittance modelled by a weighted sum of Gaussian
        :param parameters: list or numpy array of shape (n,m), where each row is a set of Gaussian to sum and make a filter
        :param filters: int corresponding to the number of filters
        :return M: numpy array of shape (801,n), the filters transmitance M
        :return Mh: numpy array of shape (801,n), the effective filters, i.e., Mh=LM
        '''
        if isinstance( parameters, list ):
            parameters = np.asarray( parameters )
            
        if parameters.ndim < 2:
            parameters = parameters.reshape( filters, -1 )
            
        # mean of the Gaussians
        mean = np.concatenate((parameters[:,0][:,np.newaxis], parameters[:,2::3] ), axis = 1)
        # standard deviation of the Gaussians
        std = np.concatenate((parameters[:,1][:,np.newaxis], parameters[:,3::3] ), axis = 1)
        # weight of the Gaussians
        weight = np.concatenate(( np.ones((filters,1)), parameters[:,4::3] ), axis = 1 )
        
        # wavelenght range, for instance, SWIR is from 900nm to 1700nm
        wavelengths = np.arange(900,1701)/1000

        M = np.exp( -( wavelengths[ :, np.newaxis, np.newaxis ] - mean )**2 / 
                   ( 2 * std**2 ) ) / ( std * np.sqrt( 2 * np.pi ) )
        M = M / np.maximum( M.max( axis = 0 ), 1e-20)
        
        n = []
        for i in range( M.shape[1] ):
            n += [ M[ :, i, : ].dot(weight[i,:] ) ]
            
        M = np.asarray( n ).T
        M = M / np.maximum( M.max( axis = 0 ), 1e-20)
        Mh = np.diag( L ).dot( M )
    
        return M, Mh

    def rcos( parameters, filters = 3 ):
        '''Function to evaluate the filter transmittance modelled by a weighted sum of raised cosine
        :param parameters: list or numpy array of shape (n,m), where each row is a set of raised cosine to sum and make a filter
        :param filters: int corresponding to the number of filters
        :return M: numpy array of shape (801,n), the filters transmitance M
        :return Mh: numpy array of shape (801,n), the effective filters, i.e., Mh=LM
        '''
        if isinstance( parameters, list ):
            parameters = np.asarray( parameters )
            
        if parameters.ndim < 2:
            parameters = parameters.reshape( filters, -1 )
            
        # period of the raised cosine
        t = np.concatenate((parameters[:,0][:,np.newaxis], parameters[:,2::3] ), axis = 1)
        # phase of the raised cosine
        zeta = np.concatenate((parameters[:,1][:,np.newaxis], parameters[:,3::3] ), axis = 1)
        # weight of the raised cosine
        weight = np.concatenate(( np.ones((filters,1)), parameters[:,4::3] ), axis = 1 )
        
        # wavelenght range, for instance, SWIR is from 900nm to 1700nm
        wavelengths = np.arange(900,1701)/1000
        M = ( 1.0 +np.cos( (2.0 * np.pi / t ) * ( wavelengths[ :, np.newaxis, np.newaxis ] - zeta )) ) / 2.0
        M[ np.abs( wavelengths[ :, np.newaxis, np.newaxis ] - zeta ) > t / 2.0] = 0
        M = M / np.maximum( M.max( axis = 0 ), 1e-20)
        
        n = []
        for i in range( M.shape[1] ):
            n += [ M[ :, i, : ].dot(weight[ i, : ] ) ]
            
        M = np.asarray( n ).T
        M = M / np.maximum( M.max( axis = 0 ), 1e-20)
        Mh = np.diag( L ).dot( M )
        return M, Mh
            
    def exp_cos( parameters, filters = 3 ):
        '''Function to evaluate the filter transmittance modelled by a weighted sum of exponential-cosine
        :param parameters: list or numpy array of shape (n,m), where each row is a set of exponential-cosine to sum and make a filter
        :param filters: int corresponding to the number of filters
        :return M: numpy array of shape (801,n), the filters transmitance M
        :return Mh: numpy array of shape (801,n), the effective filters, i.e., Mh=LM
        '''
        if isinstance( parameters, list ):
            parameters = np.asarray( parameters )
            
        if parameters.ndim < 2:
            parameters = parameters.reshape( filters, -1 )
            
        # period
        t = np.concatenate((parameters[:,0][:,np.newaxis], parameters[:,3::4] ), axis = 1)
        # phase
        zeta = np.concatenate((parameters[:,1][:,np.newaxis], parameters[:,4::4] ), axis = 1)
        # normalizing factor
        a = np.concatenate( (parameters[:,2][:,np.newaxis], parameters[:,5::4] ) , axis = 1)
        # weight
        weight = np.concatenate(( np.ones(( filters,1 )), parameters[:,6::4] ), axis = 1 )

        # wavelenght range, for instance, SWIR is from 900nm to 1700nm
        wavelengths = np.arange(900,1701)/1000
        M = ( np.exp(a * np.cos( 2.0 * np.pi * ( wavelengths[ :, np.newaxis, np.newaxis ] - zeta ) / t )) - 
             np.exp( -a ) ) / ( np.exp( a ) - np.exp( -a ) )
        M[ np.abs( wavelengths[ :, np.newaxis, np.newaxis ] - zeta ) > t / 2.0] = 0
        M = M / np.maximum( M.max( axis = 0 ), 1e-20)
        
        n = []
        for i in range( M.shape[1] ):
            n += [ M[ :, i, : ].dot(weight[ i, : ] ) ]
            
        M = np.asarray( n ).T
        M = M / np.maximum( M.max( axis = 0 ), 1e-20)
        Mh = np.diag( L ).dot( M )

        return M, Mh

    def initial_guess( filters = 3, nfunc = 1, model = 'gauss' ):
        '''Function to generate a random initial point to the Nelder-Mead optimizer considering the filter model
        :param filters: (int) number of filters
        :param nfunc: (int) number of summed functions in the model
        :param model: (string) the type of the filter transmittance model, options are 'gauss', 'rcos', and 'exp_cos'
        :return x0: numpy array of shape (n,), the initial point with the parameters of the model
        '''

        if model == 'gauss' :
            x0 = np.concatenate( ( ( 0.8 * np.random.rand( filters ) + 0.9)[ :, np.newaxis ],
                ( 0.5 * np.random.rand( filters ) + 0.001 )[ :, np.newaxis ]), axis = 1 )
            for j in range( nfunc - 1 ):
                x0 = np.concatenate( ( x0, ( 0.8 * np.random.rand( filters ) + 0.9)[ :, np.newaxis ],
                    ( 0.5 * np.random.rand( filters ) + 0.001 )[ :, np.newaxis ],
                    np.random.rand( filters )[ :, np.newaxis ]), axis = 1 )
        elif model == 'rcos' :
            x0 = np.concatenate( ( ( 2 * np.random.rand( filters ) + 0.001)[:,np.newaxis],
                ( 0.8 * np.random.rand( filters ) + 0.9 )[:,np.newaxis]), axis = 1 )
            for j in range( nfunc - 1 ):
                x0 = np.concatenate( ( x0, ( 2 * np.random.rand( filters ) + 0.001)[:,np.newaxis],
                    ( 0.8 * np.random.rand( filters ) + 0.9 )[:,np.newaxis], 
                    np.random.rand( filters )[ :, np.newaxis ] ), axis = 1 )
        elif model == 'exp_cos' :
            x0 = np.concatenate( ( ( 0.8 * np.random.rand( filters ) + 0.9)[:,np.newaxis],
                ( 0.8 * np.random.rand( filters ) + 0.9 )[:,np.newaxis],
                ( np.random.rand( filters ) )[ :, np.newaxis ] ), axis = 1 )
            for j in range( nfunc - 1 ):
                x0 = np.concatenate( ( x0, ( 0.8 * np.random.rand( filters ) + 0.9)[:,np.newaxis],
                    ( 0.8 * np.random.rand( filters ) + 0.9 )[:,np.newaxis],
                    ( np.random.rand( filters ) )[ :, np.newaxis ],
                    ( np.random.rand( filters ) )[ :, np.newaxis ]), axis = 1 )
        else:
            raise ValueError('Error: model not supported')
        
        x0 = x0.reshape(1,-1).squeeze()
        return x0

    def vora_value( parameters, space_dim = 3, filters = 3, model = 'gauss' ):
        '''Function to evaluate a weighted sum of gaussian filters using Vora value
        :param parameters: list or numpy array of shape (n,m), where each row is a set of gaussian to sum and make a filter transmittance
        :param space_dim: (int) number of dimension of the target subspace
        :param filters: (int) number of filters
        :param model: (string) the type of the filter transmittance model, options are 'gauss', 'rcos', and 'exp_cos'
        :return: The Vora value
        :rtype: float
        '''
        if isinstance( parameters, list ):
            parameters = np.asarray( parameters )
            
        if parameters.ndim < 2:
            parameters = parameters.reshape( filters, -1 )
            
        
        if model == 'gauss' :
            _, Mh = gauss( parameters, filters )
        elif model == 'rcos' :
            _, Mh = rcos( parameters, filters )
        elif model == 'exp_cos' :
            _, Mh = exp_cos( parameters, filters )
        else:
            raise ValueError('Error: model not supported')


        v = np.trace( V[ :, :space_dim ].dot(np.linalg.pinv( V[ :, :space_dim] ).dot(
            Mh.dot(np.linalg.pinv( Mh )))))/space_dim
        
        return 1.0 -v

    def err( Mh ):
        '''Function to evaluate the average and maximum MSE of the estimation errors of the effective filter Mh
        :param Mh: numpy array of shape (n,m), the effective filters
        :return: a tuple of the average and maximum MSE estimation errors
        :rtype: tuple of floats
        '''
        # Evaluate the data correction matrix
        B =  V[:,:space_dim].T.dot(R.dot( Mh ).dot( np.linalg.pinv( Mh.T.dot( R ).dot( Mh ) ) ) )#801,n

        # Evaluate the estimation errors
#        err = V[:,:space_dim].T.dot( samples.to_numpy().T) - B.dot(Mh.T.dot(samples.to_numpy().T))
        err = V[:,:space_dim].T.dot( test.to_numpy().T) - B.dot(Mh.T.dot(test.to_numpy().T))
        e_avg = ( err**2 ).mean(axis = 0 ).mean()
        e_max = ( err**2 ).mean(axis = 0 ).max() 
        
        return e_avg, e_max

    def assess( parameters, space_dim = 3, filters = 3, model = 'gauss' ):
        '''Function to evaluate the average MSE of estimation errors of filters using the model
        :param parameters: list or numpy array of shape (n,m), where each row is a set of gaussian to sum and make a filter transmittance
        :param space_dim: (int) number of dimension of the viewing subspace or target subspace
        :param filters: (int) number of filters
        :param model: (string) the type of the filter transmittance model, options are 'gauss', 'rcos', and 'exp_cos'
        :return: a tuple of the average and maximum MSE estimation errors
        :rtype: tuple of floats
        '''
        if isinstance( parameters, list ):
            parameters = np.asarray( parameters )
            
        if parameters.ndim < 2:
            parameters = parameters.reshape( filters, -1 )
            
        if model == 'gauss' :
            M, Mh = gauss( parameters, filters )
        elif model == 'rcos' :
            M, Mh = rcos( parameters, filters )
        elif model == 'exp_cos' :
            M, Mh = exp_cos( parameters, filters )
        else:
            raise ValueError('Error: model not supported')


        # Evaluate the data correction matrix
        B =  V[:,:space_dim].T.dot(R.dot( Mh ).dot( np.linalg.pinv( Mh.T.dot( R ).dot( Mh ) ) ) )#801,n

        # Evaluate the estimation errors
#        err = V[:,:space_dim].T.dot( samples.to_numpy().T) - B.dot(Mh.T.dot(samples.to_numpy().T))
        err = V[:,:space_dim].T.dot( test.to_numpy().T) - B.dot(Mh.T.dot(test.to_numpy().T))
        e_avg = ( err**2 ).mean(axis = 0 ).mean()
        e_max = ( err**2 ).mean(axis = 0 ).max() 

        print("Average MSE:", e_avg )
        print("Maximum MSE:", e_max )

        # Show the filter transmittance
        plt.plot(np.arange(900,1701),M);plt.ylabel('Normalized transmittance');plt.xlabel('Wavelength [nm]')
        plt.title('Filter set')
        plt.show()
        
        return e_avg, e_max

    # define the viewing subspace dimension
    space_dim = 3
    # define the number of filters
    filters = 3
    # define the number of gaussians in the model
    n_func = 1
    # define the model of the filter transmittance
    mode = 'gauss'
    # define the number of optimization repetitions
    rep = 30
    # variable to store the obtained results 
    res = []
    # define the performance threshold
    pt = 30e-6
    # define the process tolerance (tolerance margin)
    tol = 5e-2
    # define the total of Monte Carlo sampling
    total = int(1e5)

    print("Project of non-Gaussian filters using Nelder-Mead")
    print("Optimization parameters")
    print("Dimension of the Viewing subspace:", space_dim)
    if mode == 'gauss' :
        print("Number of filters:", filters,'\tNumber of Gaussians:', n_func)
    elif mode == 'rcos' :
        print("Number of filters:", filters,'\tNumber of Raised cosines:', n_func)
    elif mode == 'exp_cos' :
        print("Number of filters:", filters,'\tNumber of Exponential-cosine:', n_func)
    else:
        raise ValueError('Error: model not supported')

    print("Repeat", rep, "times")

    for i in range( rep ):
#        x0 = np.concatenate( ( ( 0.8 * np.random.rand( filters ) + 0.9)[ :, np.newaxis ],
#                              ( 0.5 * np.random.rand( filters ) + 0.001 )[ :, np.newaxis ]), axis = 1 )
#        for j in range(gaus-1):
#            x0 = np.concatenate( ( x0, ( 0.8 * np.random.rand( filters ) + 0.9)[ :, np.newaxis ],
#                                  ( 0.5 * np.random.rand( filters ) + 0.001 )[ :, np.newaxis ],
#                                  np.random.rand( filters )[ :, np.newaxis ]), axis = 1 )
#        x0 = x0.reshape(1,-1).squeeze()
        x0 = initial_guess( filters, n_func, mode)
       
       # optimize the parameters of the filter through Nelder-Mead
        res += [minimize( vora_value, x0, args = (space_dim, filters, mode ), 
                         method = 'nelder-mead', options = { 'disp': True, 
                                                            'maxiter': x0.shape[0]*500,
                                                            'maxfev': x0.shape[0]*500 })]
        print("\N{GREEK SMALL LETTER NU}(V_",space_dim,",Mh)=", 1-res[i].fun)
        print(res[i].x)

    # save variable res for log and debug
    np.save('Res_sum_'+str(n_func)+'_'+mode+'_'+str(filters)+'_filters_'+str(rs)+'_training_'+str(i+1)+'_rep.npy', res)

    f=[]
    for i in range(len(res)):
        f +=[1-res[i].fun]
    f=np.asarray(f)

    # Filter assessment
    print("Best filter")
    print("\N{GREEK SMALL LETTER NU}(V_",space_dim,",Mh)=", 1-res[f.argmax()].fun)
    print("Parameters of the model:\n", res[f.argmax()].x)

    print("Evaluating the estimation errors")
    assess( res[ f.argmax() ].x, space_dim, filters, mode )
    input("Press Enter to continue...")


    # Monte Carlo Simulation
    print("Monte Carlo Simulation")

    if mode == 'gauss' :
        M, _ = gauss( res[ f.argmax() ].x, filters )
    elif mode == 'rcos' :
        M, _ = rcos( res[ f.argmax() ].x, filters )
    elif mode == 'exp_cos' :
        M, _ = exp_cos( res[ f.argmax() ].x, filters )
    else:
        raise ValueError('Error: model not supported')
#    M, _ = gauss( res[ f.argmax() ].x, filters )
    e_tr = []
    out_tol = []

    for k in range( total ):
#        # Sampling the filter transmittance from a normal distribution with mean M and standard deviation M*tol/3
#        # Filter transmittance perturbed
#        M_p = ( np.random.normal(0, tol/3, (801,filters)) +1 ) * M
#        mh = np.diag( L ).dot( M_p )
#        e_tr += [ err( mh ) ]
#        # check if the sampled filter transmittance that is above the performance threshold is within the tolerance margin
#        if e_tr[k][0] > pt:
#            a = ( M_p / M )
#            out_tol += [ np.abs( a-1 ).max() ]
        # Sampling the filter transmittance from a normal distribution with mean M and standard deviation M*tol/3
        # Samples of deviations from the nominal transmittance
        w = np.random.normal( 0, tol/3, ( 801, filters ) )
        # truncate the samples of deviations
        w[ w > tol ] = tol
        w[ w < tol ] = -tol
        # Perturbed filter transmittance
        M_p = ( 1 + w ) * M
        mh = np.diag( L ).dot( M_p )
        e_tr += [ err( mh ) ]

    e_tr = np.asarray( e_tr )
#    out_tol= np.asarray( out_tol )

    # Number of samples with e_avg < pt
    n = ( e_tr[:,0] < pt ).shape
    # Estimated yield
    est_yield = n / total
    # Confidence interval
    L = 3 * np.sqrt( est_yield * ( 1-est_yield ) / total )
    print("Estimated yield:", est_yield*100,"\%")
    print("Confidence interval:", est_yield*100,"+-",L*100,"\%")
    print("Confidence level: 99.73\%")

    hist = plt.hist( e_tr[:,0], 50 )
    hist = np.concatenate( ( np.asarray( hist[1] )[ :-1, np.newaxis ], np.asarray( hist[0] )[:, np.newaxis] ), axis = 1 )
    plt.title( 'e_avg Histogram' )
    plt.xlabel( ' e_avg' ); plt.ylabel( 'N of occurences' )
    input("Press Enter to continue...")
