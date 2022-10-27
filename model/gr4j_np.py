import numpy as np

class GR4J(object):

    _params = {
        'x1': np.random.uniform(100, 1200),
        'x2': np.random.uniform(-3, 5),
        'x3': np.random.uniform(20, 300),
        'x4': np.random.uniform(1, 3)
    }

    def __init__(self, s_init=0, r_init=0):
        self.s_init = s_init
        self.r_init = r_init

    def update_params(self, x1, x2, x3, x4):
        self._params = {
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4
        }

    def _s_curve1(self, t, x4):
        """Calculate the s-curve of the unit-hydrograph 1.
        
        Args:
            t: timestep
            x4: model parameter x4 of the gr4j model.
            
        """
        if t <= 0:
            return 0.
        elif t < x4:
            return (t / x4)**2.5
        else:
            return 1.


    def _s_curve2(self, t, x4): 
        """Calculate the s-curve of the unit-hydrograph 2.
        
        Args:
            t: timestep
            x4: model parameter x4 of the gr4j model.
            
        """
        if t <= 0:
            return 0.
        elif t <= x4:
            return 0.5 * ((t / x4) ** 2.5)
        elif t < 2*x4:
            return 1 - 0.5 * ((2 - t / x4) ** 2.5)
        else:
            return 1.
    
    def run(self, prec, etp):
        
        # Number of simulation timesteps
        num_timesteps = len(prec)
        
        # Unpack the model parameters
        x1 = self._params['x1']
        x2 = self._params['x2']
        x3 = self._params['x3']
        x4 = self._params['x4']
        
        # initialize empty arrays for discharge and all storages
        s_store = np.zeros(num_timesteps+1)
        r_store = np.zeros(num_timesteps+1)
        qsim = np.zeros(num_timesteps+1)
        
        # for clean array indexing, add 0 element at the 0th index of prec and 
        # etp so we start simulating at the index 1
        prec = np.concatenate((np.zeros(1), prec))
        etp = np.concatenate((np.zeros(1), etp))
        
        # set initial values
        s_store[0] = self.s_init #* x1
        r_store[0] = self.r_init #* x3
        
        # calculate number of unit hydrograph ordinates
        num_uh1 = int(np.ceil(x4))
        num_uh2 = int(np.ceil(2*x4 + 1))
        
        # calculate the ordinates of both unit-hydrographs (eq. 16 & 17)
        uh1_ordinates = []
        uh2_ordinates = []
        
        for j in range(1, num_uh1 + 1):
            uh1_ordinates.append(self._s_curve1(j, x4) - self._s_curve1(j - 1, x4))
            
        for j in range(1, num_uh2 + 1):
            uh2_ordinates.append(self._s_curve2(j, x4) - self._s_curve2(j - 1, x4))
        
        uh1_ordinates = np.asarray(uh1_ordinates)
        uh2_ordinates = np.asarray(uh2_ordinates)

        
        # arrys to store the rain distributed through the unit hydrographs
        uh1 = np.zeros(num_uh1)
        uh2 = np.zeros(num_uh2)
        
        # Start the model simulation loop
        for t in range(1, num_timesteps+1):
            
            # Calculate netto precipitation and evaporation
            if prec[t] >= etp[t]:
                p_n = prec[t] - etp[t]
                pe_n = 0
            
                # calculate fraction of netto precipitation that fills
                #  production store (eq. 3)
                p_s = ((x1 * (1 - (s_store[t-1] / x1)**2) * np.tanh(p_n/x1)) /
                    (1 + s_store[t-1] / x1 * np.tanh(p_n / x1)))
                
                # no evaporation from production store
                e_s = 0   
            
            else:
                p_n = 0
                pe_n = etp[t] - prec[t]
                
                # calculate the fraction of the evaporation that will evaporate 
                # from the production store (eq. 4)
                e_s = ((s_store[t-1] * (2 - s_store[t-1]/x1) * np.tanh(pe_n/x1)) 
                    / (1 + (1 - s_store[t-1] / x1) * np.tanh(pe_n / x1)))
                
                # no rain that is allocation to the production store
                p_s = 0
                
            # Calculate the new storage content
            s_store[t] = s_store[t-1] - e_s + p_s
            
            # calculate percolation from actual storage level
            perc = s_store[t] * (1 - (1 + (4/9 * s_store[t] / x1)**4)**(-0.25))
            
            # final update of the production store for this timestep
            s_store[t] = s_store[t] - perc
            
            # total quantity of water that reaches the routing
            p_r = perc + (p_n - p_s)
            
            # split this water quantity by .9/.1 for diff. routing (UH1 & UH2)
            p_r_uh1 = 0.9 * p_r 
            p_r_uh2 = 0.1 * p_r
            
            # update state of rain, distributed through the unit hydrographs
            for j in range(0, num_uh1 - 1):
                uh1[j] = uh1[j + 1] + uh1_ordinates[j] * p_r_uh1
            uh1[-1] = uh1_ordinates[-1] * p_r_uh1
            
            for j in range(0, num_uh2 - 1):
                uh2[j] = uh2[j + 1] + uh2_ordinates[j] * p_r_uh2
            uh2[-1] = uh2_ordinates[-1] * p_r_uh2
            
            # calculate the groundwater exchange F (eq. 18)
            gw_exchange = x2 * (r_store[t - 1] / x3) ** 3.5
            
            # update routing store
            r_store[t] = max(0, r_store[t - 1] + uh1[0] + gw_exchange)
            
            # outflow of routing store
            q_r = r_store[t] * (1 - (1 + (r_store[t] / x3)**4)**(-0.25))
            
            # subtract outflow from routing store level
            r_store[t] = r_store[t] - q_r
            
            # calculate flow component of unit hydrograph 2
            q_d = max(0, uh2[0] + gw_exchange)
            
            # total discharge of this timestep
            qsim[t] = q_r + q_d
            
        # return all but the artificial 0's step
        return qsim[1:], s_store[1:], r_store[1:]