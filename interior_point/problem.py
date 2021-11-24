from numpy.core.shape_base import hstack
from casadi import *
import numpy  as np
from scipy.linalg import null_space
from numpy import linalg as LA
import scipy.linalg as scl


class interior_point_problem:
    def __init__(self, x, F, G, H, maxiter, xk, lk, vk, sk, nv, ne, ni):
        self.max_iter = maxiter
        self.G_ = G
        self.H_ = H
        self.F_ = F
        # Jacobian of f
        self.J_f = Function('Jf',[x],[jacobian(F(x),x)])

        # Jacobian of g
        self.J_g = Function('Jg',[x],[jacobian(G(x),x)])

        # Jacobian of h
        self.J_h = Function('Jh',[x],[jacobian(H(x),x)])

        self.H_f = Function('Hf',[x],[hessian(F(x),x)[0]])


        # Hessian of g
        self.H_g = Function('Hg',[x],[hessian(G(x),x)[0]])

        # Hessian of h
        self.H_h = Function('Hh',[x],[hessian(H(x),x)[0]])

        #algorithm parameters
        self.tau = 2
        self.k_b =0.33
        self.th_1 = 1e-8
        self.th_2 = 1e-8
        self.iter = np.zeros([nv+ne+ni+ni, maxiter])
        self.iter[:,0] = np.hstack([xk, lk, vk, sk])
        try:
            xk.shape[1]
        except:
            xk.resize([xk.shape[0],1])

        try:
            lk.shape[1]
        except:
            lk.resize([lk.shape[0],1])        
        
        try:
            vk.shape[1]
        except:
            vk.resize([vk.shape[0],1])

        try:
            sk.shape[1]
        except:
            sk.resize([sk.shape[0],1])        
        
        self.xk = xk
        self.lk = lk
        self.vk = vk
        self.sk = sk
        self.nv= nv
        self.ne = ne
        self.ni = ni
        self.alpha = 1
        self.Jg_tilde = None
        self.strict_complementarity = True
        self.sol = True
    def Jf(self,x):
        return np.array(self.J_f(x))

    def Jg(self,x):
        return np.array(self.J_g(x))
    
    def Jh(self,x):
        return np.array(self.J_h(x))

    def Hf(self,x):
        return np.array(self.H_f(x))

    def Hg(self,x):
        return np.array(self.H_g(x))
    
    def Hh(self,x):
        return np.array(self.H_h(x))

    def F(self,x):
        return np.array(self.F_(x))

    def G(self,x):
        return np.array(self.G_(x))

    def H(self,x):
        return np.array(self.H_(x))


    def solve(self):
        nv = self.nv
        ne = self.ne 
        ni = self.ni 
        xk = self.xk 
        lk = self.lk 
        vk = self.vk 
        sk = self.sk
        for i in range(0,self.max_iter):
            print(i)

            g_e     = self.G(xk)
            h_e     = self.H(xk)
            
            Jg_e    = self.Jg(xk)
            Jh_e    = self.Jh(xk)
            Jf_e    = self.Jf(xk)
            
            Hf_e    = self.Hf(xk)
            Hg_e    = self.Hg(xk)
            Hh_e    = self.Hh(xk)

            Hl      = Hf_e + Hg_e*lk + Hh_e*vk
            #print(Hl)
            
            sk_diag = np.diag(sk)
            vk_diag = np.diag(vk)
            try:
                sk_diag.shape[1]
            except:
                sk_diag.resize([sk_diag.shape[0],1])
            try:
                vk_diag.shape[1]
            except:
                vk_diag.resize([vk_diag.shape[0],1]) 

            #2nd order KKT matrix
            M = np.vstack([np.hstack([  Hl,  Jg_e.T,  Jh_e.T,     np.zeros([self.nv, self.ni])]),
                    np.hstack([Jg_e,   np.zeros([self.ne,self.ne]),	np.zeros([self.ne,self.ni]),	np.zeros([self.ne,self.ni])]),
                    np.hstack([Jh_e, np.zeros([self.ni,self.ne]),	 np.zeros([self.ni,self.ni]),	np.eye(self.ni)]),
                    np.hstack([np.zeros([self.ni,self.nv]), 	 np.zeros([self.ni,self.ne]),	sk_diag, vk_diag])])        

            #Handling the case where the dimension of np.dot(Jg_e.T,lk) becomes (n,). It should be (n,1) to be handled properly by numpy
            a=np.dot(Jg_e.T,lk)
            b=np.dot(Jh_e.T,vk)
            try:
                a.shape[1]
            except:
                a.resize([a.shape[0],1])

            try:
                b.shape[1]
            except:
                b.resize([b.shape[0],1])

            # R rhs of the equation
            rhs = - np.vstack([a + Jf_e.T + b, 
                            g_e , 
                            h_e + sk ,
                            np.dot(vk,sk) - self.tau ])
            
            z_step = np.dot(np.linalg.inv(M),rhs)
            print(M)
            #Lb = scl.cho_factor(M)
            #z_step= scl.cho_solve(Lb, rhs)
            print('_____________---')
            print(np.linalg.norm(rhs))
            print(z_step.shape)
    
            #Termination condition
            if np.linalg.norm(rhs) < self.th_1:         # if smoothed system is solved for current tau
                if self.tau < self.th_2:                     # if tau is small enough
                    print('Solution found!')
                    break
                else:
                    # decrease tau and continue
                    self.tau = self.tau*self.k_b
                    print('oh yeah')

            # alpha = self.line_search(z_step)
            x_step  = z_step[:nv,:]
            l_step  = z_step[nv:nv+ne,:]
            v_step  = z_step[nv+ne:nv+ne+ni,:]
            s_step  = z_step[nv+ne+ni:,:]
            
            print(x_step.shape)
            print(x_step.shape)
            print(v_step.shape)
            print(s_step.shape)
            print('_____________---')
            #line search part
            alpha =1
            max_ls =100
            k_ls = 0.9
            min_step = 1.0e-8
            for j in range(max_ls):    
                # Compute trial step
                v_t = vk + alpha * v_step
                s_t = sk + alpha * s_step
                if all(v_t >= 0) and all(s_t >= 0):
                    alpha = alpha
                    break

                
                #Decrease alpha
                alpha = alpha * k_ls
                
                # Terminiation condition
                if np.linalg.norm(alpha*np.array([ v_step,s_step])) < min_step:
                    print('Line search failed! Could not find dual feasible step.')
                    self.sol = False
                    return

            
            # actual step
            #print(xk)
            xk  = xk + alpha*x_step
            lk  = lk + alpha*l_step
            vk  = vk + alpha*v_step
            sk  = sk + alpha*s_step

            #print(xk)
            #print(alpha)
            # save for later processing
            temp = np.vstack([xk, lk, vk, sk])

            self.iter[:,i] = temp.resize([temp.shape[0]])

            #every now and then reprint header
            # if i%20 == 1:          
            #     print('it \t tau \t\t ||rhs|| \t alpha\n')


            # #Print some info
            # print('%d \t %e \t %e \t %e\n',i, self.tau, np.linalg.norm(rhs), alpha)
        if self.sol==False:
            print('Line search failed! Could not find dual feasible step.')

        tol =1e-6
        self.sosc(h_e, Jg_e, Jh_e, vk, tol)
        #compute the reduced hessian
        Z = null_space(self.Jg_tilde)
        redH = np.dot(np.dot(Z.T, Hl),Z)
        eigs = LA.eig(redH)
        mineig = min(eigs)
        if ~self.strict_complementarity:
            print('strict complentarity does not hold.')
            print('The conditions for the the theorem second order optimality conditions are not fulfilled.')
        elif len(Z[0]) or  mineig > tol:
            print('redH > 0. SOSC (and SONC) is fullfilled')
            print('The solution is a local minimizer.')
        elif mineig >= -tol:
            print('redH >= 0. SONC is fullfilled')
            print('The solution might be a local minimizer.')
        else:
            print('redH not PSD. Neither SONC nor SOSC hold.')
            print('The solution is not a local minimizer.')


    def line_search(self, z_step):
        #Ensure that slack variable and lagrange multiplier corresponding to inequality constraints are positive
        max_ls = 100
        alpha = self.alpha
        k_ls = 0.9
        min_step = 1.0e-8
        nv = self.nv
        ne = self.ne 
        ni = self.ni 

        xk = self.xk 
        lk = self.lk 
        vk = self.vk 
        sk = self.sk

        x_step  = z_step[1:nv]
        l_step  = z_step[nv+1:nv+ne]
        v_step  = z_step[nv+ne+1:nv+ne+ni]
        s_step  = z_step[nv+ne+ni+1:-1]

        for j in range(max_ls):    
            # Compute trial step
            v_t = vk + alpha * v_step
            s_t = sk + alpha * s_step
            if all(v_t >= 0) and all(s_t >= 0):
                self.alpha = alpha
                break

            
            #Decrease alpha
            alpha = alpha * k_ls
            
            # Terminiation condition
            if np.linalg.norm(alpha*np.array([ v_step,s_step])) < min_step:
                print('Line search failed! Could not find dual feasible step.')
                self.sol = False
                return
        return self.alpha

    def sosc(self, h_e, Jg_e, Jh_e, vk, tol):
        if np.linalg.norm(h_e)<=tol:
            Jg_tilde = np.vstack(Jg_e,Jh_e)  #augment the list of active constraints
            if vk >= tol:
                print('h is strictly active')
            else:
                print('h is weakly active')
                self.strict_complementarity = False
        else:
            print('h is inactive')
            self.Jg_tilde = Jg_e
    