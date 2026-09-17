from time import time
from datetime import datetime
import os
import sys
import math
import tensorflow as tf
import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import itertools
from scipy.integrate import odeint

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)
SEED=42
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
Smax = 21.37

class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""
    def __init__(self,
            output_dim=2,
            num_hidden_layers=4,
            num_neurons_per_layer=20,
            activation='tanh',
            kernel_initializer='glorot_normal',
            CMC=1,
            **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.CMC = CMC
        self.lambd = tf.Variable(self.CM()[0]*tf.ones(self.CM()[1]), trainable=True, dtype=DTYPE,
                                 constraint=lambda x: tf.clip_by_value(x, self.CM()[2], self.CM()[3]))
#         noise = tf.random.uniform(shape=self.lambd.shape, minval = -.05, maxval=.05, dtype=DTYPE, seed=42)
#         self.lambd.assign_add(noise)
        self.lambd_list = []
        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)

    def CM(self):
        if self.CMC == 1: #TEVP
            num_param = 6
            init = [1., 1., 1., 1., 1., 1.]
            low = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
            high = [np.infty, np.infty, np.infty, np.infty, np.infty, np.infty]
            return init, num_param, low, high

    def call(self, X):
        """Forward-pass through neural network."""
        Z = X
#         Z = self.scale(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)
#################################################################
class PINNSolver():
    def __init__(self, model, X_f, X_f0):
        self.model = model
        # Store collocation points
        self.x1_f = tf.gather(X_f,[0],axis=1)
        self.x2_f = tf.gather(X_f,[1],axis=1)
        self.x1_f0 = tf.gather(X_f0,[0],axis=1)
        self.x2_f0 = tf.gather(X_f0,[1],axis=1)
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0

    def fun_r(self, x1_f, x2_f, y1_pred, y2_pred, y1_t, y2_t):
        G, sy, es, ep, kp, kn = [self.model.lambd[j] for j in range(self.model.CM()[1])]
        smax = Smax
        R1 = y1_t - (G/(es + ep))*(-y1_pred + sy*y2_pred/smax + (es + ep*y2_pred)*x2_f/smax)
        R2 = y2_t - (kp*(1. - y2_pred) - kn*y2_pred*x2_f)
        return R1, R2

    def get_r(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x1_f)
            y_pred = self.model(tf.concat([self.x1_f,self.x2_f],axis=1))
            y1_pred = tf.gather(y_pred, [0], axis=1)
            y2_pred = tf.gather(y_pred, [1], axis=1)
        y1_t = tape.gradient(y1_pred, self.x1_f)
        y2_t = tape.gradient(y2_pred, self.x1_f)
        del tape
        return self.fun_r(self.x1_f, self.x2_f, y1_pred, y2_pred, y1_t, y2_t)

    def loss_fn(self, X_data, X_f0, y_data):
        R1, R2 = self.get_r()
        Loss_eq = tf.reduce_mean(tf.square(R1)) + tf.reduce_mean(tf.square(R2))

        y_pred = self.model(X_data)
        y_init = self.model(X_f0)

        Loss_init = tf.reduce_mean(tf.square(tf.gather(y_init, [0], axis=1) - 0.))\
         + tf.reduce_mean(tf.square(tf.gather(y_init, [1], axis=1) - 1.))

        Loss_data = tf.reduce_mean(tf.square(y_data - tf.gather(y_pred, [0], axis=1)))

        #loss = tf.math.sqrt(Loss_eq) + tf.math.sqrt(Loss_init) + tf.math.sqrt(Loss_data)#*Loss_max
        loss = Loss_data + Loss_eq + Loss_init
        loss_frac = [Loss_eq, Loss_data]
        return loss, loss_frac

    def get_grad(self, X_data, X_f0, y_data):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            loss, loss_frac = self.loss_fn(X_data, X_f0, y_data)
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, g, loss_frac

    def solve_with_TFoptimizer(self, optimizer, X_data, X_f0, y_data, N=1001):
        """This method performs a gradient descent type optimization."""
        @tf.function
        def train_step():
            loss, grad_theta, loss_frac = self.get_grad(X_data, X_f0, y_data)
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, loss_frac

        for i in range(N):
            loss, loss_frac = train_step()
            self.loss_frac = loss_frac
            self.current_loss = loss.numpy()
            self.callback()

    def solve_with_ScipyOptimizer(self, X, X_f0, u, method='L-BFGS-B', **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in Fortran
        which requires 64-bit floats instead of 32-bit floats."""

        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""

            weight_list = []
            shape_list = []

            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())

            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()

        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape

                # Weight matrices
                if len(vs) == 2:
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw

                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]

                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1

                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))

        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""

            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, grad, loss_frac = self.get_grad(X, X_f0, u)

            # Store current loss for callback function            
            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss
            self.loss_frac = loss_frac

            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())

            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)

            # Return value and gradient of \phi as tuple
            return loss, grad_flat

        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)

    def callback(self, xr=None):
        lambd = self.model.lambd.numpy()
        self.model.lambd_list.append(lambd)
        if self.iter % 5000 == 0:
            tf.print('It {:05d}: loss = {:10.4e}, {}'.format(self.iter, self.current_loss, np.round(lambd, 3)))
        self.hist.append(self.current_loss)
        self.iter+=1

#####################################################################
def TEVP(params, smax, tmin, tmax, gdot, N_d_ode):
    G, sy, es, ep, kp, kn = np.array(params)

    def odes(x, t):
        S = x[0]
        L = x[1]
        dSdt = (G/(es + ep))*(-S + sy*L/smax + (es + ep*L)*gdot/smax)
        dLdt = kp*(1 - L) - kn*L*gdot
        return [dSdt, dLdt]

    x0 = [0., 1.]
    t = np.logspace(np.log10(tmin),np.log10(tmax), N_d_ode)
    x = odeint(odes, x0 ,t)
    SS = x[:,0]
    L = x[:,1]
    SR = gdot*np.ones(N_d_ode)
    return t, SR, SS, L

##################################################################
def generate_data(N_g,gmin,gmax,tmin,tmax,Smax,N_d):
    Time, ShearRate, ShearStress, Lambda = [], [], [], []
    shear = np.linspace(gmin, gmax, N_g)
    param_exact = [50, 10, 10., 5, 0.1, 0.3] #G, sy, es, ep, kp, kn
    for gdot in shear:
        t, SR, SS, L = TEVP(param_exact, Smax, tmin, tmax, gdot, N_d)
        Time = np.append(Time, t, axis=None)
        ShearRate = np.append(ShearRate, SR, axis=None)
        ShearStress = np.append(ShearStress, SS, axis=None)
        Lambda = np.append(Lambda, L, axis=None)
    df = pd.DataFrame({"Time" : Time, "ShearRate" : ShearRate, "ShearStress" : ShearStress, "Lambda" : Lambda})
    df.to_excel("StartUp1.xlsx", index=False)

################################################################
def read_data():
    CURVE, Shuffle = 'StartUp1', True
    xlsx = pd.ExcelFile('{}.xlsx'.format(CURVE))
    df = pd.read_excel(xlsx, sheet_name=None)
    data = [[k,v] for k,v in df.items()] #k is the sheet name, v is the pandas df
    
    sample=0
    # sample data points
    x1_d = tf.reshape(tf.convert_to_tensor(data[sample][1]['Time'], dtype=DTYPE),(-1,1))
    x2_d = tf.reshape(tf.convert_to_tensor(data[sample][1]['ShearRate'], dtype=DTYPE), (-1,1))
    y1_d = tf.reshape(tf.convert_to_tensor(data[sample][1]['ShearStress'], dtype=DTYPE), (-1,1))
    y2_d = tf.reshape(tf.convert_to_tensor(data[sample][1]['Lambda'], dtype=DTYPE), (-1,1))
    
    tmin, tmax = np.min(x1_d), np.max(x1_d)
    xmin, xmax = np.min(x2_d), np.max(x2_d)
    lb = tf.constant([tmin, xmin], dtype=DTYPE).numpy()
    ub = tf.constant([tmax, xmax], dtype=DTYPE).numpy()
    
    #Collocation points
    N_d = 201
    t_f=np.logspace(np.log10(lb[0]), np.log10(ub[0]), N_d)
    x_f=np.linspace(lb[1], ub[1],4)
    X_f=list(itertools.product(t_f, x_f))
    X_f=tf.convert_to_tensor(X_f,dtype=DTYPE)
    
    #Initial points
    N_i = 50
    x1_f0 = tf.convert_to_tensor(np.linspace(lb[0], lb[0], N_i).reshape(-1,1),dtype=DTYPE)
    x2_f0 = tf.convert_to_tensor(np.linspace(lb[1], ub[1], N_i).reshape(-1,1),dtype=DTYPE)
    X_f0 = tf.concat([x1_f0, x2_f0], axis = 1)
    
    X_data = tf.concat([x1_d, x2_d], axis=1)
    y_data = tf.concat([y1_d], axis=1)  
    Xy_data = tf.concat([x1_d, x2_d, y1_d], axis=1)
    
    if Shuffle:
        X_f = tf.random.shuffle(X_f)
        X_f0 = tf.random.shuffle(X_f0)
        Xy_data = tf.random.shuffle(Xy_data)
        X_data = Xy_data[:,0:2]
        y_data = Xy_data[:,2:3]
    return lb, ub, X_f, X_f0, X_data, y_data
#############################################################
if __name__ == '__main__':
    N_g = 4
    gmin = .1
    gmax = 1.
    Smax = 21.37#16.558
    tmin, tmax = 0.01, 100.
    N_d = 201
    in_dim, out_dim = 2, 2
    #Generate data and save in Startup1 file
    generate_data(N_g,gmin,gmax,tmin,tmax,Smax,N_d)
    #Read data from Startup1 file
    lb, ub, X_f, X_f0, X_data, y_data = read_data()

    #Create model and model solver
    model = PINN_NeuralNet(output_dim=out_dim,num_hidden_layers=4,num_neurons_per_layer=10,CMC=1)
    model.build(input_shape=(None,in_dim))
    solver = PINNSolver(model, X_f, X_f0)
   
    #Train model using Adam optimizer 
    mode = 'TFoptimizer'
    t0 = time()
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([60000,100000],[1e-3,1e-3,1e-3])
    N = int(2e5 + 1)
    optim = tf.keras.optimizers.Adam(learning_rate = lr)
    solver.solve_with_TFoptimizer(optim, X_data, X_f0, y_data, N=int(N))
    #solver.solve_with_ScipyOptimizer(X_data, X_f0, y_data, method='L-BFGS-B',
    #                            options={'maxiter': 500000,'maxfun': 500000,'maxcor': 1000,
    #                                                  'maxls': 1000,'ftol' : 1.0 * np.finfo(float).eps})
    model.save_weights('my_model_f.tf')
    model.save('my_model_f')
