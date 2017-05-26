import math
import sys
import time
import cProfile
import pickle
import datetime
import cProfile
import numpy as np
import sparse
from pylab import *
from scipy import sparse
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize

#input: the target matrix P
#       the target labels Yp
#       the source matrix Q
#       the parameters params
# 		the similarity among instances S
#

def initialization(n_t, d_t, n_s, d_s, k):

	# Random initialization
	U = np.random.rand(n_t, k) - 0.5
	V_t = np.random.rand(k, d_t) - 0.5

	W = np.random.rand(n_s, k) - 0.5
	V_s = np.random.rand(k, d_s) - 0.5

	return U, V_t, W, V_s

#P and Q are required to be sparse
def compHTL(P, Yp, Q, S, k=5, max_iter=20, beta=0.1, gamma=0.1):

	#Initialize U, W, V1, V2
	n_t, d_t = P.shape
	n_s, d_s = Q.shape

	print "NS", n_s

	U, V_t, W, V_s = initialization(n_t, d_t, n_s, d_s, k)

	#Start to optimize
	loss_list = [calculateLoss(P, Q, U, W, V_t, V_s, S, beta, gamma)]
	stopFlag = False;
	it = 0;

	while (it < max_iter and not stopFlag):

		print "%d Epoch Loss=:%f" % (it, loss_list[-1])
		#First update U, V_t, and V_s
		#Then update W
		#Calculate Sw
		Sw = np.diag(np.sum(S, 0))
		[W, evl] = updateW(W, Q, U, V_s, S, Sw, beta, gamma)

		##Second fix W, V1, V2, update U
		Su = np.diag(np.sum(S, 1))
		[U, evl] = updateU(U, P, W, V_t, S, Su, beta, gamma)

		##Fix U, W, V2 and update V1
		# UpP = U.T.dot(P)
		# UpU = U.T.dot(U)
		UpP = U.T * P
		UpU = U.T.dot(U)

		V_t = (np.linalg.inv(UpU + gamma * np.eye(k))).dot(UpP)

		##Fix U, W, V1 and update V2
		WpQ = W.T * Q
		WpW = W.T.dot(W)

		V_s = (np.linalg.inv(WpW + gamma * np.eye(k))).dot(WpQ)

		loss_iter = calculateLoss(P, Q, U, W, V_t, V_s, S, beta, gamma)
		# print "Epoch:%d Loss=%f" % (it, loss_list[-1])
		loss_list.append(loss_iter)
		it += 1

	return loss_list, U, W, V_t, V_s

def calculateLoss(P, Q, U, W, V_t, V_s, S, beta, gamma):

	n_t, d_t = P.shape
	n_s, d_s = Q.shape

	obj = (np.linalg.norm(P - U.dot(V_t))**2 + np.linalg.norm(Q - W.dot(V_s))**2  + gamma
		* (np.linalg.norm(U)**2 + np.linalg.norm(W)**2 + np.linalg.norm(V_t)**2 + np.linalg.norm(V_s)**2))

	for i in range(n_t):
		for j in range(n_s):
			if S[i,j] == 0:
				continue

			obj += S[i,j] * (np.linalg.norm(U[i,:] - W[j,:]) ** 2)

	return obj

def updateW(W0, Q, U, V_s, S, Sw, beta, gamma):

	n_s = Q.shape[0]
	k = W0.shape[1]

	W0 = W0.reshape(n_s * k);
	MaxFuncEvals = 10

	def lossW(W):
		W = W.reshape(n_s, k)
		loss = (np.linalg.norm(Q-W.dot(V_s))**2 + beta * np.trace(W.T.dot(Sw).dot(W))
			- 2 * beta * np.trace(U.T.dot(S).dot(W)) + gamma * np.linalg.norm(W)**2)

		loop_loss = 0
		for i in range(S.shape[0]):
			for j in range(S.shape[1]):
				loop_loss += S[i,j] * (np.linalg.norm(U[i,:] - W[j,:]) ** 2)

		# print "Loss W =:%f" % loss
		return loss

	def gradW(W):
		W = W.reshape(n_s, k)
		grad = (-2 * Q * V_s.T + 2 * W.dot(V_s.dot(V_s.T)) + 2 * beta * Sw.dot(W)
			- 2 * beta * S.T.dot(U) + 2 * gamma * W)
		grad = grad.reshape(n_s * k)
		# print grad
		return grad

	res = minimize(fun=lossW, x0=W0, method="CG", jac=gradW, options={'maxiter':MaxFuncEvals, 'disp':False})
	W = res.x.reshape(n_s, k)
	evl = res.fun

	return W, evl

def updateU(U0, P, W, V_t, S, Su, beta, gamma):
	n_t = P.shape[0]
	k = U0.shape[1]

	U0 = U0.reshape(n_t * k)
	MaxFuncEvals = 10

	def lossU(U):
		U = U.reshape(n_t, k)
		loss = (np.linalg.norm(P - U.dot(V_t))**2 + beta * np.trace(U.T.dot(Su).dot(U))
			- 2 * beta * np.trace(U.T.dot(S).dot(W)) + gamma * np.linalg.norm(U)**2)
		# print "Loss U =%f" % loss
		return loss

	def gradU(U):
		#g = -2*P*V1'+2*U*(V1*V1')+2*param.beta*Su*U-2*param.beta*S*W+2*param.r1*U;
		U = U.reshape(n_t, k)
		grad = (-2 * P * V_t.T + 2 * U.dot(V_t.dot(V_t.T)) + 2 * beta * Su.dot(U)
		 - 2 * beta * S.dot(W) + 2 * gamma * U)

		grad = grad.reshape(n_t * k)
		return grad

	res = minimize(fun=lossU, x0=U0, method="CG", jac=gradU, options={'maxiter':MaxFuncEvals, 'disp':False})
	U = res.x.reshape(n_t, k)
	evl = res.fun

	return U, evl

def Similarity(s,t):  # compute the similarity matrx of user-app and user-article
    a = shape(s)
    b = shape(t)
    simMatrix=[]
    for i in range (a):
        for j in range (b):
            if s['user_id']==t['user_id']:
                simMatrix[i][j]=1
            else:
                simMatrix[i][j]=0
    return simMatrix



if __name__ == "__main__":

	# N_t = 100
	# d_t = 50
	# N_s = 500
	# d_s = 80
    #
	# #P = np.random.rand(N_t, d_t)
	# P = sparse.random(N_t, d_t, format='csr', density=0.3)
	# Yp = np.random.rand(N_t)
    #
	# #Q = np.random.rand(N_s, d_s)
	# Q = sparse.random(N_s, d_s, format='csr', density=0.3)
    #
	# S = np.random.rand(N_t, N_s)
	# S[S < 0.8] = 0
	# S[S > 0.8] = 1

    p = pickle.load(open('user_apps_tl_mat', 'r'))
    P = sparse(p, format='csr', density=0.3)
    Yp = pickle.load(open('user_apps_tl_label_mat', 'r'))
    Q = pickle.load(open('encoded_onlyarticle_data', 'r'))
    S = Similarity(P,Q)

    loss_list, U, W, V_t, V_s = compHTL(P, Yp, Q, S, k=5, max_iter=100, beta=0.01, gamma=0.01)
	# cProfile.run("loss_list, U, W, V_t, V_s = compHTL(P, Yp, Q, S, k=5, max_iter=100, beta=0.01, gamma=0.01)")
	# plot(loss_list, "*-", linewidth=3)
	# show()