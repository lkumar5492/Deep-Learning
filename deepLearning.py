import sys
from hw_utils import loaddata
from hw_utils import normalize
from hw_utils import testmodels
from math import exp
import time 


if __name__ == "__main__":

	# PART (C)
	trainingFeature, trainingTarget, testFeature, testTarget = loaddata("MiniBooNE_PID.txt")
	normTrainingFeature, normTestFeature = normalize(trainingFeature, testFeature)
	
	# PART (D)
	din = 50
	dout = 2
	archs = [[din, dout], [din, 50, dout], [din, 50, 50, dout], [din, 50, 50, 50, dout]]
	actfn = 'linear'
	last_act = 'softmax'
	reg_coeffs= [0.0]
	num_epoch = 30
	batch_size = 1000
	sgd_lr = 0.001
	sgd_decays = [0.0]
	sgd_moms = [0.0]
	sgd_Nesterov = False
	EStop = False
	verbose = 0
	# PART (D) ARCH = [[din, dout], [din, 50, dout], [din, 50, 50, dout], [din, 50, 50, 50, dout]]
	start = time.time()
	testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""

	archs = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]

	# PART (D) ARCH = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
	start = time.time()
	testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""

	#PART (E)
	archs = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
	actfn = 'sigmoid'
	start = time.time()
	testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""

	#PART (F)
	actfn = 'relu'
	sgd_lr = 5e-4
	start = time.time()
	testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""

	#PART (G)
	archs = [[din, 800, 500, 300, dout]]
	actfn = 'relu'
	sgd_lr = 5e-4
	reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
	start = time.time()
	testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""

	#PART (H)
	EStop = True
	start = time.time()
	best_config = testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""
	prevBestRegCoeff = best_config[1]

	#PART (I)
	archs = [[din, 800, 500, 300, dout]]
	actfn = 'relu'
	reg_coeffs = [5e-7]
	num_epoch = 100
	batch_size = 1000
	sgd_lr = 1e-5
	sgd_decays = [1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3]	
	EStop = False	
	start = time.time()
	best_config = testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""

	#PART (J)
	prevBestDecay = best_config[2]

	reg_coeffs= [0.0]
	num_epoch = 50
	batch_size = 1000
	sgd_lr = 1e-5
	sgd_decays = [prevBestDecay]
	sgd_moms = [ 0.99, 0.98, 0.95, 0.9, 0.85]
	sgd_Nesterov = True
	EStop = False
	start = time.time()
	best_config = testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""

	#PART (K)	
	prevBestMom = best_config[3]

	reg_coeffs= [prevBestRegCoeff]
	sgd_decays = [prevBestDecay]
	sgd_moms = [prevBestMom]

	num_epoch = 100
	batch_size = 1000
	sgd_lr = 1e-5
	sgd_Nesterov = True
	EStop = True
	start = time.time()
	testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""

	#PART (L)
	archs = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
	actfn = 'relu'
	last_act = 'softmax'
	reg_coeffs= [ 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
	num_epoch = 100
	batch_size = 1000
	sgd_lr = 1e-5
	sgd_decays = [ 1e-5, 5e-5, 1e-4]
	sgd_moms = [0.99]
	sgd_Nesterov = True
	EStop = True
	start = time.time()
	testmodels(normTrainingFeature, trainingTarget, normTestFeature, testTarget, archs, actfn, last_act, 
		reg_coeffs, num_epoch, batch_size, sgd_lr, sgd_decays, sgd_moms, sgd_Nesterov, EStop, verbose)
	end = time.time()
	print ""
	print "Time Taken: " + str(end-start)
	print ""
	
