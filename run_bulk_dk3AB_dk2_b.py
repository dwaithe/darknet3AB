import subprocess
import shutil
import sys
import _pickle as pickle
sys.path.append("scripts")
import reval_custom_py3 as revc
import reval_final_list as revf
import datetime
sys.path.append("../../amca/src")
import os

from exp_proc import get_class_names, get_experiment_parameters,load_dataset

#Global variables
path_to_amca_config = "/home/molimm2/dwaithe/object_detection/amca/config/"
path_to_darknet = "/home/molimm2/dwaithe/object_detection/darknet3AB/darknet/"
path_to_data = "/home/molimm2/dwaithe/object_detection/cell_datasets/"
models_path_on_server = "/scratch/dwaithe/models/darknet/"

GPU_to_use = 0
models_itr_to_test =[ 1000, 2000, 3000, 4000, 5000,6000,7000,8000,9000,10000];
#models_itr_to_test = [5000]


#local variables
def run_experiment(exp_id):


	dmia, traina, testa, rep_a, flip_a, classes, foutfile = get_experiment_parameters(path_to_amca_config,exp_id)
	
	for darknet_models, train_on_dataset, test_on_dataset_arr, number_of_reps, flip_on, num_of_classes in zip(dmia, traina, testa, rep_a, flip_a, classes):
		
		to_train = True
		print("to_train",train_on_dataset)
		if train_on_dataset == "":
			to_train = False
		else:
		#LOADS PARAMETERS FROM THE TRAINING DATA
			num_of_train,null,null,null,dataset,path_to_training_def = load_dataset(path_to_data,path_to_amca_config,train_on_dataset)
			#check about folder exists:
			model_out = models_path_on_server+dataset+num_of_train+"/"
			print('model_out',model_out)
			if not os.path.exists(model_out):
			    os.makedirs(model_out)

		
		if flip_on == "True":
			cfg_file='yolov2_dk3AB-classes-'+str(int(num_of_classes))+'-flip'
		elif flip_on == "False":
			cfg_file='yolov2_dk3AB-classes-'+str(int(num_of_classes))+'-no-flip'
		else:
			assert False,"flip_argument is not set properly"

		print('cfg_file',cfg_file)
		
		for rep in range(0,number_of_reps):

			#Training.
			if to_train == True:
				darknet_model_init = darknet_models['darknet2_model_init']
				print('model being loaded',darknet_model_init)
				if darknet_model_init[-14:-1] == 'final.weights':
						#converts the trained model, so that they can be used for transfer learning.
						parse_string = path_to_darknet+"darknet partial cfg/"+cfg_file+".cfg "+darknet_model_init[1:-1]+" "+darknet_model_init[1:-9]+".conv.23 23"
						out = subprocess.call(parse_string, shell=True)
						
						train_path = train_path = path_to_darknet+"darknet detector train "+path_to_training_def+" "+path_to_darknet+"cfg/"+cfg_file+".cfg "+darknet_model_init[1:-9]+".conv.23 -gpus "+str(GPU_to_use)
				
				else:
					train_path = path_to_darknet+"darknet detector train "+path_to_training_def+" "+path_to_darknet+"cfg/"+cfg_file+".cfg "+darknet_model_init+" -gpus "+str(GPU_to_use)
				

				
				out = subprocess.call(train_path, shell=True)
				

			for test_on_dataset in test_on_dataset_arr:
				for test_set in test_on_dataset:
					#LOADS PARAMETERS FROM THE TESTING DATA
					null,eval_year,test_set,cell_classes,eval_dataset,path_to_testing_def = load_dataset(path_to_data, path_to_amca_config,test_set)
					#Prediction and Evaluation
					for i in models_itr_to_test:
						#Path to the weights.
						
							if to_train == True:
								weights = models_path_on_server+dataset+num_of_train+"/"+cfg_file+"_"+str(i)+".weights"
							else:
								weights = darknet_model_init				
							#Calls the darknet.
							out = subprocess.call(path_to_darknet+"darknet detector valid "+path_to_testing_def+" "+path_to_darknet+"cfg/"+cfg_file+".cfg "+weights+" -i "+str(GPU_to_use), shell=True)
							
							#Renames the output
							for cell_class in cell_classes:
								inputname = path_to_darknet+"results/comp4_det_test_"+cell_class+".txt" 
								outputname = path_to_darknet+"results/comp4_det_"+test_set+"_"+cell_class+".txt"
								shutil.move(inputname,outputname)
								
								revc.do_python_eval(path_to_data+eval_dataset, eval_year, test_set,[cell_class] , path_to_darknet+"results/", str(i))
								finalname = path_to_darknet+"results/comp4_det_"+test_set+"_"+cell_class+"_"+str(i)+".txt"
								shutil.move(outputname,finalname)



					#Printing and collation.
					output_path = models_path_on_server+"_"+foutfile
					f = open(output_path+'log.txt', 'a+')  # open file in append mode
					for i in models_itr_to_test:
						for cell_class in cell_classes:
							pick_to_open = path_to_darknet+"results/"+cell_class+"_"+str(i)+"_pr.pkl"
							data = pickle.load(open(pick_to_open,'rb'))
							out = str(datetime.datetime.now())+'\t'+str(pick_to_open)+'\tcfg_file\t'+cfg_file+'\titerations\t'+str(pick_to_open.split('_')[-2])+'\tmAP\t'+str(data['ap'])
							print("saving file to:",output_path+'log.txt')
							f.write(out+'\n')
					f.close()

if __name__ == "__main__":

	run_experiment('experiment_spec_02.txt')
