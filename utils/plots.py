import os
import numpy as np
from matplotlib import pyplot as plt
from .utils import * 


def plot_result(config, basicModel,  gen):    
    nb_samples  = config["plots"]["nb_samples"]
    ft = 8
    plt.rcParams.update({'font.size': ft})
    nbplt = 2
    x,y ,coh, mn = next(gen)
    
    for j in range(nb_samples):
        plt.subplot(nb_samples, nbplt, 1+j*nbplt)
        plt.plot(range(0, len(x[j,:,:])), x[j,:,:])
        plt.ylabel("Input", fontsize=ft)
        plt.xlabel("Time (ms)", fontsize=ft)
        plt.tick_params(axis='both', which='major', labelsize=ft)
        plt.tick_params(axis='both', which='minor', labelsize=ft)
        if (config["rdm"]["discretization"] != 0):
            plt.ylim([-config["rdm"]["discretization"],config["rdm"]["discretization"]])
        else:
            plt.ylim([-1,1])
        plt.legend(["coh= "+str(round(coh[j],2))])
        results = basicModel.test(x)
        output = results[0]
        state_var = results[1]    
    
        fig = plt.subplot(nb_samples, nbplt, 2+j*nbplt)        
        plt.plot(range(0, len(y[j,:,:])), y[j,:,:])
        
        #s2 = plt.subplot(nb_samples, nbplt, 2+j*nbplt)
        fig.plot(range(0, len(output[j,:,:])),output[j,:,:])
        #plt.ylabel("P", fontsize=ft)
        plt.legend(["output", "prediction"])
    
        plt.xlabel("Time (ms)", fontsize=ft)
        plt.tick_params(axis='both', which='major', labelsize=ft)
        plt.tick_params(axis='both', which='minor', labelsize=ft)
    
    
    plt.savefig(config["plots"]["save_dir"] + config["plots"]["training_results_file"] + "_" + save_suffix(config) + ".png", dpi=72)    

def plot_characterizations(config, basicModel,  gen):    
    # Create the psychometric/chronometric function
    x = []
    y = []
    cohs = []
    cohs_actual = []
    output = []
    for i in range(0, 100):
        _x, _y, _cohs, _cohs_actual = next(gen)
        x.append(_x)
        y.append(_y)
        cohs.append(_cohs)
        cohs_actual.append(_cohs_actual)
        _output = basicModel.test(_x)
        output.append(_output[0])
    
    x = np.concatenate(x)
    y = np.concatenate(y)
    cohs = np.concatenate(cohs)
    cohs_actual = np.concatenate(cohs_actual)
    output = np.concatenate(output).squeeze()
    
    THRESH =  config["plots"]["thresh_correct_response"]   # What counts as a correct response?
    # Find the first cross of the threshold for each row
    first_cross_idx = np.argmax(np.abs(output)>THRESH, axis=1)
    # Set it to the trial length if it never crossed
    first_cross_idx[first_cross_idx==0] = len(output[0])-1
    
    corr = (np.sign(cohs) == np.sign(output[range(0,len(output)),first_cross_idx])) * (first_cross_idx != 0)
    rts = first_cross_idx*config["rdm"]["dt"] - config["rdm"]["fixation_dur"]
    
    # Psychometric function
    rt_means = []
    corr_probs = []
    task_cohs = np.asarray(list(sorted(set(np.abs(cohs)))))
    for coh in task_cohs:
        rt_means.append(np.mean(rts[np.logical_and(coh==np.abs(cohs), corr)]))
        corr_probs.append(np.mean(corr[coh==np.abs(cohs)]))
    print("Corr probs:", corr_probs)

    plt.figure()
    log_coh = np.log10(task_cohs)
    plt.plot(log_coh, corr_probs)
    plt.title("Psychometric function")
    plt.xlabel("Log(coherence odds)")
    plt.ylabel("P(Correct)")
    plt.xticks(log_coh, labels=task_cohs)
    plt.yticks([.5, .75, 1])
    plt.axis([-2, None, None, None])
    plt.savefig(config["plots"]["save_dir"] + config["plots"]["psychometric_file"]  + "_" + save_suffix(config) + ".png", dpi=72)    
    
    
    plt.figure()
    plt.plot(log_coh, rt_means)
    plt.title("Chronometric function")
    plt.xlabel("Log(coherence odds)")
    plt.ylabel("Mean response time")
    plt.xticks(log_coh, labels=task_cohs)
    plt.axis([-2, None, None, None])
    
    plt.savefig(config["plots"]["save_dir"] + config["plots"]["chronometric_file"]  + "_" + save_suffix(config) + ".png", dpi=72)    