# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 22:01:34 2019

@author: lee
"""
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
def plot1(): 
    
    
    a=np.loadtxt("tr.txt", dtype="float")
    plt.figure(figsize=(6, 4))
    plt.plot(a[:,0], a[:,2], color='c', linewidth=1.7, label='Lenet')
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig('lti.png')
    plt.show()
    
def plot2(): 
      
    a=np.loadtxt("t.txt", dtype="float")
    plt.figure(figsize=(6, 4))
    #plt.plot(a[:,0], a[:,2], color='c', linewidth=1.7, label='Lenet')
    plt.plot(a[:,0], a[:,3], color='c', linewidth=1.7, label='Lenet')
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    #plt.ylabel('Test Accuracy')
    plt.ylabel('Test Loss')
    plt.legend()
    #plt.savefig('lTAi.png')
    plt.savefig('lTLi.png')
    plt.show()   
    
def plot3(): 
      
    a=np.loadtxt("t2.txt", dtype="float")
    plt.figure(figsize=(6, 4))
    plt.plot(a[:,0], a[:,2], color='c', linewidth=1.7, label='Alexnet')
    #plt.plot(a[:,0], a[:,3], color='c', linewidth=1.7, label='Alexnet')
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')
    #plt.ylabel('Test Loss')
    plt.legend()
    plt.savefig('ATAi.png')
    #plt.savefig('ATLi.png')
    plt.show()
    
def plot4(): 
      
    a=np.loadtxt("tr2.txt", dtype="float")
    plt.figure(figsize=(6, 4))
    #plt.plot(a[:,0], a[:,2], color='c', linewidth=1.7, label='Alexnet')
    plt.plot(a[:,0], a[:,3], color='c', linewidth=1.7, label='Alexnet')
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    #plt.ylabel('Training Loss')
    plt.ylabel('Learning rate')
    plt.legend()
    plt.savefig('ali1.png')
    #plt.savefig('ati1.png')
    plt.show()
    
def plot5(): 
      
    a=np.loadtxt("tr3.txt", dtype="float")
    plt.figure(figsize=(6, 4))
    plt.plot(a[:,0], a[:,2], color='c', linewidth=1.7, label='Alexnet')
    #plt.plot(a[:,0], a[:,3], color='c', linewidth=1.7, label='Alexnet')
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    #plt.ylabel('Learning rate')
    plt.legend()
    #plt.savefig('ali2.png')
    plt.savefig('ati2.png')
    plt.show()
    
def plot6():       
    a=np.loadtxt("t3.txt", dtype="float")
    plt.figure(figsize=(6, 4))
    #plt.plot(a[:,0], a[:,2], color='c', linewidth=1.7, label='Alexnet')
    plt.plot(a[:,0], a[:,3], color='c', linewidth=1.7, label='Alexnet')
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    #plt.ylabel('Test Accuracy')
    plt.ylabel('Test Loss')
    plt.legend()
    #plt.savefig('ATAi2.png')
    plt.savefig('ATLi2.png')
    plt.show()

def plot7(): 
      
    a=np.loadtxt("tr4.txt", dtype="float")
    plt.figure(figsize=(6, 4))
    #plt.plot(a[:,0], a[:,2], color='c', linewidth=1.7, label='Alexnet')
    plt.plot(a[:,0], a[:,3], color='c', linewidth=1.7, label='Alexnet')
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    #plt.ylabel('Training Loss')
    plt.ylabel('Learning rate')
    plt.legend()
    plt.savefig('ali3.png')
    #plt.savefig('ati3.png')
    plt.show()
    
def plot8():       
    a=np.loadtxt("t4.txt", dtype="float")
    plt.figure(figsize=(6, 4))
    plt.plot(a[:,0], a[:,2], color='c', linewidth=1.7, label='Alexnet')
    #plt.plot(a[:,0], a[:,3], color='c', linewidth=1.7, label='Alexnet')
    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')
    #plt.ylabel('Test Loss')
    plt.legend()
    plt.savefig('ATAi3.png')
    #plt.savefig('ATLi3.png')
    plt.show()   
if __name__ == '__main__':
    plot7()
    plot8()
