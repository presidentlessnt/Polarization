#! /usr/bin/python3
import sys
import math
import numpy as np
import scipy.stats
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")
import modulo as mtd


## Obtener datos del .txt ##

A=np.loadtxt('A.txt')
B=np.loadtxt('B.txt')
C=np.loadtxt('C.txt')
aD=np.loadtxt('D1.txt')
bD=np.loadtxt('D2.txt')


## Separar ensayos ##

A1=A[:,:2]
A2=A[:,2:8]
A3=A[:,8:11]

B1=B[:,:2]
B2=B[:,2:8]
B3=B[:,8:11]

C1=C[:,:2]
C2=C[:,2:8]
C3=C[:,8:11]

aD1=aD[:,:2]
aD2=aD[:,2:8]
aD3=aD[:,8:11]

bD1=bD[:,:2]
bD2=bD[:,2:8]


#####################################################################################
## Incertidumbre ##

def deltaI(M):
    f=len(M)
    c=len(M[0])
    nM=np.copy(M)
    I=np.array([[.6,6,60,600,6e3,10e3],[.1e-3,1e-3,.01,1,.001e3,.01e3]])
    for i in range(f):
        for j in range(1,c):
            a=M[i,j]
            if a<I[0,0]:
                nM[i,j]=a*1e-2+I[1,0]*3
            elif I[0,0]<a and a<I[0,1]:
                nM[i,j]=a*1e-2+I[1,1]*3
            elif I[0,1]<a and a<I[0,2]:
                nM[i,j]=a*1e-2+I[1,2]*3
            elif I[0,2]<a and a<I[0,3]:
                nM[i,j]=a*1e-2+I[1,3]*3
            elif I[0,3]<a and a<I[0,4]:
                nM[i,j]=a*1.2e-2+I[1,4]*5
            elif a>I[0,5]:
                nM[i,j]=a*1.2e-2+I[1,5]*5
    return nM


def incert(M,tipo):
    dI=deltaI(M)
    dp=.05*np.pi/180
    nM=np.copy(M)
    nM[:,0]*=np.pi/180
    list2=np.array([0,0,30,45,60,90])*np.pi/180
    f=len(M)
    c=len(M[0])
    if tipo==1 or tipo==3:
        for i in range(f):
            x=nM[i,0]
            for j in range(1,c):
                a=dI[i,j]/np.cos(x)**2
                b=2*M[i,j]*np.sin(x)/np.cos(x)**3*dp
                nM[i,j]=(a**2+b**2)**.5
    if tipo==2:
        for i in range(f):
            for j in range(1,c):
                x=list2[j]
                y=x-nM[i,0]
                den=(np.cos(x)*np.cos(y))**2+(np.sin(x)*np.sin(y))**2
                numx=M[i,j]*(np.sin(y)**2-np.cos(y)**2)*np.sin(2*x)*dp/den**2
                numy=M[i,j]*(np.sin(x)**2-np.cos(x)**2)*np.sin(2*y)*dp/den**2
                a=dI[i,j]/den
                nM[i,j]=(a**2+numx**2+numy**2)**.5
    nM[0]*=0
    nM[-1]*=0
    return nM

gg=incert(A2,2)
print(gg)
print(np.average(gg,axis=0))

#####################################################################################
## Ajuste de curvas ##

def func1(N,x):
    deg=np.pi/6*0 #0,1,2,3
    y=x/180*np.pi
    return np.cos(deg-y)**2*np.cos(deg)**2+np.sin(deg-y)**2*np.sin(deg)**2, (np.cos(deg-y)**2*np.cos(deg)**2+np.sin(deg-y)**2*np.sin(deg)**2)*N 

def func2(N,x):
    deg=np.pi/6 #0,1,2,3
    y=x/180*np.pi
    return np.cos(deg-y)**2*np.cos(deg)**2+np.sin(deg-y)**2*np.sin(deg)**2, (np.cos(deg-y)**2*np.cos(deg)**2+np.sin(deg-y)**2*np.sin(deg)**2)*N

def func3(N,x):
    deg=np.pi/4 #0,1,2,3
    y=x/180*np.pi
    return np.cos(deg-y)**2*np.cos(deg)**2+np.sin(deg-y)**2*np.sin(deg)**2, (np.cos(deg-y)**2*np.cos(deg)**2+np.sin(deg-y)**2*np.sin(deg)**2)*N

def func4(N,x):
    deg=np.pi/3 #0,1,2,3
    y=x/180*np.pi
    return np.cos(deg-y)**2*np.cos(deg)**2+np.sin(deg-y)**2*np.sin(deg)**2, (np.cos(deg-y)**2*np.cos(deg)**2+np.sin(deg-y)**2*np.sin(deg)**2)*N

def funcA(N,x):
    deg=0 #0,1,2,3
    y=x/180*np.pi
    return np.cos(deg-y)**2, N*np.cos(deg-y)**2


#####################################################################################
## Graficas ##

def cplot(M,i,ax=None,**plt_kwargs):
    if ax is None:
        ax=plt.figure(i) 
    list1=['']
    list2=['0°','30°','45°','60°','90°']
    listD2=['0°','60°','45°','30°','90°']
    list3=['0°','45°']
    m=len(M[0])
    for j in range(1,m):
        plt.plot(M[:,0],M[:,j],'.-',lw=1.5)
    plt.grid(ls=':',color='grey',alpha=.5)
    plt.xticks([-90,-60,-30,0,30,60,90],['-90','-60','-30','0','30','60','90'])
    plt.xlabel(r'$\varphi\;[deg]$')
    plt.ylabel('$I\;[mA]$')
    if m==6:
        legend1=plt.legend(list2,title='$\phi$',loc=2)
    elif m==3:
        legend1=plt.legend(list3,title='$\phi$',loc=2)
    elif m==2:
        legend1=plt.legend(list1,title='Sin Lámina',loc=2)

    plt.gca().add_artist(legend1)
    plt.gca().set_ylim(bottom=0)
#    plt.gca().set_ylim(top=20)
    return ax



def ajplot(M,i,ax=None,**plt_kwargs):
    if ax is None:
        ax=plt.figure(i)
    list1=['']
    list2=['0°','30°','45°','60°','90°']
    list3=['0°','45°']
    m=len(M[0])
    if m==6:
        q1=mtd.nolingen(np.column_stack((M[:,0],M[:,1])),1,func1,1e-3)
        q2=mtd.nolingen(np.column_stack((M[:,0],M[:,2])),1,func2,1e-3) # D 4
        q3=mtd.nolingen(np.column_stack((M[:,0],M[:,3])),1,func3,1e-3)
        q4=mtd.nolingen(np.column_stack((M[:,0],M[:,4])),1,func4,1e-3) # D 2
        q5=mtd.nolingen(np.column_stack((M[:,0],M[:,5])),1,func1,1e-3)
        plt.plot(M[:,0],(np.cos(0)**2*np.cos(M[:,0]/180*np.pi-0)**2+np.sin(0)**2*np.sin(M[:,0]/180*np.pi-0)**2)*q1[0][0],label=r'$E_0^2=%1.2f,\;R^2=%1.4f$'%(q1[0][0],q1[1][0]),lw=.5,color='C0')
        plt.plot(M[:,0],(np.cos(np.pi/6)**2*np.cos(M[:,0]/180*np.pi-np.pi/6)**2+np.sin(np.pi/6)**2*np.sin(M[:,0]/180*np.pi-np.pi/6)**2)*q2[0][0],label=r'$E_0^2=%1.2f,\;R^2=%1.4f$'%(q2[0][0],q2[1][0]),lw=.5,color='C1')
        plt.plot(M[:,0],(np.cos(np.pi/4)**2*np.cos(M[:,0]/180*np.pi-np.pi/4)**2+np.sin(np.pi/4)**2*np.sin(M[:,0]/180*np.pi-np.pi/4)**2)*q3[0][0],label=r'$E_0^2=%1.2f,\;R^2=%1.4f$'%(q3[0][0],q3[1][0]),lw=.5,color='C2')
        plt.plot(M[:,0],(np.cos(np.pi/3)**2*np.cos(M[:,0]/180*np.pi-np.pi/3)**2+np.sin(np.pi/3)**2*np.sin(M[:,0]/180*np.pi-np.pi/3)**2)*q4[0][0],label=r'$E_0^2=%1.2f,\;R^2=%1.4f$'%(q4[0][0],q4[1][0]),lw=.5,color='C3')
        plt.plot(M[:,0],(np.cos(np.pi/2)**2*np.cos(M[:,0]/180*np.pi-np.pi/2)**2+np.sin(np.pi/2)**2*np.sin(M[:,0]/180*np.pi-np.pi/2)**2)*q5[0][0],label=r'$E_0^2=%1.2f,\;R^2=%1.4f$'%(q5[0][0],q5[1][0]),lw=.5,color='C4')
#        plt.legend(list2,title='$\phi$')
    elif m==3:
        p1=mtd.nolingen(np.column_stack((M[:,0],M[:,1])),1,funcA,1e-3)
        p2=mtd.nolingen(np.column_stack((M[:,0],M[:,2])),1,funcA,1e-3)
        plt.plot(M[:,0],(np.cos(M[:,0]/180*np.pi)**2)*p1[0][0],label='$E_0^2=%1.2f,\;R^2=%1.4f$'%(p1[0][0],p1[1][0]),lw=.5,color='C0')
        plt.plot(M[:,0],(np.cos(M[:,0]/180*np.pi)**2)*p2[0][0],label='$E_0^2=%1.2f,\;R^2=%1.4f$'%(p2[0][0],p2[1][0]),lw=.5,color='C1')
#        plt.legend(list3,title='$\phi$')
    elif m==2:
        r1=mtd.nolingen(np.column_stack((M[:,0],M[:,1])),1,funcA,1e-3)
        plt.plot(M[:,0],(np.cos(M[:,0]/180*np.pi)**2)*r1[0][0],label='$E_0^2=%1.2f,\;R^2=%1.4f$'%(r1[0][0],r1[1][0]),lw=.5,color='C0')
    plt.grid(ls=':',color='grey',alpha=.5)
    plt.legend(title='Parámetros de ajuste')
    plt.xticks([-90,-60,-30,0,30,60,90],['-90','-60','-30','0','30','60','90'])
    plt.xlabel(r'$\varphi\;[deg]$')
    plt.ylabel('$I\;[mA]$')
    plt.gca().set_ylim(bottom=0)
#    plt.xlim(bottom=0)
    return ax



cplot(bD1,1)
cplot(bD2,2)
cplot(aD3,3)

ajplot(bD1,1)
ajplot(bD2,2)
ajplot(aD3,3)




#plt.show()



