#!/usr/bin/env python
# coding: utf-8

# In[63]:


crp_arr = [128, 7, 2]
fc = 1024

def value_crp() :
    return crp_arr
def value_fc() :
    return fc

def crpdata() :
    crpo = value_crp()
    crp_arr = update_crp(value_crp())
    return crpo

def update_crp(crp_arr) :
    if crp_arr[0] != 1024 :
        crp_arr[0] = crp_arr[0]*2
    if crp_arr[1] > 3 :
        crp_arr[1] = crp_arr[1]-1
    return crp_arr

def fcdata() :
    fco = value_fc()
    global fc
    fc = update_fc(fc)
    return fco

def update_fc(var):
    if var != 4096 :
        var = var + 1024
    return var


# In[87]:


import random
# i, j = 5, 3

def n_layers() :
    crp_n = random.randint(3,5)
    fc_n = random.randint(2,5)
    n_lay = [crp_n, fc_n]
    return n_lay

