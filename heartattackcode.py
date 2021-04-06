#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Please Enter Required Infomation for Heart Attack Predition: ")

Age = int(input("Age:"))
Chol=float(input("Chol:"))
Thalach= float(input("Thalach:"))
Cp= float(input("Cp:"))
Treshbps= float(input("Treshbps:"))
Slope=float(input("Slope:"))
fbs=float(input("fbs:"))
Thal= float(input("Thal:"))
Ca= float(input("Ca:"))
Restecg=float(input("Restecg:"))  


 # Converting outputs of the patients
Ages=(Age)
Chols=(Chol)
Thalachs=(Thalach)
Cps=(Cp)
Treshbpss=(Treshbps)
Slopes=(Slope)
fbss=(fbs)
Thals=(Thal)
Cas=(Ca)
Restecgs=(Restecg)  


            
#heartattack predition information 
mediumlow= 47
highrisk= 54
lowmedium= 61
lowlow= 46
slightlylow= 77

cpnum=0.43
thalachnum=0.42
slopenum=0.35
restecgnum= 0.14
fbsnum=-0.0028
cholnum=-0.085
treshbpsnum=-0.14
thalsnum=-0.34
canum=-0.39


#If Statements based on age 

print("Results of Heart Attack Prediction based age. ")

if Ages <= mediumlow and lowlow > mediumlow:
    print("Your Heart Attack Prediction rate is medium low risk based on age being", Ages)      
if mediumlow > lowlow:
    print("Your Heart Attack Prediction rate is very low risk based on age being", Ages)            
if Ages > mediumlow and highrisk>= Ages:
    print("Your Heart Attack Prediction is very high risk based on age being", Ages)            
if Ages >= slightlylow:
    print("Your Heart Attack Prediction is slightly low risk due to age being", Ages)    
if Ages <= lowmedium and highrisk < Ages:
    print("Your Heart Attack Prediction is medium risk due to age",Ages)

# If statements based on top variables and age

print ("Results of cholerstol risk level:")


if Chols >= cholnum: 
    print( "highrisk") 
else:
    print("lowrisk")
    
print ("Results of Thalach risk level:")

if Thalachs >= thalachnum:
    print("highrisk") 
else:
    print("lowrisk")
    
print("Results of CP risk level:")
   
if Cps >= cpnum:
    print("highrisk") 
else:
    print("lowrisk")
    
print("Results of Treshbpss risk level:")

if Treshbpss >= treshbpsnum:
    print("highrisk") 
else:
    print("lowrisk")
    
print("Results of Slope based on target:")

if Slopes>= slopenum:
    print("highrisk") 
else:
    print("lowrisk")
    
print("Results of Fbs risk level:")

if fbss >= fbsnum:
    print("highrisk")
else:
    print("lowrisk")
    
print("Results of Thal risk level:")

if Thals>= thalsnum:
    print("highrisk") 
else:
    print("lowrisk")
    
print("Results of Ca risk level:")

if Cas >= canum:
    print("highrisk") 
else:
    print("lowrisk")
    
print("Results of Restecgs risk level:")

if Restecgs >=restecgnum:
    print("highrisk")
else:
    print("lowrisk")
    
# Information to determine if patient is high, low, or medium risk to having a heart attack based off the top five indicators
    
print ("1. If the top five indicators of a heartattack are high and age is highrisk you are possibly at a very high chance of having a heart attack.") 
print ("2. If the top five indicators are high risk and age is low risk you may still be at a high risk of having a heart attack.")
print("3. If the top five indicators are low risk and your age is high or low risk then you are probably not at risk of having a heartattack.")
    


# In[ ]:




