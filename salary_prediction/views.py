from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib import messages
import pandas as pd
import numpy as np


# Create your views here.
def job(request):
    return render(request , 'job.html')


def predict(request):
    age = int(request.POST['age'])
    gen = request.POST['gender']
    edu = request.POST['education']
    job_title = request.POST['jobTitle']
    exp = int(request.POST['experience'])
    model = pd.read_pickle('/home/ritik/mysite/salary_prediction/DR_model.pickle')
    edu_enc = pd.read_pickle('/home/ritik/mysite/salary_prediction/edu_enc.pickle')
    gen_enc = pd.read_pickle('/home/ritik/mysite/salary_prediction/gen_enc.pickle')
    job_enc = pd.read_pickle('/home/ritik/mysite/salary_prediction/job_enc.pickle')

    test_input = np.array([age, gen, edu, job_title, exp],dtype=object).reshape(1,5)

    ti_sex = gen_enc.transform(test_input[: ,1]).reshape(1,1)

    ti_edu = edu_enc.transform(test_input[: ,2]).reshape(1,1)

    ti_job = job_enc.transform(test_input[: ,3]).reshape(1,1)

    test_input_transformed = np.concatenate((test_input[:,[0]],ti_sex,ti_edu,ti_job,test_input[:,[4]]),axis=1)

    ans = int(model.predict(test_input_transformed))
    messages.info(request,"Predicted Salary : {}".format(ans))
    return render(request,"predicted_salary.html",{'age':age , 'edu' : edu , 'gen' : gen, 'job_title' : job_title , 'exp' : exp , 'ans':ans})

   
