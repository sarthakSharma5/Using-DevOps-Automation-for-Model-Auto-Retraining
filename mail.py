#!/usr/bin/env python
# coding: utf-8

# In[3]:


accStd = open('accuracy.txt','r')
accuracy = float(accStd.read())
accStd.close()


# In[20]:


import smtplib

# SET EMAIL LOGIN REQUIREMENTS
gmail_user = 'sarthaksharma10022000@gmail.com'
gmail_app_password = 'vmto vspf ccjy hnsa'

# SET THE INFO ABOUT THE SAID EMAIL
sent_from = gmail_user
sent_to = ['sarthaksharma10022000@gmail.com', 'sarthaksharma575@gmail.com']
sent_subject = "Model Ready"
sent_body = """ REPORT \n\n Model Accuracy: %s""" % (accuracy)

email_text = """From: %s
To: %s
Subject: %s

%s
""" % (sent_from, ", ".join(sent_to), sent_subject, sent_body)

# Details: http://www.samlogic.net/articles/smtp-commands-reference.htm
try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(gmail_user, gmail_app_password)
    server.sendmail(sent_from, sent_to, email_text)
    server.close()

    print('Email sent!')
except Exception as exception:
    print("Error: %s!\n\n" % exception)


# In[ ]:




