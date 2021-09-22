# Single file attachement code 

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
mail_content = '''Hello,
This is a mail from Fraction Analytics.
In this mail we are sending some attachments for today Events.
The mail is sent using Python SMTP library.
Thank You
'''
#The mail addresses and password
sender_address = 'sumathi0251@gmail.com'
sender_pass = '*********'
receiver_address = 'dhanya0251@gmail.com'
#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'This mail from Fraction Analytics. It has an attachment.'
#The subject line
#The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'plain'))
attach_file_name = '/home/sumathi/cap/pytorch-tutorial/tutorials/advanced/image_captioning/caption.txt'
attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
payload = MIMEBase('application', 'octate-stream')
payload.set_payload((attach_file).read())
encoders.encode_base64(payload) #encode the attachment
#add payload header with filename
payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
message.attach(payload)
#Create SMTP session for sending the mail
session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
session.starttls() #enable security
session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()
print('Mail Sent')


##################################################### ****************** ########################################################################


import os 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
# from email.mime.mimeimage import MIMEImage
from email import encoders

mail_content = ''' Hello,
This is a mail from Fraction Analytics.
In this mail we are sending some attachments for today Events.
The mail is sent using Python SMTP library.
Thank You'''


files = "/home/sumathi/cap/Files" #Folder of files 
filenames = [os.path.join(files, f) for f in os.listdir(files)]
# print(filenames)


#Set up users for email
gmail_user = "sumathi0251@gmail.com"
gmail_pwd = "**********"
recipients = 'dhanya0251@gmail.com'

#Create Module
def mail(to, subject, text, attach):
   msg = MIMEMultipart()
   msg['From'] = gmail_user
   msg['To'] = recipients
   msg['Subject'] = subject

   msg.attach(MIMEText(text))
   msg.attach(MIMEText(mail_content, 'plain'))

   #get all the attachments
   for file in filenames:
      part = MIMEBase('application', 'octet-stream')
      part.set_payload(open(file, 'rb').read())
      encoders.encode_base64(part)
      part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file))
      msg.attach(part)

   mailServer = smtplib.SMTP("smtp.gmail.com", 587) #Port 587: This is the default mail submission port. 
   mailServer.ehlo()
   mailServer.starttls() # (tls) >> TransportLayerSecurity >>secure transmission of data between one server to another server
   mailServer.ehlo()
   mailServer.login(gmail_user, gmail_pwd)
   mailServer.sendmail(gmail_user, to, msg.as_string())
   mailServer.close()

#send it
mail(recipients,
   "This message is from Fraction Analytics",
   "Today's Report" ,
   filenames)
print('Mail Sent')