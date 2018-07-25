import os
i=800
abs_path="/home/eman/Documents/female-classifier/classifier/train/women-some"

for filename in os.listdir(abs_path):
        os.rename(os.path.join(abs_path,filename), os.path.join(abs_path,"woman"+str(i)) )
        i+=1
