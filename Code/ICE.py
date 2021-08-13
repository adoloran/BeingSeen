import pandas as pd
import os
import numpy as np
import math
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#from validation import f1
import ast
from collections import Counter
from sklearn.cluster import OPTICS
from google.colab import files
import cv2
from google.colab.patches import cv2_imshow

def best_eps(cluster_data, num_clusters):
    #min_samples = int(np.round(np.log(cluster_data.shape[0])))

    min_samples = cluster_data.shape[0] / 100 # the size of a cluster is determined by this
                        # (100/10000) = 1% at least to be considered a cluster
    #min_samples = 9
    eps_range = np.linspace(0.001, 1, num=50)
    prev = None

    for i in reversed(range(0, eps_range.shape[0])):
        current_eps = eps_range[i]
        db = DBSCAN(eps=current_eps, min_samples=min_samples).fit(cluster_data)
        labels = db.labels_
        cnt = (Counter(labels))
        ratio = cnt.most_common(1)[-1][1] / cnt.most_common(2)[-1][1]
        if(len(set(labels)) >= num_clusters and ratio < 10):
            if(prev == None):
                return current_eps, min_samples
            else:
                return current_eps , min_samples
        prev = current_eps
    default_val = 0.001
    return default_val, min_samples

def plot_eps(cluster_data, num_clusters):
    #min_samples = int(np.round(np.log(cluster_data.shape[0])))

    min_samples = cluster_data.shape[0] / 100 # the size of a cluster is determined by this
                        # (100/10000) = 1% at least to be considered a cluster
    #min_samples = 9
    eps_range = np.linspace(0.001, 1, num=50)
    prev = None

    ratio_list = []
    for i in reversed(range(0, eps_range.shape[0])):
        current_eps = eps_range[i]
        db = DBSCAN(eps=current_eps, min_samples=min_samples).fit(cluster_data)
        
        labels = db.labels_
        cnt = (Counter(labels))
        if(len(list(set(labels))) == 1 or cnt.most_common(1)[-1] == -1):
            ratio_list.append(len(labels))
            continue
        
        ratio = cnt.most_common(1)[-1][1] / cnt.most_common(2)[-1][1]
        ratio_list.append(ratio)
        prev = current_eps
    default_val = 0.001
    return ratio_list
    
#output_f = []
#cnt = 0
# for file in os.listdir('validation/openface_normalized/'):
    #for x in range(1,2):
        #if(file.startswith('.') or file == '2018-09-08_11-43-44-665-W-B-user602.csv'):
            #continue
        #if('I-T' in file or 'I-B' in file):
         #   continue
        #print('Processing ',file)
        #path = os.path.join('validation/openface_normalized/', file)
        
        #read data
        #df = pd.read_csv(path)
        
        #hso = 900
        #current_data = df.values[:,[df.values.shape[1]-2, df.values.shape[1]-1]][0:900*x]
        #current_data = df.values[:,[df.values.shape[1]-2, df.values.shape[1]-1]]
        #num_clusters = 2 #2 people speaking
        #ratio_list = plot_eps(current_data, num_clusters)
        #output_f.append([file, ratio_list])
        #print('Done with ', file)
        #if(cnt > 25):
            #break
        #cnt += 1
        #continue
df = pd.read_csv('csvtestf3.csv')
df.drop(df.iloc[:,13:293],1,inplace=True) # permet de trier les colonne eylandmarks
print(df.shape)
all_data = df.values
all_data_out = []
current_data = all_data[:,[df.values.shape[1]-2, df.values.shape[1]-1]] #juste quand on export les gaze features ( -gaze ) avec cette ligne on ne prend que le gaze angle_x et le gaze angle_y (la derniere et avand derniere colonne du tableau gaze representant donc ces deux features)
        #sucess_frame = all_data[:, 3]                                 # shape est un vecteur contenant la dimension des ligne shape(0) et celle des colonnes shap(1)
        #training_data = []
    
        #for i in range(sucess_frame.shape[0]):
            #try:
                #if(sucess_frame[i] >= 0.9):
                    #training_data.append(current_data[i])
                    #all_data_out.append(all_data[i])
            #except:
                #continue
        #current_data = np.array(training_data).reshape(-1,2)
    
num_clusters = 2 #2 people speaking        
        
eps_val, min_samples = best_eps(current_data, num_clusters)
        #if(eps_val == 0.001):
            #continue
        
        
db = DBSCAN(eps=eps_val, min_samples=min_samples).fit(current_data)
        #db = OPTICS(min_samples=int(current_data.shape[0]/50)).fit(current_data)
labels = db.labels_
labels = list(labels)
        #print(Counter(labels))
        #continue
        
        
most_common,num_most_common = Counter(labels).most_common(1)[0]    
cluster_count = {x:labels.count(x) for x in labels}
contact_cluster = max(cluster_count, key=cluster_count.get) #recupère l'indice (key) du label avec le plus de point dedans,
        
        #============================================================#
        # draw the bounding box
x_list = []  #store x axis of contact cluster
y_list = []  #store y axis of contact cluster
    
for i in range(0, len(labels)):  # prend tous les points du plus gros cluster (label[i]= contact_cluster, et les stock dans une liste, les x dans une x_list et les y dans la y_list
  if(labels[i] == contact_cluster):
    x_list.append(current_data[i][0])
    y_list.append(current_data[i][1])
    
upper_bound_x = max(x_list)  #right-most angle  #récupere les bords du plus gros cluster avec de tracer le carré centrale qui delimitera donc les 8 autres 
lower_bound_x = min(x_list)  #left-most angle
upper_bound_y = max(y_list)  #lowest angle
lower_bound_y = min(y_list)  #highest angle    
        
        #============================================================#
contact_list = []
contact_list_1 = []
contact_list_2 = [] 
contact_list_3 = []
contact_list_4 = []
contact_list_5 = []
contact_list_6 = []
contact_list_7 = []
contact_list_8 = []
contact_list_9 = []
    
all_data = df.values[:,[df.values.shape[1]-2, df.values.shape[1]-1]]
print(all_data) # pour verifier que c'est bien les gaze angle qui sont traités.
for frame in all_data:       # Stock chaque point dans une des 9 case en fonction des boundaries du carré centrale 
  if (frame[0] < lower_bound_x  and frame[1] < lower_bound_y):
    contact = 1
    contact_list_1.append(1)
    contact_list_2.append(0)
    contact_list_3.append(0)
    contact_list_4.append(0)
    contact_list_5.append(0) # cette boucle parcours toutes les frame frame[:,colonne y et x ] frame[i][0] = gaze_angle_x, frame[i][1]=gaze_angle_x 
    contact_list_6.append(0) # mais comme nous somme deja dans la boucle qui parcours tout les i de frame frame[i][0]<=>frame[0]
    contact_list_7.append(0) # donc pour chaque frame il verifie le x et le y et ajoute a la liste correspondante avec le numéro correspondant de la région un "1"
    contact_list_8.append(0) # on ajoute "1" a la contacte liste n si le x et le y de cette frame corresponde a cette region 
    contact_list_9.append(0) # on alors un tableau qui sur chaque frame à 1 sur une région pour dire qu' a cette frame précise le regard est orienté dans cette région
    contact_list.append(1)
  elif (frame[1] < lower_bound_y and frame[0] > lower_bound_x and frame[0] < upper_bound_x):
    contact = 2
    contact_list_1.append(0)
    contact_list_2.append(1)
    contact_list_3.append(0)
    contact_list_4.append(0)
    contact_list_5.append(0)
    contact_list_6.append(0)
    contact_list_7.append(0)
    contact_list_8.append(0)
    contact_list_9.append(0)
    contact_list.append(2)
  elif (frame[1] < lower_bound_y and frame[0] > upper_bound_x):
    contact = 3
    contact_list_1.append(0)
    contact_list_2.append(0)
    contact_list_3.append(1)
    contact_list_4.append(0)
    contact_list_5.append(0)
    contact_list_6.append(0)
    contact_list_7.append(0)
    contact_list_8.append(0)
    contact_list_9.append(0) 
    contact_list.append(3)
  elif (frame[1] > lower_bound_y and frame[1] < upper_bound_y and frame[0] < lower_bound_x):
    contact = 4
    contact_list_1.append(0)
    contact_list_2.append(0)
    contact_list_3.append(0)
    contact_list_4.append(1)
    contact_list_5.append(0)
    contact_list_6.append(0)
    contact_list_7.append(0)
    contact_list_8.append(0)
    contact_list_9.append(0) 
    contact_list.append(4)
  elif (frame[1] > lower_bound_y and frame[1] < upper_bound_y and frame[0] > lower_bound_x and frame[0] < upper_bound_x):
    contact = 5
    contact_list_1.append(0)
    contact_list_2.append(0)
    contact_list_3.append(0)
    contact_list_4.append(0)
    contact_list_5.append(1)
    contact_list_6.append(0)
    contact_list_7.append(0)
    contact_list_8.append(0)
    contact_list_9.append(0) 
    contact_list.append(5)
  elif (frame[1] > lower_bound_y and frame[1] < upper_bound_y and frame[0] > upper_bound_x):
    contact = 6
    contact_list_1.append(0)
    contact_list_2.append(0)
    contact_list_3.append(0)
    contact_list_4.append(0)
    contact_list_5.append(0)
    contact_list_6.append(1)
    contact_list_7.append(0)
    contact_list_8.append(0)
    contact_list_9.append(0)   
    contact_list.append(6)
  elif (frame[1] > upper_bound_y and frame[0] < lower_bound_x):
    contact = 7
    contact_list_1.append(0)
    contact_list_2.append(0)
    contact_list_3.append(0)
    contact_list_4.append(0)
    contact_list_5.append(0)
    contact_list_6.append(0)
    contact_list_7.append(1)
    contact_list_8.append(0)
    contact_list_9.append(0)  
    contact_list.append(7)
  elif (frame[1] > upper_bound_y and frame[0] < upper_bound_x and frame[0] > lower_bound_x):
    contact = 8  
    contact_list_1.append(0)
    contact_list_2.append(0)
    contact_list_3.append(0)
    contact_list_4.append(0)
    contact_list_5.append(0)
    contact_list_6.append(0)
    contact_list_7.append(0)
    contact_list_8.append(1)
    contact_list_9.append(0)
    contact_list.append(8)
  elif (frame[1] > upper_bound_y and frame[0] > upper_bound_x):
    contact = 9
    contact_list_1.append(0)
    contact_list_2.append(0)
    contact_list_3.append(0)
    contact_list_4.append(0)
    contact_list_5.append(0)
    contact_list_6.append(0)
    contact_list_7.append(0)
    contact_list_8.append(0)
    contact_list_9.append(1)
    contact_list.append(9)
  else: # for 4 points at the boundaries, they actually belong to the contact class
    contact = 5
    contact_list_1.append(0)
    contact_list_2.append(0)
    contact_list_3.append(0)
    contact_list_4.append(0)
    contact_list_5.append(1)
    contact_list_6.append(0)
    contact_list_7.append(0)
    contact_list_8.append(0)
    contact_list_9.append(0)
    contact_list.append(5)
    
all_data = df.values # crée un nouveau tableau incluant donc les point et la région a la quelle ils apartiennent
new_df = pd.DataFrame({'Region 1':contact_list_1, 'Region 2':contact_list_2, 'Region 3':contact_list_3, 'Region 4':contact_list_4, 'Region 5':contact_list_5, 'Region 6':contact_list_6, 'Region 7':contact_list_7, 'Region 8':contact_list_8, 'Region 9':contact_list_9, 'Region':contact_list})
new_data = np.concatenate((all_data,new_df.values), axis = 1)
        #new_data = contact_list
output = pd.DataFrame(new_data)
header_list = list(df.columns) + list(new_df.columns)
    
        #path= 'validation/openface_0921/'
        #name = file.split('.')[0]+'_new.csv'
        #name = file
        #if('11-27-01' in file):
            #name = 'user0.csv'
        #elif('11-43-44' in file):
            #name = 'user1.csv'
        #elif('12-01-43' in file):
            #name = 'user2.csv'
        #elif('12-17-43' in file):
            #name = 'user3.csv'
        #elif('12-34-07' in file):
            #name = 'user4.csv'
        #elif('12-54-50' in file):
            #name = 'user5.csv'
        #elif('13-09-49' in file):
            #name = 'user6.csv'
        #elif('13-37-59' in file):
           # name = 'user8.csv'
        #elif('13-53-53' in file):
            #name = 'user9.csv'
        #name = name.split('.')[0]+"_first"+str(x)+'.csv'

output.to_csv('Finalnew3.csv', header=header_list, index=False)
files.download('Finalnew3.csv')
print("Extraction Done")


#Test de conversion afin d'afficher les poin directement sur l'ecran 


#conversion de -pi/2=>pi/2 des gaze angle en point sur l'ecran pour verifier visuellement le cadre RVE 

#print(lower_bound_x,lower_bound_y,upper_bound_x,upper_bound_y)
#image = cv2.imread('opentest.JPG')
#x1= lower_bound_x + (np.pi)/2
#x2= upper_bound_x + (np.pi)/2
#y1= lower_bound_y + (np.pi)/2
#y2 = upper_bound_y + (np.pi)/2


#x1=(x1/np.pi)*image.shape[0]
#x2=(x2/np.pi)*image.shape[0]
#y1=(y1/np.pi)*image.shape[1]
#y2=(y2/np.pi)*image.shape[1]

#cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)), (255,0,0), 2) #il existe aussi cv.rectangle 
#cv2.circle(image, (x,y), radius=0, color=(0, 0, 255), thickness=-1) pour un point cercle de rayon 0
#cv2_imshow(image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#demain : tenter de supprimer les eyelendmarks sur un autre code python et relancer le code sur un tableau sans les 280 landmarks, 
#tester aussi avec une video plus longu avec des regards plus inssitant sur les cotés 
#df_out = pd.DataFrame(output_f)
#df_out.to_csv('ratio_m.csv', index=False, header=None)