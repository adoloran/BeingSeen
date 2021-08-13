import pandas as pd
from google.colab import files
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow # pour remplacer le cv2.imshow qui ne cfonctionne pas sur collab, car collab ne prend pas en compte la creation de  window il le fait automatiquement



#df = pd.read_csv('csvtest.csv') #le delimiter permet donc de choisir quel motif sépare les colonnes (generalise alors le coma separated file à ce que tu veux separted file) pratique dans le cas ou les valeurs ont déja des virugles
#df.drop(df.tail(1).index,inplace=True) #permet de supprimer les n dernières lignes ici (que la derniere donc n=1 (head pour les premieres lignes))
#df.drop(columns=['B', 'C']) permet de supprimer la conne B et C par exemple
#print(df.shape) # permet de renvoyer un vecteur contenant les dimension du tableau du tableau 
#df.drop(df.iloc[:,13:293],1,inplace=True)
#Call pd.DataFrame.drop(labels=None, axis=1, inplace=True) with labels set to pd.DataFrame.columns[x] where x is the column index to be deleted.
#df.iloc[:,0] = (df.iloc[:,0] - df.iloc[0,0])/1000 #enlève l'offset lié au unixtimestamp et mets en secondes 
#df.iloc[:,7]= df.iloc[:,7] + abs(df.iloc[:,7]) + df.iloc[:,12]+ np.arctan(df.iloc[:,10],(df.iloc[:,4]+df.iloc[:,7])/2)
#s=pd.Series(df.iloc[:,2]) #permet de créer un vecteur à manipuler via divers méthodes 
#pd.to_numeric(s,downcast='float') #tranforme le type object en float 
#df['gaze_an'].mean() #calcul la moyenne sur une colonne donnée 
#df.dtypes affiches le type 
#df.sample(n) affiche n sample 
#df
 
#df.to_csv('csvtest2.csv',index=False, header=True) #retelcharge le fichier csv avec les bonnne separation et le bon temps 
#files.download('csvtest2.csv')

#for i in range(1, len(data)):   #permet donc de faire des calcule entre les elements d'une même colonne par exemple
		#pos_before = [data['X'].iloc[i-1], data['Y'].iloc[i-1]]
		#pos_actual = [data['X'].iloc[i], data['Y'].iloc[i]]
		
		#spd_x = (pos_actual[0] - pos_before[0]) / (data['Time'].iloc[i] - data['Time'].iloc[i-1])
		#spd_y = (pos_actual[1] - pos_before[1]) / (data['Time'].iloc[i] - data['Time'].iloc[i-1])
		#spd = math.sqrt((spd_x * spd_x) + (spd_y * spd_y))
		#speed.append(spd)



#sLength = len(df['a']) prend la longueur de la colonne 'a'

#df = df.assign(e=pd.Series(np.random.randn(sLength)).values) ajoute les valeur de la série s en colonne de plus 
#ou 
#data['new_col'] = list_of_values
#data.loc[ : , 'new_col'] = list_of_valuesdat

#df = df[['mean', '0', '1', '2', '3']] permet d'assigner manuellement le nouvel ordre des colonne de la df 

#mean_df = df['grade'].mean() calcul la moyenne d'une colonne definit ici grade par exemple


#image = cv2.imread('opentest.JPG')
#print(image.shape)
#cv2.line(image,(200,200), (400,200), (255,0,0), 2) #il existe aussi cv.rectangle / la fonction ne prend que des int en arg donc faire int(float) pour transformer les float en int
#cv2.circle(image, (0,0), radius=0, color=(0, 0, 255), thickness=-1) #pour un point cercle de rayon 0 /thickness of -1 px will fill the rectangle shape by the specified color.
#cv2_imshow(image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
