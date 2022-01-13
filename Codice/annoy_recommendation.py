# Per caricare i features vectors delle immagini
import numpy as np

# Per effettuare letture/scritture da file
import glob
import os.path

# Per salvare i risultati su un file json
import json

# Per effettuare il calcolo della similarità
from annoy import AnnoyIndex
from scipy import spatial

# Per mostrare le immagini
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




# Chiama più volte il metodo add_item per aggiungere vettori di features all'indice di Annoy
def get_annoy_index(dims, file_index_to_file_vector, file_index_to_product_id):

  allfiles = glob.glob('./feature_vectors/*.npz')

  t = AnnoyIndex(dims, metric='angular')

  for i, file in enumerate(allfiles):
    file_vector = np.loadtxt(file)
    file_name = os.path.basename(file).split('.')[0]
    file_index_to_file_vector[i] = file_vector
    file_index_to_product_id[i] = file_name 
    t.add_item(i, file_vector)
        
  print("--------------------------------------") 
  print("+ ANNOY index generation - Completed +")
  print("--------------------------------------")  
    
  return t




#-----------------------------------------------------------------------------




# Vengono presi in input tutti i feature vectors risultanti dalla Feature Extraction e 
# vengono inseriti nell’indice di Annoy
def add_items(t, file, file_index_to_file_vector, file_index_to_product_id):
    
    position = len(file_index_to_product_id)
    file_vector = np.loadtxt(file)
    file_name = os.path.basename(file).split('.')[0]
    file_index_to_file_vector[position] = file_vector
    file_index_to_product_id[position] = file_name 
    t.add_item(position, file_vector)




#-----------------------------------------------------------------------------
    
    
  

# Viene costruito (build) l’indice di Annoy vero e proprio
def build_forest(t, trees):
  t.build(trees)




#-----------------------------------------------------------------------------
        
   

  
# Vengono calcolati i migliori nearest neighbours tramite la formula della cosine similarity e
# viene salvato il tutto in un file JSON contenente i risultati
def score_calculation(n_nearest_neighbors, file_index_to_product_id, file_index_to_file_vector, t, nearest_id, json_output_path):
  
  named_nearest_neighbors = []
  final_dict = {}

  # Viene fatto un +2 perché rispetto al numero di nn che vogliamo, 
  # uno è rappresentato dalla query e uno dal prodotto prodotto più simile alla query al 100% (la query stessa)
  n_nearest_neighbors = n_nearest_neighbors + 2

  last_index = list(file_index_to_product_id.keys())[-1]
  nearest_neighbors = t.get_nns_by_item(last_index, n_nearest_neighbors)
  
  for i, j in enumerate(nearest_neighbors):      
      similarity = 1 - spatial.distance.cosine(file_index_to_file_vector[last_index], file_index_to_file_vector[j])
      rounded_similarity = int((similarity * 10000)) / 10000.0
      nearest_id[file_index_to_product_id[j]] = rounded_similarity
      
  # filtro per categoria
  for id, score in nearest_id.items():
      query_category = file_index_to_product_id[last_index].split("_")
      similar_category = id.split("_")
      
      if (similar_category[1] == query_category[1]):
          final_dict[id] = score
          

  final_dict.pop(file_index_to_product_id[last_index])
  named_nearest_neighbors.append({
    'original_product_id': file_index_to_product_id[last_index],
    'similar_products_id': final_dict})


  print("---------------------------------") 
  print("Query Product ID: %s" %file_index_to_product_id[last_index]) 
  print("---------------------------------") 
  print("Nearest Products IDs (with score): %s" %final_dict)
  print("---------------------------------") 


  print("--------------------------------------------") 
  print("+ Similarity score calculation - Completed +")
  print("--------------------------------------------") 
  
  
  with open(json_output_path, 'w') as out:
      json.dump(named_nearest_neighbors, out)


  print("------------------------------------------------") 
  print("+ Data stored in 'nearest_neighbors.json' file +")
  print("------------------------------------------------")





#-----------------------------------------------------------------------------




# Funzione per stampare qualche risultato
def print_query_and_suggestions(json_path):
    with open(json_path) as json_file: 
        print_dict = json.load(json_file)
    
    query_id = print_dict[0]['original_product_id']
    path = './cropped_and_labeled_images/{}.jpg'.format(query_id)
    img = cv2.imread(path)
    img = cv2.resize(img, (1800, 2400))
    blue, green, red = cv2.split(img)
    frame_rgb = cv2.merge((red, green, blue))
    imgplot = plt.imshow(frame_rgb)
    plt.title("\nQuery ID: " + query_id + "\n",  fontsize=14)
    plt.savefig('query.png')
    
    
    
    i = 0
    j = 1
    rows = 1
    columns = 5
    figure=plt.figure(figsize=(30, 20), tight_layout=True)
    
    for id in print_dict[0]['similar_products_id']:
        list_score = list(print_dict[0]['similar_products_id'].values())
        path = './cropped_and_labeled_images/{}.jpg'.format(id)
        image = cv2.imread(path)
        image = cv2.resize(image, (1800, 2400))
        blue, green, red = cv2.split(image)
        frame_rgb = cv2.merge((red, green, blue))
        figure.add_subplot(rows, columns, j)
        plt.title("\nID: " + id + "\n",  fontsize=22)
        plt.xlabel("\nScore: " + str(list_score[i]),  fontsize=24)
        plt.imshow(frame_rgb)

        i = i + 1
        j = j + 1
        if(j > (rows*columns)):
            break
    plt.savefig('similar_products.png')
    

        


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
        
        
        

def annoy_script(dims, n_nearest_neighbors, trees, json_output_path, query_path):


  # Mappatura indice ANNOY - file_vector
  file_index_to_file_vector = {}
  
  # Mappatura indice ANNOY - id product
  file_index_to_product_id = {}

  # Dizionario contenente coppie id:somiglianza dei nearest neighbors
  nearest_id = {}


  # 1) Elenchiamo gli elementi dell'indice di annoy
  t = get_annoy_index(dims, file_index_to_file_vector, file_index_to_product_id)

  # 2) Aggiungiamo all'indice l'elemento di query in ultima posizione
  add_items(t, query_path, file_index_to_file_vector, file_index_to_product_id)

  # 3) Costruiamo l'indice di annoy
  build_forest(t, trees)

  # 4) Calcoliamo i K-NN
  score_calculation(n_nearest_neighbors, file_index_to_product_id, file_index_to_file_vector, t, nearest_id, json_output_path)

  # 5) Stampiamo i risultati
  print_query_and_suggestions(json_output_path)




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------