# Per compiere inferenza sui vari TF-Hub module di base
import tensorflow as tf
import tensorflow_hub as hub

# Per il download delle immagini dalle varie fonti possibili (cvs, json, http)
import csv
import json
import requests
import os

# Per ritagliare e mostrare le immagini
from PIL import Image
import cv2
import matplotlib.pyplot as plt




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# FUNZIONI DI UTILITA'




# Metodo (opzionale) se si vuole partire da un file CSV al posto di un JSON
def csv_to_json(csv_path, json_path):
    
    json_array = [] # messo qui dentro perché viene inizializzato solo se necessario (essendo un metodo opzionale)
      
    with open(csv_path, encoding='utf-8') as csvf: 
        csv_reader = csv.DictReader(csvf) 

        for row in csv_reader: 
            json_array.append(row)
  
    with open(json_path, 'w', encoding='utf-8') as jsonf: 
        json_string = json.dumps(json_array, indent=4)
        jsonf.write(json_string)
          



#-----------------------------------------------------------------------------




# Metodo per reperire tutte le immagini e salvarle in una directory locale
def save_initial_images(json_path):
    with open(json_path) as json_file: 
        original_images_dict = json.load(json_file)
        
    for i in range(len(original_images_dict)):
        id = original_images_dict[i]['id']
        url = original_images_dict[i]['path']
        page = requests.get(url)
        # ho specificato qui il path perché stiamo eseguendo il codice localmente... 
        file_name = './original_images/{}'.format(id)
        
        original_images_dict[i]['path'] = file_name

        with open(file_name, 'wb') as f:
            f.write(page.content)
            
            
    print("--------------------------") 
    print("+ Original images loaded +")
    print("--------------------------") 
            
    return original_images_dict
 



#-----------------------------------------------------------------------------
    



# Chiamata a Tensorflow Hub per utilizzare l'architettura FasterRCNN
def load_model(module_handle):
    detector = hub.load(module_handle).signatures['default']
    print("---------------------") 
    print("+ FasterRCNN loaded +")
    print("---------------------") 
    return detector




#-----------------------------------------------------------------------------




# Questo medoto:
# (a) trasforma l'immagine in un tensore dopo averla normalizzata
# (b) Chiama il detector (FasterRCNN) sull'immagine
# (c) Salva i risultati dentro result
# (d) Chiama il metodo crop_objects.
def run_detector(detector, original_images_dict, cropped_images_dict, class_list):
    for i in range(len(original_images_dict)):
        path = original_images_dict[i]['path']
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)

        converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        result = detector(converted_img)

        result = {key:value.numpy() for key,value in result.items()}
        crop_objects(img.numpy(), result, path, cropped_images_dict, class_list)
        
    print("--------------------------------") 
    print("+ Object Detection - Completed +")
    print("--------------------------------") 
  
    
    
    
#-----------------------------------------------------------------------------    
  
 

  
# Questo metodo prende i risultati della detection (i detection_boxes) e, 
# se hanno una percentuale di confidenza di rilevazione superiore ad un certo valore (per noi del 60%) 
# ed appartengono ad una classe del contesto fashion, 
# vengono ritagliati e salvati con l’informazione legata alla loro classe di appartenenza
def crop_objects(img, result, path, cropped_images_dict, class_list, max_boxes=3, min_score=0.6):
    image = Image.fromarray(img)
    width, height = image.size
    for i in range(min(result['detection_boxes'].shape[0], max_boxes)):
        if (result['detection_scores'][i] >= min_score):
            detected_class = "{}".format(result["detection_class_entities"][i].decode("ascii"))
            if(detected_class in class_list):
                ymin = int(result['detection_boxes'][i,0]*height)
                xmin = int(result['detection_boxes'][i,1]*width)
                ymax = int(result['detection_boxes'][i,2]*height)
                xmax = int(result['detection_boxes'][i,3]*width)
                crop_img = img[ymin:ymax, xmin:xmax].copy()
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                detected_class = detected_class.replace(" ", "-")
                outfile_name = os.path.basename(path).split('.')[0] + "-" + str(i) + "_" + detected_class
                # anche qui ho specificato il path perché stiamo eseguendo il codice localmente...
                cv2.imwrite("./cropped_and_labeled_images/{}.jpg".format(outfile_name), crop_img)
                cropped_images_dict[outfile_name] = detected_class




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




# Script che verrà richiamato da main.py
def detector_script(csv_path, json_path, module_handle_detector, class_list):

    # Stampiamo la versione di TF (deve essere maggiore di 2.0.0)
    print("Tensorflow "+ tf.__version__)
    

    # Dizionario contenente id:path di ogni immagine iniziale
    original_images_dict = {}
    # Dizionario contenente id:path di ogni immagine ritagliata e classificata
    cropped_images_dict = {}


    #1) Si parte da un CSV e lo si converte in JSON (opzionale)
    csv_to_json(csv_path, json_path)

    #2) A partire dal JSON facciamo il download delle immagini
    original_images_dict = save_initial_images(json_path)

    #3) Carichiamo il modello
    detector = load_model(module_handle_detector)

    #4) Facciamo crop+label con il modello caricato, sulle immagini salvate
    run_detector(detector, original_images_dict, cropped_images_dict, class_list)




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------