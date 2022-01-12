# Per compiere inferenza sui vari TF-Hub module di base
import tensorflow as tf
import tensorflow_hub as hub

# Per operazioni di trasformazione di immagini/pixels
import numpy as np

# Per effettuare letture/scritture da file
import os.path
import glob




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




# Reperisce le immagini derivanti dalla operazione di Object Detection
def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

  return img




#-----------------------------------------------------------------------------
  



# (a)Esso in prima istanza reperisce l'immagine derivante dalla operazione di Object Detection (metodo load_img)
# (b)Successivamente chiama l’extractor per calcolare ed estrarre effettivamente gli image feature vectors delle immagini
# (c)Effettua poi una operazion edi squeeze,che elimina tutte le single-dimensional entries, fornendo quindi un array unidimensionale
# (d)Salva i file .npz contenenti gli image feature vectors
def get_feature_vectors(module_handle):

  module = hub.load(module_handle)
  print("-----------------------") 
  print("+ MobileNet_v2 loaded +")
  print("-----------------------") 

  # ho specificato qui il path perché stiamo eseguendo il codice localmente... 
  for filename in glob.glob('./cropped_and_labeled_images/*.jpg'):
    img = load_img(filename)
    features = module(img)   
    feature_set = np.squeeze(features)  
    outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
    # ho specificato qui il path perché stiamo eseguendo il codice localmente... 
    out_path = os.path.join('./feature_vectors', outfile_name)
    np.savetxt(out_path, feature_set, delimiter=',')    


  print("------------------------------------------") 
  print("+ Generating Feature Vectors - Completed +")
  print("------------------------------------------")   
    



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




def extractor_script(module_handle_extractor):

  #1) Otteniamo i features vectors a partire dalla immagini croppate ed etichettate
  get_feature_vectors(module_handle_extractor)





#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------