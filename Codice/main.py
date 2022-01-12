# Per creare cartelle localmente
import os

# 3 script principali
import detector
import extractor
import annoy_recommendation




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# VARIABILI D'AMBIENTE - DETECTOR




# path iniziale CSV (opzionale)
csv_path = './original_images_subset.csv'

# path iniziale JSON (in caso il CSV verrà trasformato in JSON e messo qui)
json_path = './json/images_subset.json'

# Modello utilizzato per object detection
module_handle_detector = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" 

# Lista delle classi di nostro interesse
class_list = ["Clothing", "Cowboy hat", "Sombrero", "Sun hat", "Scarf",
              "Skirt", "Miniskirt", "Jacket", "Fashion accessory", "Glove", 
              "Baseball glove", "Belt", "Sunglasses", "Tiara", "Necklace",
              "Sock", "Earrings", "Tie", "Goggles", "Hat", "Fedora", "Handbag",
              "Watch", "Umbrella", "Glasses", "Crown", "Swim cap", "Trousers",
              "Jeans", "Dress", "Swimwear", "Brassiere", "Shirt", "Coat", "Suit"
              "Footwear", "Roller skates", "Boot", "High heels", "Sandal",
              "Sports uniform", "Luggage & bags", "Backpack", "Suitcase",
              "Briefcase", "Helmet", "Bicycle helmet", "Football helmet"]




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# VARIABILI D'AMBIENTE - EXTRACTOR




# Modello utilizzato per questa fase
module_handle_extractor = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# VARIABILI D'AMBIENTE - ANNOY




dims = 1792

n_nearest_neighbors = 5

trees = 10000

json_output_path = './json/nearest_neighbors.json'


query_path = "./feature_vectors/15970-0_Shirt.npz"




#-----------------------------------------------------------------------------
#-----------------------------OPZIONALE---------------------------------------

#Si specifica qui la nuova query (qualsiasi file .npz nella cartella feature_vectors)
new_query_path = "./feature_vectors/59263-0_Watch.npz" #questo è un ulteriore esempio

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



def main():

    # 0) (OPZIONALE) Creiamo la cartella dove andremo a salvare localmente le immagini 
    # è opzionale perché si può far girare il codice in Cloud e salvare tutto su un Drive
    os.makedirs("./original_images/", exist_ok=True)
    os.makedirs("./cropped_and_labeled_images/", exist_ok=True)
    os.makedirs("./feature_vectors/", exist_ok=True)
    os.makedirs("./json/", exist_ok=True)

    # Esecuzione script detection e classification
    detector.detector_script(csv_path, json_path, module_handle_detector, class_list)

    # Esecuzione script feature extraction
    extractor.extractor_script(module_handle_extractor)

    # Esecuzione script reccomendation system 
    annoy_recommendation.annoy_script(dims, n_nearest_neighbors, trees, json_output_path, query_path)








#-----------------------------------------------------------------------------



# Ho creato questo metodo se in caso si volessero fare nuove prove con altre query
def another_inference(new_query_path):
    annoy_recommendation.annoy_script(dims, n_nearest_neighbors, trees, json_output_path, new_query_path)




#-----------------------------------------------------------------------------




if __name__ == "__main__":
    main()
    #another_inference(new_query_path)





#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------