Progetto didattico del corso di MISDS, Università degli Studi di Padova, 2022
Francesco Bari, 2020285


L’archivio contiene:

- il file “relazione.pdf” che descrive il progetto
- la cartella “Codice” che contiene tutto il codice relativo al progetto
- una ulteriore cartella “Appendice A” dove ho allegato degli screen relativi a diverse query


(Se non già installate) Ci sono delle librerie da installare, nello specifico: 

- tensorflow (maggiore di 2.0.0)
- tensorflow_hub
- numpy
- scipy
- PIL
- cv2
- matplotlib
- annoy


Per eseguire il tutto, basterà eseguire lo script “main.py” o da riga di comando o con l’editor che si desidera.


Se si volessero fare ulteriori query al sistema di raccomandazione, ho creato una funzione ad hoc. Per utilizzarla basterà:

1. Decommenntare una riga (riga 133, file main.py)
2. Scegliere una query (riga 83, file main.py)

Come query si può utilizzare qualsiasi file .npz nella cartella “feature_vectors”



(In ogni caso, NON consiglio l'esecuzione in locale per via delle lunghe tempistiche (CPU vs GPU). All'interno della relazione ho linkato un notebook Kaggle già compilato su cui poter fare delle query. Per eseguirle basterà modificare le stesse informazioni che ho spiegato poc'anzi)


All’interno del file “main.py” sono anche specificate tutte le variabili d’ambiente necessarie per eseguire il codice. NON c’è bisogno di cambiare niente, neanche i path perché sono relativi.




Il dataset è stato volutamente consegnato in forma ridotta, sia perché si ritene che funzioni comunque bene con 1000 esempi del nostro indice, sia perché se si volesse eseguire localmente su singola CPU, per l’intero dataset ci sarebbe voluto molto tempo (eseguendo in Cloud con molteplici GPU il tempo diventa invece sostenibile)