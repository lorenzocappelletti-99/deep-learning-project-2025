# README — Classificatore di spettri (progetto.ipynb)

**Lingua:** Italiano

## Descrizione

Questo repository contiene il lavoro svolto per costruire, addestrare e valutare una rete neurale convoluzionale per la classificazione di spettri (immagini generate a partire da profili spettrali). Il notebook principale è `progetto.ipynb` e include tutto: generazione di spettri sintetici, pipeline di training, trasfer learning, ottimizzazione degli iperparametri con *hyperopt* e script per valutazione.

Il progetto è organizzato in più "versioni" (es. VERSIONE 1: una sola classe spettrale; VERSIONE 2: più classi nella stessa immagine) ed è pensato per essere riproducibile con dati organizzati in cartelle (data-as-folders).

---

## Contenuto chiave del notebook

* Generazione di spettri sintetici (funzioni per creare immagini .png a partire da liste di linee spettrali e parametri casuali).
* Preparazione dataset con `tf.keras.utils.image_dataset_from_directory` (cartelle con immagini per classe).
* Rete CNN base (TensorFlow / Keras): 3 blocchi Conv2D + MaxPooling, Flatten, Dense(128), Dropout(0.5) e softmax finale.
* Pipeline di addestramento con callback utili: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `TensorBoard`.
* Transfer learning: caricamento di un modello base (`mymodel_spectra`), congelamento dei layer, aggiunta di un nuovo classificatore e fine-tuning.
* Hyperparameter tuning con **hyperopt**: definizione dello spazio di ricerca (learning rates per fasi, momentum, numero neuroni, dropout) e funzione obiettivo che esegue due fasi di training (fase 1: allenamento solo nuovi layer; fase 2: sblocco di alcuni layer e fine-tuning).
* Valutazione: accuratezza globale, accuratezza per classe (confusion matrix), grafici di loss/accuracy e salvataggio dei modelli migliori (es. `best_model.h5`, `final_composite_model`, `mymodel_spectra`, etc.).

---

## Requisiti / Dipendenze

Consigliato creare un ambiente virtuale (venv o conda). Esempio (pip):

```bash
python -m venv venv
source venv/bin/activate   # o venv\Scripts\activate su Windows
pip install -r requirements.txt
```

Esempio di `requirements.txt` (minimo necessario):

```
python>=3.8
numpy
matplotlib
tensorflow>=2.6
pandas
hyperopt
```
---
## Struttura del progetto

```text
project/
├── data/         # dataset (train/val/test, immagini spettrali organizzate per classe)
├── models/       # modelli salvati (SavedModel, .h5, checkpoint)
├── others/       # script ausiliari, file di log, risorse varie
└── progetto.ipynb # notebook principale con pipeline completa
```
---

## File/modelli prodotti (i più significativi)

* `mymodel_spectra` — modello base salvato (spettri singolo).
* `final_composite_model` — modello finale salvato per riconoscimento spettri multi-elemento.

---

## Note sui risultati e osservazioni

* Il modello che rinosce spettri singoli ha un'accuracy complessiva di circa il 97%.
* Il modello finale multi-elemnento ha un'accuracy complessiva di circa il 90% (fatica a rinoscoscere 2 classi molto simili, mentre per le altre è solido).


---

## Come contribuire / miglioramenti futuri

* Aggiungere pipeline di augmentazione più sofisticata (ruotazioni, jitter, scalings specifici per spettri).
* Valutare architetture più profonde o l'uso di modelli pre-addestrati su immagini (con adattamenti appropriati).
* Aggiungere test unitari e script CLI per training/valutazione separati dal notebook.

---
