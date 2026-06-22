# Landscape Image Classification

Progetto realizzato per il corso di **Metodi Predittivi per l’Azienda**.

## Autori

* Giovanni Giacomo Cerasoli
* Alessandro Ricchebono
* Lorenzo Del Corso

## Obiettivo

L’obiettivo del progetto è classificare automaticamente immagini di paesaggi, assegnando a ciascuna fotografia la corretta categoria di appartenenza.

Le classi considerate sono:

```r id="m15h1a"
Buildings
Forest
Glacier
Mountain
Sea
Street
```

## Dataset

Il dataset contiene:

```r id="x8er4m"
16.826 immagini
```

Ogni immagine è stata standardizzata a:

```r id="y8am1q"
150 x 150 pixel
```

La suddivisione dei dati è la seguente:

```r id="z6q3lg"
Training set = 13.942 immagini
Test set     = 2.984 immagini
```

Le classi risultano abbastanza bilanciate sia nel training set sia nel test set.

## Preprocessing e riduzione della dimensionalità

Le immagini sono state preprocessate attraverso:

* ridimensionamento a `150 x 150` pixel;
* correzione e riorganizzazione dei set di train e test;
* riduzione del numero di feature;
* analisi delle componenti principali.

Il numero iniziale di variabili è stato ridotto da:

```r id="zv1o9b"
30.276 feature
```

a:

```r id="8uif2n"
2.000 feature
```

Per visualizzare la struttura dei dati sono state utilizzate le prime:

```r id="6b0u9x"
25 componenti principali
```

L’analisi esplorativa ha evidenziato una maggiore sovrapposizione tra:

```r id="gbsrl5"
Mountain e Glacier
Street e Buildings
```

Queste categorie risultano quindi più difficili da distinguere rispetto alle altre.

## Modelli confrontati

Sono stati confrontati quattro algoritmi di classificazione:

```r id="6pavj5"
K-Nearest Neighbors
Multilayer Perceptron
Random Forest
Support Vector Machine
```

Per il modello K-NN, attraverso diverse sperimentazioni, è stato selezionato:

```r id="nz40q0"
k = 6
```

Per la rete neurale è stata utilizzata un’architettura con:

```r id="e60s8r"
30 nodi
```

Per la Random Forest sono stati generati:

```r id="nsgkz3"
500 alberi
```

## Risultati

| Algoritmo     | Errore |
| ------------- | -----: |
| K-NN          | 0.6434 |
| MLP           | 0.3827 |
| Random Forest | 0.3331 |
| SVM           | 0.3099 |

Il modello migliore risulta quindi:

```r id="dbjm3p"
Support Vector Machine
```

con un errore di classificazione pari a:

```r id="0ryv9e"
0.3099
```

## Complessità computazionale

| Algoritmo     | Tempo di esecuzione |
| ------------- | ------------------: |
| MLP           |         243 secondi |
| Random Forest |         298 secondi |
| K-NN          |         474 secondi |
| SVM           |        1429 secondi |

Il modello SVM ottiene la migliore accuratezza, ma richiede anche il tempo di esecuzione più elevato.

## Conclusioni

Il progetto mostra come tecniche di machine learning possano essere utilizzate per classificare automaticamente immagini di paesaggi.

I risultati evidenziano un compromesso tra accuratezza e costo computazionale:

* SVM ottiene le performance migliori, ma è il modello più lento;
* Random Forest offre un buon compromesso tra errore e tempo di elaborazione;
* MLP ottiene risultati intermedi con il tempo di esecuzione più basso;
* K-NN risulta il modello meno efficace nel distinguere correttamente le categorie di paesaggio.

Le principali difficoltà di classificazione riguardano paesaggi visivamente simili, in particolare montagne e ghiacciai, oltre a immagini urbane riconducibili a strade ed edifici.

