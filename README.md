# QUBO Binary Classification - Gruppo G37

Progetto di Ingegneria del Software (A.A. 2025-2026).
Questo progetto implementa una pipeline di classificazione binaria che utilizza un algoritmo QUBO per la feature selection (riduzione delle caratteristiche). 

**Gruppo G37:**
* Michele Tronu (Matricola: 60/61/66470)
* Nicolò Loi (Matricola: 60/61/66461)

---

## Installazione e Setup

Assicurati di avere Python 3.11 (o versione superiore) installato. 
Per installare tutte le librerie necessarie, posizionati nella cartella principale del progetto ed esegui:

```bash
pip install -r requirements.txt
```

---

## Interfaccia Grafica (GUI)

Il progetto è dotato di una dashboard interattiva realizzata in Streamlit.
Per avviare la GUI, esegui il seguente comando dalla root del progetto:

```bash
PYTHONPATH=. streamlit run src/qubo_project/gui.py

```

Questo comando aprirà automaticamente l'interfaccia nel tuo browser predefinito (solitamente all'indirizzo `http://localhost:8501`). Dalla GUI potrai selezionare il dataset, eseguire tutte le fasi della pipeline e visualizzare i grafici e i risultati.

---

## Esecuzione dei Test Automatici

La suite di test automatizzati (scritta in pytest) verifica il corretto funzionamento di ogni fase della pipeline, come richiesto dalle specifiche. Per eseguire i test, lancia il comando:

```bash
PYTHONPATH=. pytest -v
```

---

## Interfaccia a Riga di Comando

È possibile eseguire i singoli moduli direttamente da terminale. Assicurati che i file di output vengano salvati nella cartella `outputs/`.

**1. Preprocessing:**

```bash
PYTHONPATH=. python src/qubo_project/preprocessing.py \
  --input data/input_dataset.csv \
  --target target \
  --out-data outputs/normalized.csv \
  --out-json outputs/preprocessing_result.json \
  --min-perc-valid 0.05
```

**2. Feature Selection (QUBO):**

```bash
PYTHONPATH=. python src/qubo_project/feature_selection.py \
  --in-normalized outputs/normalized.csv \
  --out-train outputs/training_reduced.csv \
  --out-test outputs/test_reduced.csv \
  --out-optimizations outputs/optimizations.csv \
  --out-json outputs/feature_selection_result.json \
  --target target \
  --perc-selected 0.20 \
  --allowance 1 \
  --perc-test 0.30 \
  --seed 42 \
  --alpha-computations 25
```

**3. Model Training:**

```bash
PYTHONPATH=. python src/qubo_project/model.py train \
  --classifier random_forest \
  --in-reduced outputs/training_reduced.csv \
  --target target \
  --out-model outputs/model.joblib \
  --out-metrics outputs/training_metrics.json \
  --seed 42
```

**4. Prediction:**

```bash
PYTHONPATH=. python src/qubo_project/model.py predict \
  --input-testset outputs/test_reduced.csv \
  --target target \
  --model outputs/model.joblib \
  --out-predictions outputs/predictions.csv \
  --out-stats outputs/classification_stats.json
```