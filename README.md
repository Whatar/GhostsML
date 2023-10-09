
  

![Ghosts ML](https://github.com/Whatar/GhostsML/blob/main/imgs/GhostsML.gif)

  

# Metodi di Intelligenza Artificiale per personaggi non giocanti nei videogiochi

  

  

## 1. Introduzione

  

  

Nel contesto dello sviluppo di videogiochi, l'applicazione dell'Intelligenza Artificiale (IA) per la creazione di personaggi non giocanti (NPC) rappresenta una sfida affascinante. Questi personaggi devono dimostrare comportamenti intelligenti e reattivi, arricchendo così l'esperienza di gioco. In questa tesi, si esplora l'uso di Deep Q-Network (DQN), una tecnica di apprendimento automatico, per ottenere una versione alternativa all'IA dei fantasmi nel celebre gioco Pac-Man.

  

  

### 1.1 Obiettivi del progetto

  

  

Gli obiettivi principali di questo progetto di tesi sono i seguenti:

  

  

-  **Ottenere un Intelligenza Artificiale dei fantasmi automatica**: Si cercerà di sviluppare un'IA per i fantasmi nel classico gioco arcade Pac-Man e si cercherà tramite l'allenamento e una logica epsilong greedy, di renderla in grado di navigare nell'ambiente muovendo due fantasmi (blinky e pinky), alla ricerca di pacman.

  

  

-  **Utilizzo di Deep Q-Network (DQN)**: Si sfrutteranno le capacità dell'apprendimento automatico per trovare una soluzione di coordinazione tra due fantasmi per poter catturare Pacman.

  

  

-  **Valutazione delle prestazioni**: Condurremo una valutazione accurata delle prestazioni dei fantasmi addestrati con DQN. Utilizzeremo metriche specifiche per misurare il loro comportamento in gioco.

  

  

### 1.2 Contesto e rilevanza dell'uso del ML nello sviluppo delle AI nei videogiochi

  

  

Tradizionalmente, le AI dei videogiochi sono state programmate con algoritmi come il behavior tree e altri generi di regole fisse, che chiamiamo per semplicità euristiche, che in molti casi è stato più che sufficiente per offrire una buona esperienza di gioco.
  

Talvolta però questi metodi euristici hanno portato a AI prevedibili e alquanto limitate nella loro risposta alle azioni dei giocatori. L'uso del ML consente di creare AI più razionali e adattive e che possono in alcune situazioni, migliorare l'immersione del giocatore.
  

C'è ricerca da tempo in questo campo, da divisioni come [LaForge](https://www.ubisoft.com/en-us/studio/laforge) di ubisoft, che ha prodotto alcuni paper interessanti a riguardo, come questo su un [AI di guida autonoma](https://arxiv.org/abs/1912.11077) per Watch Dogs.

  

  

### 1.3 Struttura degli argomenti

  

  

In questo progetto si osserva Pac-Man programmato utilizzando degli algoritmi che fanno uso di euristiche. per collezionare tutti i pellet e fuggire quando si trova nella vicinanza di un fantasma. I fantasmi sono mossi da un'unica DQN, e hanno come obbiettivo minimizzare la somma delle loro distanze da Pac-Man, e catturarlo.

  

  

Si vedrà, in ordine: la struttura del progetto, le metriche di valutazione, la metodologia utilizzata, l'implementazione, i problemi e le soluzioni, i risultati e la valutazione, le considerazioni future e la conclusione.

  

  

## 2. Fondamenti del Progetto

  

  

### 2.1 L'ambiente di gioco

  

  

Il labirinto ha l'aspetto e il comportamento di quello del gioco originale, ma alcuni aspetti del gioco sono stati modificati per semplificare l'implementazione dell'IA. In particolare, la frutta è ignorata da tutti i tipi di agenti (pacman/ghost), e i power pellet sono stati rimossi, questo perché nel gioco originale modificano il comportamento dei fantasmi, facendoli fuggire verso la loro cella iniziale- Lasciare questo comportamento avrebbe comportato una sorta di salto temporale dal momento in cui la DQN vede lo stato precedente al power pellet mangiato, e quello successivo, in cui alla fine del ritorno a casa da parte dell'AI originale, si sarebbe trovato teletrasportato. Il che avrebbe reso il training più instabile.

  

  

Dalla libreria si è ereditata la gestione dei nodi, che però non è sufficientemente granulare per gli scopi del progetto, quindi si è implementata una gestione di celle che permettesse inanzitutto di poter indirizzare pacman verso i pellet, e successivamente potesse fornire un'informazione più dettagliata ai fantasmi, per poterli indirizzare verso pacman.

  

  

Il labirinto è equivalente a una matrice 28x36, dove ogni cella è un quadrato di 16x16 pixel. Il labirinto è composto da 4 tipi di celle:

  

  

-  **Pareti**: Le pareti sono le celle che non possono essere attraversate da nessun personaggio. Sono rappresentate da un quadrato blu scuro.

  

-  **Passaggi**: I passaggi sono le celle che possono essere attraversate da tutti i personaggi. Sono rappresentate da un quadrato bianco.

  

-  **Portali**: Le porte sono le celle attraverso la quale si può passare per essere teletrasportati dall'altro lato del labirinto.

  

  

Pacman e i fantasmi sono equamente veloci, ma i fantasmi hanno un tempo di reazione più lento, questo per ridurre la dimensione dello spazio di ricerca, e per rendere più semplice l'allenamento.

  

  

Nello specifico, pacman può decidere ogni frame quale azione eseguire, mentre i fantasmi (la cui mossa viene scelta da una DQN per entrambi) possono decidere ogni 10 frame (0.2 secondi a 50fps).

  

  

### 2.2 Metriche di Valutazione

  

  

Per valutare l'efficacia dell'implementazione dell'algoritmi di intelligenza artificiale nei personaggi non giocanti di Pac-Man, è essenziale definire metriche di valutazione adeguate. Queste metriche consentono di misurare le prestazioni, il comportamento e il livello di sfida dei nostri NPC.

  

  

Le metriche di valutazione impostate sono:

  

  

1.  **Reward**: Il reward è una misura della performance di un agente in un determinato stato.

  

2.  **Pellet catturati da pacman**: Il numero di pellet catturati da pacman (in 5 vite), è una misura utile ad indicare la pressione che i fantasmi hanno esercitato su pacman.

  

  

## 3. Metodologia

  

  

### 3.1 Descrizione del Metodo di Apprendimento

  

  

#### Architettura del Modello

  

  

Dopo alcuni test, si è rilevato che il seguente modello è sufficientemente efficace per risolvere il problema:

  

  

-  **L1:** 6 input ( x, y di: Pinky, Blinky, Pacman)

  

-  **L2-3:** 256 Neuroni

  

-  **L4:** 16 output (4 direzioni possibili per pinky, moltiplicato per le 4 direzioni possibili per blinky)


Le funzione di attivazione adoperata per i layer intermedi è ReLU.

  

#### Algoritmo Deep Q-Network (DQN)

  

  

Il nucleo dell'approccio di apprendimento automatico implementato è l'algoritmo Deep Q-Network (DQN). Il DQN è una forma di RL che utilizza reti neurali profonde per approssimare la funzione Q, che quantifica il valore atteso delle azioni in un determinato stato. Questa funzione Q è fondamentale per prendere decisioni ottimali.

La teoria dell'apprendimento profondo applicato a giochi arcade, nello specifico atari, è un argomento trattato nel dettaglio [in questo paper](https://arxiv.org/abs/1312.5602), che fornisce una base teorica completa. Per l'implementazione si sono invece seguite le indicazioni ufficiali di [PyTorch per le DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

  

  

Il cuore del DQN è la rete neurale, che approssima la funzione Q. La rete neurale prende in input lo stato di gioco e restituisce un valore Q per ogni azione possibile. L'azione con il valore Q più alto viene selezionata e utilizzata per prendere la decisione finale. Questo quando la rete neurale è addestrata e ha imparato a stimare correttamente i valori Q per ogni azione.

L'implementazione della rete neurale, similmente all'implementazione ufficiale di PyTorch, ha questo aspetto:


    class DQN(nn.Module):
        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 256)
            self.layer2 = nn.Linear(256, 256)
            self.layer3 = nn.Linear(256, 256)
            self.layer4 = nn.Linear(256, n_actions)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
    
            return self.layer4(x)

La funzione di scelta dell'azione:

    def select_action(state, game, dt):
        global steps_done
        sample = random.random()
        if game.training:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
        else:
            eps_threshold = 0.00
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # Calculate Q-values for all actions
                q_values = policy_net(state)

                action = q_values.max(1)[1].view(1, 1)
    
                return action
        else:
            return randomAction(state, game, dt)

La azione casuale è ottenuta scegliendo un intero casuale nel range delle azioni possibili. 

    def randomAction(state, game, dt):
        # random action
        action = random.randrange(n_actions)
        # possibleActions = list(range(n_actions))
        # while not isValidAction(action, game, dt):
        #     q_values = policy_net(state)
    
        #     q_values[0][action] = -float('inf')
    
        #     possibleActions.remove(action)
        #     action = random.choice(possibleActions)
    
        return torch.tensor([[action]], device=device, dtype=torch.long)

Originariamente sono stati effettuati dei test con dei filtri intermedi all'interno della funzione di selezione casuale, per evitare che fosse selezionata un azione con evidenti risvolti negativi, come scegliere una direzione che porta almeno uno dei due fantasmi a scontrarsi contro un muro, ma dopo alcune sessioni di training è risultato che questa strategia porta a risultati peggiori rispetto a penalizzare nel reward questo comportamento, per questo motivo la parte di validazione dell'azione è stata disabilitata.
  

  

## Fasi durante l'Addestramento dell'Agente

  

  

1.  **Acquisizione dello Stato dell'Ambiente:**

  

  

- Inizialmente, l'agente rileva e acquisisce lo stato corrente dell'ambiente, che costituirà l'input per la rete neurale.

  

  

2.  **Esecuzione dell'Azione con Strategia Epsilon-Greedy:**

  

  

- Viene selezionata un'azione da eseguire nell'ambiente, utilizzando una strategia epsilon-greedy. Questo approccio bilancia l'esplorazione tramite azioni casuali con lo sfruttamento delle stime correnti della rete neurale.

  

  

3.  **Calcolo del Reward e Aggiornamento della Rete Neurale con optimize_model():**

- Dopo l'esecuzione dell'azione, si calcola il reward (che si vedrà successivamente nel dettaglio)

  

- Successivamente, la rete neurale viene aggiornata utilizzando la funzione `optimize_model()`.

  

  

Analizzando la funzione optimize_model(), i passi eseguiti sono i seguenti:

  

  

1.  **Controllo sulla dimensione della memoria:**

  

  

- Se la dimensione della memoria (`memory`) è inferiore alla dimensione del batch (`BATCH_SIZE`), non ci sono abbastanza esperienze accumulate per eseguire un adeguato aggiornamento del modello. In tal caso, la funzione interrompe l'esecuzione della fase di ottimizzazione senza eseguire ulteriori operazioni.

  

  

2.  **Estrazione di transizioni casuali dalla memoria:**

  

  

- L'estrazione casuale delle transizioni dalla memoria contribuisce a decorrelare i dati di addestramento, migliorando la stabilità dell'apprendimento e la convergenza dell'algoritmo.
- Una transizione è codificata in questo formato: `Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))` ovvero (_S, A,  _S_′, R_). La codifica dello stato è illustrata nel capitolo [3.2]()

  

  

3.  **Preparazione del batch di dati:**

  

- Trasponendo le transizioni si ottiene un formato più adatto per l'addestramento della rete neurale, consentendo una gestione più efficiente del batch.  Ovvero si spacchettano le transazioni, che sono inserite in una lista di questo tipo: 
`[[state1, action1, reward1, next_state1],[state2, action2, reward2, next_state2]]` 
e le si inseriscono in una lista più facilmente gestibile, di questo tipo:  
`[[state1, state2],[action1, action2], [reward1, reward2], [next_state1, next_state2]]`

  

- La creazione della maschera `non_final_mask` è necessaria per identificare gli stati successivi che non sono stati terminali, contribuendo a calcolare correttamente i valori Q attesi.

  

  

4.  **Calcolo dei valori Q attesi:**

  

  

- Il calcolo dei valori Q attesi per gli stati successivi si basa sulla rete target, fornendo stabilità all'addestramento attraverso la tecnica di target fixing.

  

- L'utilizzo di una maschera assicura che i valori relativi agli stati terminali non contribuiscano al calcolo dei valori Q attesi finali.

  

  

5.  **Calcolo della perdita (loss):**

  

- La perdita di Huber viene utilizzata al posto della perdita quadratica per mitigare gli effetti degli outlier nei dati di addestramento, fornendo una maggiore robustezza all'algoritmo, alcuni dettagli aggiuntivi e riferimenti ad approfondimenti disponibili [qui](https://arxiv.org/pdf/2108.12627.pdf).
In particolare, la perdita di Huber a differenza della perdita quadratica, è meno influenzata da valori estremi, come si può vedere nell'immagine sottostante.
![Huber Loss](https://en.wikipedia.org/wiki/File:Huber_loss.svg)

  

- L'applicazione della maschera assicura che la perdita venga calcolata solo per gli stati non finali nel batch, concentrando l'aggiornamento del modello sugli esempi rilevanti.

  

  

6.  **Ottimizzazione del modello:**

- Azzerare i gradienti è essenziale prima di ogni passo di ottimizzazione per evitare l'accumulo di gradienti nel modello.

  

- Il clipping dei gradienti è utilizzato per evitare problemi di instabilità numerica e per stabilizzare l'addestramento, limitando la magnitudine dei gradienti durante la retropropagazione. PyTorch mette a disposizione un metodo molto intuitivo per effettuare il clipping: 
`torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)`

  

- L'ottimizzazione mediante l'aggiornamento dei pesi del modello è il passo finale per migliorare la capacità predittiva del modello.

  

  

Per quanto concerne la memoria di replay, questa è una struttura dati che memorizza le esperienze accumulate durante il gioco, dove salviamo le sovracitate transizioni. La memoria di replay è implementata come un buffer circolare, che memorizza le esperienze come tuple `(state, action, reward, next_state)`. Queste tuple sono estratte casualmente dalla memoria durante la fase di addestramento.

  

  

### 3.2 Pre-processamento dei Dati

  

  

Lo stato di gioco viene inserito come input alla DQN sotto forma di un array di 6 elementi, la posizione (x, y) di pacman, blinky, pinky divisa per la larghezza / altezza delle celle.

In questo modo si ottiene l'indice della cella in cui pacman e gli agenti si trovano.

    state = np.zeros(6)
    state[0] = self.pacman.position.x / TILEWIDTH
    state[1] = self.pacman.position.y / TILEHEIGHT
    state[2] = self.ghosts.blinky.position.x / TILEWIDTH
    state[3] = self.ghosts.blinky.position.y / TILEHEIGHT
    state[4] = self.ghosts.pinky.position.x / TILEWIDTH
    state[5] = self.ghosts.pinky.position.y / TILEHEIGHT

  

  

  

### 3.2.2 Normalizzazione

  

  

Il processo di normalizzazione ha questo aspetto:

  

  

state[0] = state[0] / 28

  

state[1] = state[1] / 36

  

state[2] = state[2] / 28

  

state[3] = state[3] / 36

  

state[4] = state[4] / 28

  

state[5] = state[5] / 36

  

  

Dividiamo per il numero di celle verticali e orizzontali

  

  

### 3.3 Funzione di Reward

  

  

Il reward è lineare se la somma delle distanze BFS dei fantasmi da pacman è minore di 20, altrimenti è quadratico.

  

Inoltre esistono alcune penalità e bonus:

  

  

-  **Bonus**: Se pacman muore, il reward è 100

  

-  **Penalità**: Se il gioco non è finito, malus di 1, se il gioco è finito e pacman è vivo, il malus è 100, inoltre per ogni fantasma fermo il malus è 50.

  

  

Codice:

  

  

blinkyDistance = game.cells.getBFSDistance(game.ghosts.blinky.cell, game.pacman.cell)

pinkyDistance = game.cells.getBFSDistance(game.ghosts.pinky.cell, game.pacman.cell)

  

# Trattamento delle distanze non valide

if pinkyDistance == -1:

pinkyDistance = 20 # 20 is the starting distance

if blinkyDistance == -1:

blinkyDistance = 20 # 20 is the starting distance

  

# Calcolo della distanza totale

distance = pinkyDistance + blinkyDistance

  

# Calcolo del reward in base alla distanza

if distance < 20:

reward = 20 - distance

else:

reward = -(distance**2)/100

  

# Penalizzazione se i fantasmi non si sono mossi

if self.pinky.lastPosition and self.pinky.position == self.pinky.lastPosition:

reward -= 50

if self.blinky.lastPosition and self.blinky.position == self.blinky.lastPosition:

reward -= 50

  

# Aggiornamento delle ultime posizioni dei fantasmi

self.blinky.lastPosition = self.blinky.position

self.pinky.lastPosition = self.pinky.position

  

# Gestione del reward in caso di morte del pacman

if game.pacman.dead:

reward = 100

game.pacman.dead = False

terminated = True

else:

if not terminated:

reward -= 1

else:

reward -= 100

  

reward = torch.tensor([reward], device=device)

  

  

### 3.4 Ottimizzazione e Parametri

  

  

Uno schema dei cambiamenti durante la fase di ottimizzazione

![Exel](https://github.com/Whatar/GhostsML/blob/main/imgs/exel.png)

  

  

## 4. Implementazione

  

  

### 4.1 Librerie e Strumenti Utilizzati

  

  

#### 4.1.1 Pacman Code

  

  

Pacman Code è una libreria sviluppata da [Jonathan Richards](https://www.youtube.com/@jonathanrichards7969) che ho utilizzato come base per il gioco di Pacman. Ringrazio ampiamente per la generosità nell'offrire una overview completa del codice sul suo [sito](https://pacmancode.com).

  

  

#### 4.1.2 Libraries

  

  

- Python 3.10.7

  

- PyTorch 2.0.1

  

- matplotlib 3.7.1

  

- numpy 1.23.5

  

  

### 4.2 Descrizione del Codice

  

  

Le parti rilevanti sono:

  

  

**run.py** il processo principale da cui si richiamano i vari moduli per costruire l'environment, inizializzare pacman e i fantasmi, e incominciare la partita

  

  

**pacman.py** il module di gestione di pacman, comandabile sia da un player, sia da un Ai euristica con 4 livelli di forza:

  

  

- pacman lv1: Cerca la cella più distante dai due fantasmi e la imposta come proprio obbiettivo, ogni 3 secondi cambia bersaglio (questo lento aggiornamento lo costringe spesso a scontrarsi contro i fantasmi)

  

- pacman lv2: Ottimizzato per la più veloce raccolta dei pellet possibile, ignorando i fantasmi, spesso in grado di vincere contro i fantasmi originali semplicemente per la grande quantità di svolte eseguite alle intersezioni

  

- pacman lv3: In grado di fuggire i fantasmi dal momento in cui si avvicinano, ma limitato da una risorsa virtuale (denominata mana), con una ricarica di 5 secondi. Attualmente configurato per poter fuggire per 32 frame prima di dover ricaricare il mana

  

- pacman lv4: Non ha limiti di mana.

  

  

**ghosts.py** il modulo di gestione dei fantasmi e dell'algoritmo DQN. Le due funzioni più rilevanti sono complete_training, dove viene eseguito il calcolo del reward e aggiunti i dati della nuova transizione in memoria, e optimize_model, dove viene calcolato l'aggiornamento del gradiente e della target_network.

  

  

### 4.3 Problemi e Soluzioni

  

  

**Asincronia tra il Gioco e l'IA**: Un problema significativo che è stato affrontato è rappresentato dall'asincronia tra il tempo di gioco in Pac-Man e l'aggiornamento dell'IA. Inizialmente, cercavamo di aggiornare la target_network dopo un update di gioco successivo all'aver effettuato ogni azione, ma questo ha creato problemi di sincronizzazione, poiché un solo update di gioco può corrispondere a 1 frame, mentre effettivi cambiamenti nello stato di gioco avvengono in tempi più lunghi. Un diagramma del problema e della soluzione:

  

Prima del cambiamento:

  

[ **ACTION** | **LEARN** | FRAME | FRAME | FRAME | FRAME | FRAME | **ACTION** | **LEARN**]

  

Dopo il cambiamento:

  

[ **ACTION** | FRAME | FRAME | FRAME | FRAME | FRAME | **LEARN** | **ACTION** ]

  

Sostanzialmente viene analizzata la situazione di gioco appena precedente al momento in cui si rende necessario eseguire un'altra azione.


### 5 Risultati e valutazione

  

  

Alcuni grafici che mostrano i risultati delle varie configurazioni:

  

I 4 livelli di pacman contro i fantasmi originali:

  

[pcm1 pcm2 pcm3 pcm4]

  

  

I 4 livelli di pacman contro i fantasmi DQN

  

[pcm1 pcm2 pcm3 pcm4]

  

  

i 4 livelli di pacman contro una versione sperimentale con 4 fantasmi (dove ci sono due DQN, una per blinky e pinky, una per gli altri due fantasmi)

  

[pcm1 pcm2 pcm3 pcm4]

  

  

### 6 Considerazioni future e conclusione

  

  

#### 6.1 Tecniche alternative

  

  

Una tecnica alternativa è quella della Double Deep Q Network (DDQN), di cui si tratta estensivamente in questo [paper](https://arxiv.org/abs/1509.06461), utile per stabilizzare il training e evitare l'overfitting, che in effetti è un problema che si è avuto modo di incontrare, quando durante il training dei fantasmi contro pacman di livello 4, data la difficoltà nel catturare il bersaglio, i fantasmi preferivano accontentarsi di incastrarlo in una sezione piccola del labirinto per poter mantenere una piccola distanza, senza però effettivamente mangiarlo.

  

  

#### 6.2 Implicazioni e applicazioni future

  

  

Si è potuto constatare quanto sia complesso effettuare il debug di un applicazione di machine learning di questo genere, soprattutto in un ambiente custom che non segue le direttive di gymnasium. Questo è limitante perché le librerie open source di palestre di gymnasium sono spesso datate e non più compatibili con le versioni attuali di gym.

  

  

Per questo motivo si ritiene, che all'inizio di un progetto di questo tipo, sia opportuno investire del tempo per costruire dei solidi strumenti di logging, che possano mostrare cosa sta realmente accadendo durante il training.

  

Ci sono inoltre alcune interessanti esperimenti che si potrebbero avviare a partire da questo progetto. Per esempio potrebbe essere molto interessante cercare una funzione del reward che possa permettere ai fantasmi di inseguire pacman ma non necessariamente di catturarlo troppo presto, per permettergli di muoversi liberamente per un certo periodo all'inizio della partita, per poi aumentare la pressione.

  

Questo si è scoperto essere particolarmente complicato, considerando la facilità con cui, l'algoritmo di DQN di questo progetto tende a bloccarsi sui minimi locali, quindi è difficile immaginare come potrebbe, durante il corso di una partita, scoprire l'accesso ad una nuova fonte di reward, che inizialmente non era presente.

  

In alternativa si potrebbe aggiungere un malus alla cattura dipendende dal numero di pellet in gioco. Chiaramente questo richiederebbe aggiungere questa informazione all'input della DQN (anche se, anche in questo caso una DNQ probabilmente non sarebbe adatta a questo scenario più complesso).

  

Infine, si potrebbe configurare il reward per essere massimo ad una certa distanza da pacman, e diminuire questa distanza ogni x frame, ma anche questo rende l'addestramento più instabile.

  

Risulta comunque una teoria interessante, che potrebbe effettivamente permettere ai fantasmi di applicare una pressione gradualmente maggiore al giocatore.