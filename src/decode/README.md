
# Krótki opis tego co się dzieje na tej gałęzi
**NOTE! :** To jest szkic dokumentacji. Będzie to jeszcze poprawiane i dopieszczane (i przetłumaczone na ang)

***Overview problemu w tym module :*** Estymacja posiadania piłki w zaszumionym wideo na podstawie danych wejściowych z systemów śledzenia.

***Temporal Decoding :*** Wykorzystanie wyjściowych prawdopodobieństw posiadania dla każdej klatki ( `frame 01 | 48% H | 50% A | 2% D` ), które będą dostarczone od Janka i wykorzystanie zależności czasowych do zbudowania spójnej sekwencji przebiegu gry.

***Co chce osiągnąć :*** Ograniczenie krótkich, przypadkowych przeskoków etykiet spowodowanych błędami predykcji ale przy jednoczesnym braniu pod uwagę faktycznych krótkich zmian posiadania.

### *Ukryte Modele Markowa*
Ukryte Modele Markowa (HMM - Hidden Markov Models) to model matematyczny, który pozwala wnioskować o ukrytym stanie systemu na podstawie obserwowanych danych. Właściwość Markowa polega na tym, że prawdopodobieństwo kolejnego stanu zależy od aktualnego stanu a nie od całej historii wcześniejszych stanów. Jest prosty wzór, który matematycznie opisuje tę zależność

$$
p_{ij}^{(n)} = P(E_j^{(n)} \mid E_i^{(n-1)})
$$

Gdzie :
* $p_{ij}^{(n)}$ – Prawdopodobieństwo przejścia w kroku czasowym $n$ ze stanu wyjściowego $i$ do stanu docelowego $j$.
* $P(X \mid Y)$ – Operator prawdopodobieństwa warunkowego (prawdopodobieństwo zajścia zdarzenia $X$ pod warunkiem, że zaszło zdarzenie $Y$).
* $E_j^{(n)}$ – Zdarzenie oznaczające, że układ w bieżącej chwili $n$ przyjmuje stan $j$.
* $E_i^{(n-1)}$ – Zdarzenie oznaczające, że układ w poprzedniej chwili $n-1$ przyjmował stan $i$.
* $i, j$ – Zmienne indeksujące stany z przestrzeni stanów (w Twoim projekcie: $0$ (Home), $1$ (Away), $2$ (Dead Ball)).
* $n$ – Dyskretny wskaźnik czasu (numer bieżącej klatki wideo).

Przykładowo jeżeli w danej klatce piłkę ma `H`, to w następnej klatce prawdopodobieństwo zostania przy `H` może być wysokie. HMM opiera się na dwóch rodzajach informacji: 

1) ***Prawdopodobieństwo przejścia (Transition Probability)***

Określa jakie jest prawdopodobieństwo przejścia z jednego stanu do drugiego. Przykładowa macierz przejść, gdzie wiersze to stan obecny, a kolumny to stan następny : 


| | H | A | D |
|---|---|---|---|
| **H** | 0.90 | 0.05 | 0.05 |
| **A** | 0.05 | 0.90 | 0.05 |
| **D** | 0.40 | 0.40 | 0.20 |



2) ***Prawdopodobieństwo emisji (Emission Probability)***

To kluczowy parametr w ukrytych modelach Markowa, który określa szansę na zaobserwowanie konkretnego sygnału, gdy system znajduje się w danym ukrytym stanie. 

##### Po co stosować HMM ? 
Załóżmy, że w meczu gospodarze mają piłkę przez kilka sekund. System, który analizuje obraz generuje prawdopodobieństwa dla kolejnych klatek, które mogą wyglądać następująco : 

* Frame01 - `H:90% | A:5% | D:5%`
* Frame02 - `H:88% | A:7% | D:5%`
* Frame03 - `H:15% | A:80%| D:5%`
* Frame04 - `H:87% | A:8% | D:5%`
* Frame05 - `H:91% | A:4% | D:5%`

Patrząc na te wyniki można założyć, że w `Frame03` wystąpił błąd pomiarowy. Gdyby kryterium wyboru następnego stanu było tylko największe prawdopodobieństwo otrzymalibyśmy : 

`H -> H -> A -> H -> H` - To może być niepoprawne 

HMM pozwala uwzględnić nie tylko prawdopodobieństwa obserwacji, ale również prawdopodobieństwa przejść między stanami, dlatego można uznać, że bardziej prawdopodobna jest sekwencja : 

`H -> H -> H -> H -> H`

Podsumowując HMM analizuję sekwencję jako całość, a nie każdą klatkę całkowicie niezależnie. 
___

### Algorytm Viterbiego 
Algorytm Viterbiego to algorytm programowania dynamicznego służący do znajdowania najbardziej prawdopodobnej sekwencji ukrytych stanów (tzw. ścieżki Viterbiego) na podstawie ciągu obserwacji w Ukrytych Modelach Markowa. Zamiast weryfikować każdą możliwą ścieżkę i wybierać z nich najlepszą ( co byłoby bardzo nieoptymalne pamięciowo, bo złożoność jest równa $O(3^{ NumOfFrames })$ ), algorytm w każdym kroku czasowym zachowuje jedynie najlepszą drogę dojścia do danego węzła / stanu. Załóżmy, że mamy prawdopodobieństwa początkowe, które wynoszą :

$\pi = [0.4, 0.5, 0.1]$

Dla pierwszej klatki (t=1) oblicza się prawdopodobieństwo przebywania w każdym ze stanów, mnożąc wektor stanów przez pierwszą emisję : 

$$V_{1, j} = \pi_j \cdot B_j(O_1)$$

Dla każdej kolejnej klatki $t$ i dla każdego możliwego stanu docelowego $j$, algorytm sprawdza wszystkie możliwe stany początkowe $i$ z poprzedniej klatki. 

$$V_{t, j} = \max_{i=1 \dots N} \left( V_{t-1, i} \cdot A_{ij} \right) \cdot B_j(O_t)$$

## Opis plików
`config.py` - Definiuje globalne stałe, które będą używane w całym projekcie. 

`generator.py` - Tutaj są generowane dane wejściowe, które symulują to co będzie wynikiem modelu Janka. 

`viterbi_decoder.py` - Realizuje algorytm *Viterbiego* dla *Ukrytych Modeli Markowa (HMM)* operujący na logarytmach. 


