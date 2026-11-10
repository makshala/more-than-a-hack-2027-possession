# Krótki opis tego co się dzieje na tej gałęzi
**NOTE! :** To jest narazie szkic dokumentacji. Będzie to jeszcze poprawiane i dopieszczane (i przetłumaczone na ang) ;)

***Overview problemu w tym module :*** Estymacja posiadania piłki w zaszumionym wideo na podstawie danych wejściowych z systemów śledzenia.

***Temporal Decoding :*** Wykorzystanie wyjściowych prawdopodobieństw posiadania dla każdej klatki ( `frame 01 | 48% H | 50% A | 2% D` ), które będą dostarczone od Janka i wykorzystanie zależności czasowych do zbudowania spójnej sekwencji przebiegu gry.

***Co chce osiągnąć :*** Ograniczenie krótkich, przypadkowych przeskoków etykiet spowodowanych błędami predykcji ale przy jednoczesnym braniu pod uwagę faktycznych krótkich zmian posiadania.

### *Ukryte Modele Markowa*
Ukryte Modele Margowa (HMM - Hidden Markov Models) to model matematyczny, który pozwala wnioskować o ukrytym stanie systemu na podstawie obserwowanych danych. Właściwość Markowa polega na tym, że prawdopodobieństwo kolejnego stanu zależy od aktualnego stanu a nie od całej historii wcześniejszych stanów. Przykładowo jeżeli w danej klatce piłkę ma `H`, to w następnej klatce prawdopodobieństwo zostania przy `H` może być wysokie. HMM opiera się na dwóch rodzajach informacji: 

1) ***Prawdopodobieństwo przejścia (Transition Probability)***

Określa jakie jest prawdopodobieństwo przejścia z jednego stanu do drugiego. Przykładowa macierz przejść, gdzie wiersze to stan obecny, a kolumny to stan następny : 


| | H | A | D |
|---|---|---|---|
| **H** | 0.90 | 0.05 | 0.05 |
| **A** | 0.05 | 0.90 | 0.05 |
| **D** | 0.40 | 0.40 | 0.20 |

2) ***Prawdopodobieństwo emisji (Emission Probability)***

To informacja o tym, jakie obserwacje są prawdopodobne dla danego, ukrytego stanu. Czyli informacja jak bardzo następny stan jest prawdopodobny na bazie aktualnego stanu. 

##### Po co stosować HMM ? 
Załóżmy, że w meczu gospodarze mają piłkę przez kilka sekund. System, który analizuje obraz generuje prawdopodobieństwa dla kolejnych klatek, które mogą wyglądać następująco : 

* Frame01 - `H:90% | A:5% | D:5%`
* Frame02 - `H:88% | A:7% | D:5%`
* Frame03 - `H:15% | A:80% | D:5%`
* Frame04 - `H:87% | A:8% | D:5%`
* Frame05 - `H:91% | A:4% | D:5%`

Patrząc na te wyniki można założyć, że w `Frame03` wystąpił błąd pomiarowy. Gdyby kryterium wyboru następnego stanu było tylko największe prawdopodobieństwo otrzymalibyśmy : 

`H -> H -> A -> H -> H` - To może być niepoprawne 

HMM pozwala uwzględnić nie tylko prawdopodobieństwa obserwacji, ale również prawdopodobieństwa przejść między stanami, dlatego można uznać, że bardziej prawdopodobna jest sekwencja : 

`H -> H -> H -> H -> H`

Podsumowując HMM analizuję sekwencję jako całość, a nie każdą klatkę całkowicie niezależnie. 







## Opis plików
`config.py` - Definiuje globalne stałe, które będą używane w całym projekcie. 

`generator.py` - Tutaj są generowane dane wejściowe, które symulują to co będzie wynikiem modelu Janka. 

`viterbi_decoder.py` - Realizuje algorytm *Viterbiego* dla *Ukrytych Modeli Markowa (HMM)* operujący na logarytmach. r
