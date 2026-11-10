# Krótki opis tego co się dzieje na tej gałęzi, czym są dane pliki i co w nim jest 

***Overview problemu w tym module :*** Estymacja posiadania piłki w zaszumionym wideo na podstawie danych wejściowych z systemów śledzenia.

***Temporal Decoding :*** Wykorzystanie wyjściowych prawdopodobieństw posiadania dla każdej klatki ( `frame 01 | 48% H | 50% A | 2% D` ), które będą dostarczone od Janka i wykorzystanie zależności czasowych do zbudowania spójnej sekwencji przebiegu gry.

***Co chce osiągnąć :*** Ograniczenie krótkich, przypadkowych przeskoków etykiet spowodowanych błędami predykcji ale przy jednoczesnym braniu pod uwagę faktycznych krótkich zmian posiadania.

### *Ukryte Modele Markowa*





## Opis plików
`config.py` - Definiuje globalne stałe, które będą używane w całym projekcie. 

`generator.py` - Tutaj są generowane dane wejściowe, które symulują to co będzie wynikiem modelu Janka. 

`viterbi_decoder.py` - Realizuje algorytm *Viterbiego* dla *Ukrytych Modeli Markowa (HMM)* operujący na logarytmach. r
