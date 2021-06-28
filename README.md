# People Counter

Projekt zaliczeniowy na przedmiot **Analiza i przetwarzanie obrazów**.

Celem  projektu  było  pozyskanie  sekwencji  video,  ze  znaną  liczbą  osób  i  opracowanie  algorytmu zliczającego osoby na każdej z poszczególnych ramek filmu. Całość osiągnięto przy pomocy *OpenCV* przy zastosowaniu do rzeczywistych filmów.

## Instalacja biblotek
Wszystkie wymagane bibloteki i moduły zostały uwzględnione w pliku *requirements.txt*:
 ```
 pip3 install -r requirements.txt
 ```

## Uruchomienie
Do projektu w folderze *videos* załączone zostały trzy przykładowe filmy, na których został zastosowany zaimplementowany algorytm. 

Przykładowe uruchomienie:
```
python3 Run.py -p mobilenet_ssd/MobileNetSSD_deploy.prototxt -m mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4

```

## Twórcy

- Baczyński Mikołaj
- Fil Klaudia
- Marcinkowska Joanna