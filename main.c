/**************************************************************************************************
* main.c: Demonstrerar implementering av ett dynamiskt neuralt n�tverk, d�r antalet dolda lager 
*         samt parametrar i respektive lager kan justeras efter behov. Ju mer komplex m�nster 
*         som n�tverket skall prediktera, desto fler dolda lager beh�vs, vilket dock medf�r �kad
*         m�ngd ber�kningar, b�de vid tr�ning samt vid prediktion. Tr�ningsdata kan antingen 
*         l�sas in fr�n en fil eller genom att pekare till tv� tv�dimensionella vektorer 
*         inneh�llande flyttal passeras.
* 
*         Nedan implementeras ett neuralt n�tverk inneh�llande tre ing�ngar, tre dolda lager 
*         samt en utg�ng, som tr�nas till att prediktera en tre-ing�ngars XOR-grind via 
*         tr�ningsdata inl�st fr�n filen data.txt. Tr�ningen genomf�rs under 10 000 epoker med 
*         en l�rhastighet p� 1 %, vilket medf�r perfekt prediktion vid kompilering med Visual C++
*         samt k�rning i Visual Studio. Dessa parametrar kan beh�va �ndras vid k�rning i en annan
*         milj�, exempelvis vid k�rning i ett Linuxbaserat operativsystem.
* 
*         Vid k�rning i Linux, kompilera koden och skapa en fil d�pt main med f�ljande kommando:
*         $ gcc *.c -o main -Wall
* 
*         K�r sedan programmet med f�ljande kommando:
*         $ ./main
**************************************************************************************************/
#include "ann.h"

/**************************************************************************************************
* main: Skapar ett neuralt n�tverk, l�ser in tr�ningsdata fr�n filen data.txt och tr�nar modellen.
*       Tr�ningsdatan anv�nds sedan som insignaler f�r att testa modellen, d�r varje bin�r
*       kombination av insignalerna testas en efter en och motsvarande predikterade v�rde skrivs 
*       ut i terminalen.
**************************************************************************************************/
int main(void)
{
   struct ann ann1;
   ann_new(&ann1, 3, 4, 1);
   ann_add_hidden_layers(&ann1, 2, 3);
   ann_load_training_data(&ann1, "data.txt");
   ann_train(&ann1, 10000, 0.01);
   const struct double_2d_vector* inputs = &ann1.training_data.in;
   ann_predict_range(&ann1, inputs, stdout);
   return 0;
}