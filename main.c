/**************************************************************************************************
* main.c: Demonstrerar implementering av ett dynamiskt neuralt nätverk, där antalet dolda lager 
*         samt parametrar i respektive lager kan justeras efter behov. Ju mer komplex mönster 
*         som nätverket skall prediktera, desto fler dolda lager behövs, vilket dock medför ökad
*         mängd beräkningar, både vid träning samt vid prediktion. Träningsdata kan antingen 
*         läsas in från en fil eller genom att pekare till två tvådimensionella vektorer 
*         innehållande flyttal passeras.
* 
*         Nedan implementeras ett neuralt nätverk innehållande tre ingångar, tre dolda lager 
*         samt en utgång, som tränas till att prediktera en tre-ingångars XOR-grind via 
*         träningsdata inläst från filen data.txt. Träningen genomförs under 10 000 epoker med 
*         en lärhastighet på 1 %, vilket medför perfekt prediktion vid kompilering med Visual C++
*         samt körning i Visual Studio. Dessa parametrar kan behöva ändras vid körning i en annan
*         miljö, exempelvis vid körning i ett Linuxbaserat operativsystem.
* 
*         Vid körning i Linux, kompilera koden och skapa en fil döpt main med följande kommando:
*         $ gcc *.c -o main -Wall
* 
*         Kör sedan programmet med följande kommando:
*         $ ./main
**************************************************************************************************/
#include "ann.h"

/**************************************************************************************************
* main: Skapar ett neuralt nätverk, läser in träningsdata från filen data.txt och tränar modellen.
*       Träningsdatan används sedan som insignaler för att testa modellen, där varje binär
*       kombination av insignalerna testas en efter en och motsvarande predikterade värde skrivs 
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