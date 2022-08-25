/**************************************************************************************************
* ann.c: Inneh�ller funktionsdefinitioner som anv�nds f�r implementering av neurala n�tverk.
**************************************************************************************************/
#include "ann.h"

/* Statiska funktioner: */
static void ann_feedforward(struct ann* self, 
                            const struct double_vector* input);
static void ann_backpropagate(struct ann* self, 
                              const struct double_vector* reference);
static void ann_optimize(struct ann* self,
                         const double learning_rate);
static void print_line(const struct double_vector* self, 
                       FILE* ostream, 
                       const double threshold);

/**************************************************************************************************
* ann_new: Initierar angivet neuralt n�tverk. Vid start allokeras minne f�r ett enda dolt lager,
*          men vid behov kan fler l�ggas till via anrop av funktioner ann_add_hidden_layer samt
*          ann_add_hidden_layers.
* 
*          - self       : Pekare till det neurala n�tverket.
*          - num_inputs : Antalet noder i ing�ngslagret.
*          - num_hidden : Antalet noder i det dolda lagret.
*          - num_outputs: Antalet noder i utg�ngslagret.
**************************************************************************************************/
void ann_new(struct ann* self, 
             const size_t num_inputs, 
             const size_t num_hidden,
             const size_t num_outputs)
{
   self->num_inputs = num_inputs;
   self->num_outputs = num_outputs;
   self->input_layer = 0;

   dense_layer_new(&self->output_layer, self->num_outputs, num_hidden);
   training_data_new(&self->training_data, self->num_inputs, self->num_outputs);
   dense_layer_vector_new(&self->hidden_layers);
   dense_layer_vector_add_layer(&self->hidden_layers, num_hidden, num_inputs);
   return;
}

/**************************************************************************************************
* ann_delete: Nollst�ller angivet neuralt n�tverk genom att minne f�r samtliga noder och
*             tr�ningadata frig�rs.
*            
*             - self: Pekare till det neurala n�tverket.
**************************************************************************************************/
void ann_delete(struct ann* self)
{
   dense_layer_delete(&self->output_layer);
   dense_layer_vector_delete(&self->hidden_layers);
   training_data_delete(&self->training_data);

   self->input_layer = 0;
   self->num_inputs = 0;
   self->num_outputs = 0;
   return;
}

/**************************************************************************************************
* ann_ptr_new: Returnerar en pekare till ett nytt heapallokerat neuralt n�tverk, som vid start 
*              best�r av ett ing�ngslager, ett dolt lager samt ett utg�ngslager. N�tverkets
*              parametrar initieras till l�mpliga startv�rden.
*  
*              - num_inputs : Antalet noder i ing�ngslagret.
*              - num_hidden : Antalet noder i det dolda lagret.
*              - num_outputs: Antalet noder i utg�ngslagret.
**************************************************************************************************/
struct ann* ann_ptr_new(const size_t num_inputs, 
                        const size_t num_hidden,
                        const size_t num_outputs)
{
   struct ann* self = (struct ann*)malloc(sizeof(struct ann));
   if (!self) return 0;
   ann_new(self, num_inputs, num_outputs, num_hidden);
   return self;
}

/**************************************************************************************************
* ann_ptr_delete: Raderar heapallokerat neuralt n�tverk samt s�tter motsvarande pekare till null.
* 
*                 - self: Adressen till pekaren som pekar p� det heapallokerade neurala n�tverket.
**************************************************************************************************/
void ann_ptr_delete(struct ann** self)
{
   ann_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* ann_add_hidden_layer: L�gger till ett nytt dolt lager i angivet neuralt n�tverk och justerar
*                       antalet vikter per nod i utg�ngslagret efter detta.
* 
*                       - self     : Pekare till det neurala n�tverket.
*                       - num_nodes: Antalet noder i det nya dolda lagret.
**************************************************************************************************/
int ann_add_hidden_layer(struct ann* self, 
                         const size_t num_nodes)
{
   const size_t num_weights = dense_layer_vector_last(&self->hidden_layers)->num_nodes;

   if (dense_layer_vector_add_layer(&self->hidden_layers, num_nodes, num_weights))
   {
      return 1;
   }
   else
   {
      dense_layer_resize(&self->output_layer, self->num_outputs, num_nodes);
      return 0;
   } 
}

/**************************************************************************************************
* ann_add_hidden_layers: L�gger till angivet antal dolda lager i angivet neuralt n�tverk och 
*                        justerar antalet vikter per nod i utg�ngslagret efter detta.
* 
*                        - self      : Pekare till det neurala n�tverket.
*                        - num_layers: Antalet dolda lager som skall l�ggas till.
*                        - num_nodes : Antalet noder som skall l�ggas till i varje nytt dolt lager.
**************************************************************************************************/
int ann_add_hidden_layers(struct ann* self, 
                          const size_t num_layers, 
                          const size_t num_nodes)
{
   const size_t num_weights = dense_layer_vector_last(&self->hidden_layers)->num_nodes;

   if (dense_layer_vector_add_layers(&self->hidden_layers, num_layers, num_nodes, num_weights))
   {
      return 1;
   }
   else
   {
      dense_layer_resize(&self->output_layer, self->num_outputs, num_nodes);
      return 0;
   }
}

/**************************************************************************************************
* ann_load_training_data: L�ser in tr�ningsdata till angivet neuralt n�tverk fr�n en fil.
*               
*                         - self    : Pekare till det neurala n�tverket.
*                         - filepath: Pekare till fils�kv�gen som tr�ningsdatan skall l�sas fr�n.
**************************************************************************************************/
void ann_load_training_data(struct ann* self, 
                            const char* filepath)
{
   training_data_load(&self->training_data, filepath);
   return;
}

/**************************************************************************************************
* ann_set_training_data: L�gger till tr�ningsdata till angivet neuralt n�tverk lagrat via 
*                        var sin tv�dimensionell vektor.
* 
*                        - self     : Pekare till det neurala n�tverket.
*                        - train_in : Pekare till vektor med indata f�r tr�ning.
*                        - train_out: Pekare till vektor med utdata f�r tr�ning.
**************************************************************************************************/
void ann_set_training_data(struct ann* self, 
                           const struct double_2d_vector* train_in,
                           const struct double_2d_vector* train_out)
{
   training_data_set(&self->training_data, train_in, train_out);
   return;
}

/**************************************************************************************************
* ann_train: Tr�nar angivet neuralt n�tverk angivet antal epoker. Inf�r varje epok randomiseras
*            ordningen p� tr�ningsupps�ttningarna. D�refter genomf�rs en feedforward f�r att 
*            uppdatera utsignalerna i varje lager. D�refter genomf�rs en backprop f�r att ber�kna 
*            avvikelser i hela n�tverket. Slutligen sker optimering i syfte att minska uppm�tta
*            avvikelser. D�rmed justeras bias samt vikter i n�tverket f�r att minimera avvikelser
*            och d�rigenom f�rb�ttrad precision vid prediktion.
* 
*            - self         : Pekare till det neurala n�tverket.
*            - num_epochs   : Antalet epoker/omg�ng tr�ning som skall genomf�ras.
*            - learning_rate: L�rhastigheten, avg�r justeringsgraden vid avvikelse.
**************************************************************************************************/
void ann_train(struct ann* self,
               const size_t num_epochs,
               const double learning_rate)
{
   for (size_t i = 0; i < num_epochs; ++i)
   {
      training_data_shuffle(&self->training_data);
      for (size_t j = 0; j < self->training_data.sets; ++j)
      {
         const size_t k = self->training_data.order.data[j];
         const struct double_vector* input = &self->training_data.in.data[k];
         const struct double_vector* reference = &self->training_data.out.data[k];

         ann_feedforward(self, input);
         ann_backpropagate(self, reference);
         ann_optimize(self, learning_rate);
      }
   }
   return;
}

/**************************************************************************************************
* ann_predict: Genomf�r prediktion med angivet neuralt n�tverk utifr�n givna insignaler och 
*              returnerar adressen till ett f�lt inneh�llande predikterade utsignaler.
* 
*              - self : Pekare till det neurala n�tverket.
*              - input: Pekare till vektor inneh�llande indata till det neurala n�tverket.
**************************************************************************************************/
double* ann_predict(struct ann* self, 
                    const struct double_vector* input)
{
   ann_feedforward(self, input);
   return self->output_layer.output.data;
}

/**************************************************************************************************
* ann_predict_range: Genomf�r prediktion med angivet neuralt n�tverk f�r multipla kombinationer 
*                    av insignaler och genomf�r utskrift av predikterade utsignaler via angiven 
*                    utstr�m, d�r standardutenheten stdout anv�nds som default f�r utskrift i
*                    terminalen.
* 
*                    - self   : Pekare till det neurala n�tverket.
*                    - inputs : Pekare till tv�dimensionell vektor inneh�llande multipla 
*                               kombinationer av indata till det neurala n�tverket.
*                    - ostream: Pekare till angiven utstr�m (default = stdout).
**************************************************************************************************/
void ann_predict_range(struct ann* self, 
                       const struct double_2d_vector* inputs, 
                       FILE* ostream)
{
   const double threshold = 1.0 / 10000;
   if (!ostream) ostream = stdout;
   fprintf(ostream, "----------------------------------------------------------------------------\n");

   for (const struct double_vector* i = inputs->data; i < inputs->data + inputs->size; ++i)
   {
      ann_feedforward(self, i);
      fprintf(ostream, "Input: ");
      print_line(i, ostream, threshold);

      fprintf(ostream, "Predicted output: ");
      print_line(&self->output_layer.output, ostream, threshold);
   }

   fprintf(ostream, "----------------------------------------------------------------------------\n\n");
   return;
}

/**************************************************************************************************
* ann_feedforward: Ber�knar nya utsignaler f�r samtliga noder i angivet neuralt n�tverk via ny
*                  indata till n�tverkets ing�ngslager.
* 
*                  - self : Pekare till det neurala n�tverket.
*                  - input: Pekare till vektor inneh�llande indata till det neurala n�tverket.
**************************************************************************************************/
static void ann_feedforward(struct ann* self, 
                            const struct double_vector* input)
{
   const struct double_vector* hidden_output = &dense_layer_vector_last(&self->hidden_layers)->output;
   if (input->size < self->num_inputs) return;

   self->input_layer = input;
   dense_layer_vector_feedforward(&self->hidden_layers, self->input_layer);
   dense_layer_feedforward(&self->output_layer, hidden_output);
   return;
}

/**************************************************************************************************
* ann_backpropagate: Ber�knar avvikelser f�r samtliga noder i aktuellt neuralt n�tverk utefter
*                    angivna referensv�rden fr�n tr�ningsdatan.
* 
*                    - self     : Pekare till det neurala n�tverket.
*                    - reference: Referensv�rden fr�n tr�ningsdatan.
**************************************************************************************************/
static void ann_backpropagate(struct ann* self, 
                              const struct double_vector* reference)
{
   dense_layer_compare_with_reference(&self->output_layer, reference);
   dense_layer_vector_backpropagate(&self->hidden_layers, &self->output_layer);
   return;
}

/**************************************************************************************************
* ann_optimize: Minimerar avvikelser i angivet neuralt n�tverk genom att justera bias samt vikter
*               f�r samtliga noder. Angiven l�rhastighet avg�r justeringsgraden vid avvikelse.
* 
*               - self         : Pekare till det neurala n�tverket.
*               - learning_rate: L�rhastigheten, avg�r justeringsgraden vid avvikelse.
**************************************************************************************************/
static void ann_optimize(struct ann* self,
                         const double learning_rate)
{
   const struct double_vector* hidden_output = &dense_layer_vector_last(&self->hidden_layers)->output;
   dense_layer_optimize(&self->output_layer, hidden_output, learning_rate);
   dense_layer_vector_optimize(&self->hidden_layers, self->input_layer, learning_rate);
   return;
}

/**************************************************************************************************
* print_line: Skriver ut flyttal lagrat i angiven vektor p� en enda rad via angiven utstr�m.
*
*             - self   : Pekare till vektorn inneh�llande flyttalen som skall skrivas ut.
*             - ostream: Pekare till angiven utstr�m.
**************************************************************************************************/
static void print_line(const struct double_vector* self, 
                       FILE* ostream, 
                       const double threshold)
{
   for (const double* i = self->data; i < self->data + self->size; ++i)
   {
      if (*i > -threshold && *i < threshold)
      {
         fprintf(ostream, "0 ");
      }
      else
      {
         fprintf(ostream, "%g ", *i);
      }
   }

   fprintf(ostream, "\n");
   return;
}
