/**************************************************************************************************
* ann.c: Innehåller funktionsdefinitioner som används för implementering av neurala nätverk.
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
* ann_new: Initierar angivet neuralt nätverk. Vid start allokeras minne för ett enda dolt lager,
*          men vid behov kan fler läggas till via anrop av funktioner ann_add_hidden_layer samt
*          ann_add_hidden_layers.
* 
*          - self       : Pekare till det neurala nätverket.
*          - num_inputs : Antalet noder i ingångslagret.
*          - num_hidden : Antalet noder i det dolda lagret.
*          - num_outputs: Antalet noder i utgångslagret.
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
* ann_delete: Nollställer angivet neuralt nätverk genom att minne för samtliga noder och
*             träningadata frigörs.
*            
*             - self: Pekare till det neurala nätverket.
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
* ann_ptr_new: Returnerar en pekare till ett nytt heapallokerat neuralt nätverk, som vid start 
*              består av ett ingångslager, ett dolt lager samt ett utgångslager. Nätverkets
*              parametrar initieras till lämpliga startvärden.
*  
*              - num_inputs : Antalet noder i ingångslagret.
*              - num_hidden : Antalet noder i det dolda lagret.
*              - num_outputs: Antalet noder i utgångslagret.
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
* ann_ptr_delete: Raderar heapallokerat neuralt nätverk samt sätter motsvarande pekare till null.
* 
*                 - self: Adressen till pekaren som pekar på det heapallokerade neurala nätverket.
**************************************************************************************************/
void ann_ptr_delete(struct ann** self)
{
   ann_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* ann_add_hidden_layer: Lägger till ett nytt dolt lager i angivet neuralt nätverk och justerar
*                       antalet vikter per nod i utgångslagret efter detta.
* 
*                       - self     : Pekare till det neurala nätverket.
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
* ann_add_hidden_layers: Lägger till angivet antal dolda lager i angivet neuralt nätverk och 
*                        justerar antalet vikter per nod i utgångslagret efter detta.
* 
*                        - self      : Pekare till det neurala nätverket.
*                        - num_layers: Antalet dolda lager som skall läggas till.
*                        - num_nodes : Antalet noder som skall läggas till i varje nytt dolt lager.
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
* ann_load_training_data: Läser in träningsdata till angivet neuralt nätverk från en fil.
*               
*                         - self    : Pekare till det neurala nätverket.
*                         - filepath: Pekare till filsökvägen som träningsdatan skall läsas från.
**************************************************************************************************/
void ann_load_training_data(struct ann* self, 
                            const char* filepath)
{
   training_data_load(&self->training_data, filepath);
   return;
}

/**************************************************************************************************
* ann_set_training_data: Lägger till träningsdata till angivet neuralt nätverk lagrat via 
*                        var sin tvådimensionell vektor.
* 
*                        - self     : Pekare till det neurala nätverket.
*                        - train_in : Pekare till vektor med indata för träning.
*                        - train_out: Pekare till vektor med utdata för träning.
**************************************************************************************************/
void ann_set_training_data(struct ann* self, 
                           const struct double_2d_vector* train_in,
                           const struct double_2d_vector* train_out)
{
   training_data_set(&self->training_data, train_in, train_out);
   return;
}

/**************************************************************************************************
* ann_train: Tränar angivet neuralt nätverk angivet antal epoker. Inför varje epok randomiseras
*            ordningen på träningsuppsättningarna. Därefter genomförs en feedforward för att 
*            uppdatera utsignalerna i varje lager. Därefter genomförs en backprop för att beräkna 
*            avvikelser i hela nätverket. Slutligen sker optimering i syfte att minska uppmätta
*            avvikelser. Därmed justeras bias samt vikter i nätverket för att minimera avvikelser
*            och därigenom förbättrad precision vid prediktion.
* 
*            - self         : Pekare till det neurala nätverket.
*            - num_epochs   : Antalet epoker/omgång träning som skall genomföras.
*            - learning_rate: Lärhastigheten, avgör justeringsgraden vid avvikelse.
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
* ann_predict: Genomför prediktion med angivet neuralt nätverk utifrån givna insignaler och 
*              returnerar adressen till ett fält innehållande predikterade utsignaler.
* 
*              - self : Pekare till det neurala nätverket.
*              - input: Pekare till vektor innehållande indata till det neurala nätverket.
**************************************************************************************************/
double* ann_predict(struct ann* self, 
                    const struct double_vector* input)
{
   ann_feedforward(self, input);
   return self->output_layer.output.data;
}

/**************************************************************************************************
* ann_predict_range: Genomför prediktion med angivet neuralt nätverk för multipla kombinationer 
*                    av insignaler och genomför utskrift av predikterade utsignaler via angiven 
*                    utström, där standardutenheten stdout används som default för utskrift i
*                    terminalen.
* 
*                    - self   : Pekare till det neurala nätverket.
*                    - inputs : Pekare till tvådimensionell vektor innehållande multipla 
*                               kombinationer av indata till det neurala nätverket.
*                    - ostream: Pekare till angiven utström (default = stdout).
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
* ann_feedforward: Beräknar nya utsignaler för samtliga noder i angivet neuralt nätverk via ny
*                  indata till nätverkets ingångslager.
* 
*                  - self : Pekare till det neurala nätverket.
*                  - input: Pekare till vektor innehållande indata till det neurala nätverket.
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
* ann_backpropagate: Beräknar avvikelser för samtliga noder i aktuellt neuralt nätverk utefter
*                    angivna referensvärden från träningsdatan.
* 
*                    - self     : Pekare till det neurala nätverket.
*                    - reference: Referensvärden från träningsdatan.
**************************************************************************************************/
static void ann_backpropagate(struct ann* self, 
                              const struct double_vector* reference)
{
   dense_layer_compare_with_reference(&self->output_layer, reference);
   dense_layer_vector_backpropagate(&self->hidden_layers, &self->output_layer);
   return;
}

/**************************************************************************************************
* ann_optimize: Minimerar avvikelser i angivet neuralt nätverk genom att justera bias samt vikter
*               för samtliga noder. Angiven lärhastighet avgör justeringsgraden vid avvikelse.
* 
*               - self         : Pekare till det neurala nätverket.
*               - learning_rate: Lärhastigheten, avgör justeringsgraden vid avvikelse.
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
* print_line: Skriver ut flyttal lagrat i angiven vektor på en enda rad via angiven utström.
*
*             - self   : Pekare till vektorn innehållande flyttalen som skall skrivas ut.
*             - ostream: Pekare till angiven utström.
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
