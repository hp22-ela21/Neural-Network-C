/**************************************************************************************************
* dense_layer.c: Inneh�ller funktionsdefinitioner som anv�nds f�r implementering av enskilda 
*                dense-lager i neurala n�tverk.
**************************************************************************************************/
#include "dense_layer.h"

/* Statiska funktioner: */
static void dense_layer_init(struct dense_layer* self);
static void dense_layer_set_nodes(struct dense_layer* self, 
                                  const size_t num_nodes);
static void dense_layer_set_weights(struct dense_layer* self, 
                                    const size_t num_weights);
static inline double get_random_start_val(void);
static inline double relu(const double x);
static inline double delta_relu(const double x);
static void print_line(const struct double_vector* self, 
                       FILE* ostream);

/**************************************************************************************************
* dense_layer_new: Initierar angivet dense-lager. Minne allokeras f�r lagrets noder och samtliga 
*                  parametrar tilldelas startv�rden.
* 
*                  - self       : Pekare till dense-lagret.
*                  - num_nodes  : Antalet noder som skall tillf�ras i dense-lagret.
*                  - num_weights: Antalet vikter som skall tillf�ras per nod.
**************************************************************************************************/
void dense_layer_new(struct dense_layer* self, 
                     const size_t num_nodes, 
                     const size_t num_weights)
{
   double_vector_new(&self->output);
   double_vector_new(&self->bias);
   double_vector_new(&self->error);
   double_2d_vector_new(&self->weights);
   self->num_nodes = num_nodes;
   self->num_weights = num_weights;
   dense_layer_init(self);
   return;
}

/**************************************************************************************************
* dense_layer_delete: Nollst�ller angivet dense-lager.
* 
*                     - self: Pekare till dense-lagret. 
**************************************************************************************************/
void dense_layer_delete(struct dense_layer* self)
{
   double_vector_delete(&self->output);
   double_vector_delete(&self->bias);
   double_vector_delete(&self->error);
   double_2d_vector_delete(&self->weights);
   self->num_nodes = 0;
   self->num_weights = 0;
   return;
}

/**************************************************************************************************
* dense_layer_ptr_new: Returnerar en pekare till ett nytt heapallokerat dense-lager.
* 
*                      - num_nodes  : Antalet noder som skall tillf�ras i dense-lagret.
*                      - num_weights: Antalet vikter som skall tillf�ras per nod.
**************************************************************************************************/
struct dense_layer* dense_layer_ptr_new(const size_t num_nodes, 
                                        const size_t num_weights)
{
   struct dense_layer* self = (struct dense_layer*)malloc(sizeof(struct dense_layer));
   if (!self) return 0;
   dense_layer_new(self, num_nodes, num_weights);
   return self;
}

/**************************************************************************************************
* dense_layer_ptr_delete: Raderar heapallokerat dense-lager genom att frig�ra minnet f�r denna
*                         och s�tter motsvarande pekare till null. 
* 
*                         - self: Adressen till dense-lagerpekaren.
**************************************************************************************************/
void dense_layer_ptr_delete(struct dense_layer** self)
{
   dense_layer_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* dense_layer_clear: Nollst�ller parametrar i angivet dense-lager.
* 
*                    - self: Pekare till dense-lagret.
**************************************************************************************************/
void dense_layer_clear(struct dense_layer* self)
{
   double_vector_delete(&self->output);
   double_vector_delete(&self->bias);
   double_vector_delete(&self->error);
   double_2d_vector_delete(&self->weights);
   return;
}

/**************************************************************************************************
* dense_layer_reset: �terst�ller parametrar f�r angivet dense-lager.
* 
*                    - self: Pekare till dense-lagret.
**************************************************************************************************/
void dense_layer_reset(struct dense_layer* self)
{
   dense_layer_clear(self);
   dense_layer_init(self);
   return;
}
/**************************************************************************************************
* dense_layer_resize: �ndrar antalet noder och/eller vikter i angivet dense-lager.
* 
*                     - self       : Pekare till dense-lagret.
*                     - num_nodes  : Nytt antal noder i dense-lagret.
*                     - num_weights: Nytt antal vikter per nod i dense-lagret.
**************************************************************************************************/
void dense_layer_resize(struct dense_layer* self, 
                        const size_t num_nodes, 
                        const size_t num_weights)
{
   if (num_nodes != self->num_nodes)
   {
      dense_layer_set_nodes(self, num_nodes);
   }
   if (num_weights != self->num_weights)
   {
      dense_layer_set_weights(self, num_weights);
   }
   return;
}

/**************************************************************************************************
* dense_layer_feedforward: Ber�knar ny utdata f�r angivet dense-lager via ny indata.
* 
*                          - self : Pekare till dense-lagret.
*                          - input: Pekare till vektor inneh�llande ny indata.
**************************************************************************************************/
void dense_layer_feedforward(struct dense_layer* self, 
                             const struct double_vector* input)
{
   for (size_t i = 0; i < self->num_nodes; ++i)
   {
      double sum = self->bias.data[i];
      struct double_vector* weights = &self->weights.data[i];

      for (size_t j = 0; j < self->num_weights && j < input->size; ++j)
      {
         sum += input->data[j] * weights->data[j];
      }
      
      self->output.data[i] = relu(sum);
   }
   return;
}

/**************************************************************************************************
* dense_layer_compare_with_reference: Ber�knar avvikelser i angivet utg�ngslager via j�mf�relse
*                                     med referensv�rden fr�n tr�ningsdatan. 
* 
*                                     - self     : Pekare till dense-lagret.
*                                     - reference: Pekare till vektor inneh�llande referensv�rden
*                                                  fr�n tr�ningsdata.
**************************************************************************************************/
void dense_layer_compare_with_reference(struct dense_layer* self, 
                                        const struct double_vector* reference)
{
   for (size_t i = 0; i < self->num_nodes && i < reference->size; ++i)
   {
      const double error = reference->data[i] - self->output.data[i];
      self->error.data[i] = error * delta_relu(self->output.data[i]);
   }
   return;
}

/**************************************************************************************************
* dense_layer_backpropagate: Ber�knar avvikelser i angivet dolt lager via data fr�n efterf�ljande
*                            dense-lager, vilket kan vara antingen ett utg�ngslager eller ett 
*                            annat dolt lager.
*
*                            - self      : Pekare till dense-lagret.
*                            - next_layer: Pekare till efterf�ljande dense-lager.
**************************************************************************************************/
void dense_layer_backpropagate(struct dense_layer* self, 
                               const struct dense_layer* next_layer)
{
   for (size_t i = 0; i < self->num_nodes; ++i)
   {
      double deviation = 0;

      for (size_t j = 0; j < next_layer->num_nodes; ++j)
      {
         const struct double_vector* weights = &next_layer->weights.data[j];
         deviation += next_layer->error.data[j] * weights->data[i];
      }

      self->error.data[i] = deviation * delta_relu(self->output.data[i]);
   }
   return;
}

/**************************************************************************************************
* dense_layer_optimize: Justerar bias samt vikter f�r angivet dense-lager med angiven 
*                       l�rhastighet f�r att minska fel. Utdatan fr�n f�reg�ende lager, som utg�r
*                       indata p� angivet lager, anv�nds f�r att justera vikterna.
*                       
*                       - self         : Pekare till angivet dense-lager.
*                       - input        : Pekare till vektor inneh�llande utdata fr�n f�reg�ende 
*                                        dense-lager, vilket utg�r indata till angivet lager.
*                       - learning_rate: L�rhastigheten, avg�r graden av justering vid avvikelse.
**************************************************************************************************/
void dense_layer_optimize(struct dense_layer* self, 
                          const struct double_vector* input,
                          const double learning_rate)
{
   for (size_t i = 0; i < self->num_nodes; ++i)
   {
      const double change_rate = self->error.data[i] * learning_rate;
      struct double_vector* weights = &self->weights.data[i];
      self->bias.data[i] += change_rate;

      for (size_t j = 0; j < self->num_weights && j < input->size; ++j)
      {
         weights->data[j] += change_rate * input->data[j];
      }
   }

   return;
}

/**************************************************************************************************
* dense_layer_print: Skriver ut information g�llande givet dense-lager via angiven utstr�m, d�r
*                    standardutenheten stdout anv�nds som default f�r utskrift i terminalen.
* 
*                    - self   : Pekare till angivet dense-lager.
*                    - ostream: Pekare till angiven utstr�m (default = stdout).
**************************************************************************************************/
void dense_layer_print(const struct dense_layer* self, 
                       FILE* ostream)
{
   if (!self->num_nodes) return;
   if (!ostream) ostream = stdout;

   fprintf(ostream, "Number of nodes: %zu\n", self->num_nodes);
   fprintf(ostream, "Weights per node: %zu\n", self->num_weights);
   fprintf(ostream, "----------------------------------------------------------------------------\n");

   fprintf(ostream, "Outputs: ");
   print_line(&self->output, ostream);

   fprintf(ostream, "Bias: ");
   print_line(&self->bias, ostream);

   fprintf(ostream, "Error: ");
   print_line(&self->error, ostream);

   fprintf(ostream, "\nWeights:\n");

   for (size_t i = 0; i < self->num_nodes; ++i)
   {
      fprintf(ostream, "\tNode %zu: ", i + 1);
      struct double_vector* weights = &self->weights.data[i];
      print_line(weights, ostream);
   }

   fprintf(ostream, "----------------------------------------------------------------------------\n\n");
   return;
}

/**************************************************************************************************
* dense_layer_init: Allokerar minne och s�tter startv�rden p� parametrar i angivet dense-lager.
*                   Bias och vikter tilldelas randomiserade startv�rden mellan 0.0 - 1.0, �vriga
*                   parametrar tilldelas 0.0 som startv�rde.
* 
*                   - self: Pekare till dense-lagret.
**************************************************************************************************/
static void dense_layer_init(struct dense_layer* self)
{
   double_vector_resize(&self->output, self->num_nodes);
   double_vector_resize(&self->bias, self->num_nodes);
   double_vector_resize(&self->error, self->num_nodes);
   double_2d_vector_resize(&self->weights, self->num_nodes);

   for (size_t i = 0; i < self->num_nodes; ++i)
   {
      struct double_vector weights = { .data = 0, .size = 0 };
      double_vector_resize(&weights, self->num_weights);

      for (size_t j = 0; j < self->num_weights; ++j)
      {
         weights.data[j] = get_random_start_val();
      }

      self->output.data[i] = 0;
      self->bias.data[i] = get_random_start_val();
      self->error.data[i] = 0;
      self->weights.data[i] = weights;
   }

   return;
}

/**************************************************************************************************
* dense_layer_set_nodes: Justerar antalet noder i angivet dense-lager. Ifall nya noder l�ggs till
*                        s� tilldelas startv�rden till samtliga parametrar.
* 
*                        - self     : Pekare till dense-lagret.
*                        - num_nodes: Nytt antal noder i dense-lagret.
**************************************************************************************************/
static void dense_layer_set_nodes(struct dense_layer* self, 
                                  const size_t num_nodes)
{
   double_vector_resize(&self->output, num_nodes);
   double_vector_resize(&self->bias, num_nodes);
   double_vector_resize(&self->error, num_nodes);
   double_2d_vector_resize(&self->weights, num_nodes);

   if (num_nodes > self->num_nodes)
   {
      for (size_t i = self->num_nodes; i < num_nodes; ++i)
      {
         struct double_vector weights = { .data = 0, .size = 0 };
         double_vector_resize(&weights, self->num_weights);

         for (size_t j = 0; j < self->num_weights; ++j)
         {
            weights.data[j] = get_random_start_val();
         }

         self->output.data[i] = 0;
         self->bias.data[i] = get_random_start_val(); 
         self->error.data[i] = 0;
         self->weights.data[i] = weights;
      }
   }

   self->num_nodes = num_nodes;
   return;
}

/**************************************************************************************************
* dense_layer_set_weights: Justerar antalet vikter f�r varje nod i angivet dense-lager. Ifall
*                          nya vikter l�ggs till initieras dessa med randomiserade startv�rden.
* 
*                          - self: Pekare till dense-lagret.
*                          - num_weights: Nytt antal vikter per nod i dense-lagret.
**************************************************************************************************/
static void dense_layer_set_weights(struct dense_layer* self, 
                                    const size_t num_weights)
{
   for (size_t i = 0; i < self->num_nodes; ++i)
   {
      double_vector_resize(&self->weights.data[i], num_weights);
   }

   if (num_weights > self->num_weights)
   {
      for (size_t i = 0; i < self->num_nodes; ++i)
      {
         struct double_vector* weights = &self->weights.data[i];
         for (size_t j = self->num_weights; j < num_weights; ++j)
         {
            weights->data[j] = get_random_start_val();
         }
      }
   }

   self->num_weights = num_weights;
   return;
}

/**************************************************************************************************
* get_random_start_val: Returnerar ett randomiserat flyttal mellan 0.0 - 1.0.
**************************************************************************************************/
static inline double get_random_start_val(void)
{
   return rand() / (double)RAND_MAX;
}

/**************************************************************************************************
* relu: Returnerar ReLU (Rectified Linear Unit) ur angiven insignal x:
*       x > 0.0  => ReLU(x) = x
*       x <= 0.0 => ReLU(x) = 0.0
* 
*       - x: Aktuell insignal.
**************************************************************************************************/
static inline double relu(const double x)
{
   return x > 0.0 ? x : 0.0;
}

/**************************************************************************************************
* delta_relu: Returnerar derivatan av angiven insignal x:
*             x > 0.0  => ReLU(x) = x   => ReLU'(x) = 1.0
*             x <= 0.0 => ReLU(x) = 0.0 => ReLU'(x) = 0.0
* 
*             - x: Aktuell insignal.
**************************************************************************************************/
static inline double delta_relu(const double x)
{
   return x > 0.0 ? 1.0 : 0.0;
}

/**************************************************************************************************
* print_line: Skriver ut flyttal lagrat i angiven vektor p� en enda rad via angiven utstr�m.
* 
*             - self   : Pekare till vektorn inneh�llande flyttalen som skall skrivas ut.
*             - ostream: Pekare till angiven utstr�m.
**************************************************************************************************/
static void print_line(const struct double_vector* self, 
                       FILE* ostream)
{
   for (const double* i = self->data; i < self->data + self->size; ++i)
   {
      fprintf(ostream, "%g ", *i);
   }

   fprintf(ostream, "\n");
   return;
}