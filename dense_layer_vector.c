/**************************************************************************************************
* dense_layer_vector.c: Inneh�ller funktionsdefinitioner som anv�nds f�r implementering av
*                       multipla dense-lager i neurala n�tverk, prim�rt avsett f�r dolda lager.
**************************************************************************************************/
#include "dense_layer_vector.h"

/**************************************************************************************************
* dense_layer_vector_new: Initierar angiven dense-lagervektor.
* 
*                         - self: Pekare till dense-lagervektorn.
**************************************************************************************************/
void dense_layer_vector_new(struct dense_layer_vector* self)
{
   self->data = 0;
   self->size = 0;
   return;
}

/**************************************************************************************************
* dense_layer_vector_delete: T�mmer angiven dense-lagervektor. 
* 
*                            - self: Pekare till dense-lagervektorn.
**************************************************************************************************/
void dense_layer_vector_delete(struct dense_layer_vector* self)
{
   free(self->data);
   self->data = 0;
   self->size = 0;
   return;
}

/**************************************************************************************************
* dense_layer_vector_ptr_new: Returerar en pekare till en ny heapallokerad dense-lagervektor.
**************************************************************************************************/
struct dense_layer_vector* dense_layer_vector_ptr_new(void)
{
   struct dense_layer_vector* self = (struct dense_layer_vector*)malloc(sizeof(struct dense_layer_vector));
   if (!self) return 0;
   self->data = 0;
   self->size = 0;
   return self;
}

/**************************************************************************************************
* dense_layer_vector_ptr_delete: Raderar heapallokerad dense-lagervektor genom att frig�ra minne
*                                och s�tter motsvarande pekare till null.
* 
*                                - self: Adressen till pekaren som pekar p� dense-lagervektorn.
**************************************************************************************************/
void dense_layer_vector_ptr_delete(struct dense_layer_vector** self)
{
   dense_layer_vector_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* dense_layer_vector_resize: �ndrar storleken angiven dense-lagervektor via omallokering.
*                            - self    : Pekare till dense-lagervektorn.
*                            - new_size: Dense-lagervektorns nya storlek efter omallokering.
**************************************************************************************************/
int dense_layer_vector_resize(struct dense_layer_vector* self, 
                              const size_t new_size)
{
   struct dense_layer* copy = (struct dense_layer*)realloc(self->data, 
      sizeof(struct dense_layer) * new_size);
   if (!copy) return 1;
   self->data = copy;
   self->size = new_size;
   return 0;
}

/**************************************************************************************************
* dense_layer_vector_push: L�gger till ett nytt dense-lager l�ngst bak angiven dense-lagervektor.
* 
*                          - self     : Pekare till dense-lagervektorn.
*                          - new_layer: Pekare till det nya lager som skall l�ggas till.
**************************************************************************************************/
int dense_layer_vector_push(struct dense_layer_vector* self, 
                            const struct dense_layer* new_layer)
{
   struct dense_layer* copy = (struct dense_layer*)realloc(self->data, 
      sizeof(struct dense_layer) * (self->size + 1));
   if (!copy) return 1;
   copy[self->size++] = *new_layer;
   self->data = copy;
   return 0;
}

/**************************************************************************************************
* dense_layer_vector_pop: Tar bort ett dense-lager l�ngst bak i angiven dense-lagervektor, 
*                         f�rutsatt att denna vektor inte �r tom.
* 
*                         - self: Pekare till dense-lagervektorn.
**************************************************************************************************/
int dense_layer_vector_pop(struct dense_layer_vector* self)
{
   if (self->size <= 1)
   {
      dense_layer_vector_delete(self);
   }
   else
   {
      struct dense_layer* copy = (struct dense_layer*)realloc(self->data,
         sizeof(struct dense_layer) * (self->size - 1));
      if (!copy) return 1;
      self->data = copy;
      self->size--;
   }
   return 0;
}

/**************************************************************************************************
* dense_layer_vector_add_layer: L�gger till ett nytt dense-lager med specificerat antal noder
*                               och vikter per nod l�ngst bak i angiven dense-lagervektor.
* 
*                               - self       : Pekare till angiven dense-lagervektor.
*                               - num_nodes  : Antalet noder i det nya dense-lagret.
*                               - num_weights: Antalet vikter per nod i det nya dense-lagret.
**************************************************************************************************/
int dense_layer_vector_add_layer(struct dense_layer_vector* self, 
                                 const size_t num_nodes,
                                 const size_t num_weights)
{
   struct dense_layer new_layer;
   dense_layer_new(&new_layer, num_nodes, num_weights);
   return dense_layer_vector_push(self, &new_layer);
}

/**************************************************************************************************
* dense_layer_vector_add_layers: L�gger till angivet antal dense-lager i angiven dense-lagervektor.
*                                Varje nytt dense-lager initieras med angivet antal noder samt
*                                vikter per nod.
*
*                               - self       : Pekare till angiven dense-lagervektor.
*                               - num_layers : Antalet dense-lager som skall l�ggas till.
*                               - num_nodes  : Antalet noder i varje nytt dense-lager.
*                               - num_weights: Antalet vikter per nod i varje nytt dense-lager.
**************************************************************************************************/
int dense_layer_vector_add_layers(struct dense_layer_vector* self, 
                                  const size_t num_layers,
                                  const size_t num_nodes, 
                                  const size_t num_weights)
{
   const size_t old_size = self->size;
   const size_t new_size = old_size + num_layers;

   if (dense_layer_vector_resize(self, new_size))
   {
      return 1;
   }
   else
   {
      struct dense_layer* begin = self->data + old_size;
      struct dense_layer* end = self->data + new_size;

      for (struct dense_layer* i = begin; i < end; ++i)
      {
         struct dense_layer new_layer;
         dense_layer_new(&new_layer, num_nodes, num_weights);
         *i = new_layer;
      }
   }

   return 0;
}

/**************************************************************************************************
* dense_layer_vector_print: Skriver ut information om varje dense-lager i angiven dense-lagervektor
*                           via angiven utstr�m.
* 
*                           - self   : Pekaren till dense-lagervektorn.
*                           - ostream: Pekare till angiven utstr�m.
**************************************************************************************************/
void dense_layer_vector_print(const struct dense_layer_vector* self, 
                              FILE* ostream)
{
   for (const struct dense_layer* i = self->data; i < self->data + self->size; ++i)
   {
      dense_layer_print(i, ostream);
   }
   return;
}

/**************************************************************************************************
* dense_layer_vector_feedforward: Uppdaterar utsignaler f�r noder i samtliga dense-lager lagrade
*                                 i angiven dense-lagervektor. Indata till f�rsta dense-lagret
*                                 utg�rs av utdata fr�n f�reg�ende ing�ngslager, vilket m�ste 
*                                 passeras. F�r resterande dense-lager anv�nds utdata fr�n 
*                                 f�reg�ende lager som indata.
* 
*                                 - self : Pekare till dense-lagervektorn.
*                                 - input: Utdata fr�n f�reg�ende ing�ngslager.
**************************************************************************************************/
void dense_layer_vector_feedforward(struct dense_layer_vector* self, 
                                    const struct double_vector* input)
{
   dense_layer_feedforward(self->data, input);

   for (struct dense_layer* i = self->data + 1; i < self->data + self->size; ++i)
   {
      const size_t num = i - self->data;
      const struct double_vector* previous_output = &((i - 1)->output);
      dense_layer_feedforward(i, previous_output);
   }
   return;
}

/**************************************************************************************************
* dense_layer_vector_backpropagate: Ber�knar avvikelser i samtliga dense-lager lagrade i angiven
*                                   dense-lagervektor. Avvikelserna i det sista dense-lagret 
*                                   ber�knas via data fr�n efterf�ljande utg�ngslager, avvikelser
*                                   i �vriga lager ber�knas via efterf�jande dense-lager.
*                                       
*                                   - self        : Pekare till dense-lagret.
*                                   - output_layer: Pekare till efterf�ljande utg�ngslager.
**************************************************************************************************/
void dense_layer_vector_backpropagate(struct dense_layer_vector* self, 
                                      const struct dense_layer* output_layer)
{
   struct dense_layer* first = self->data;
   struct dense_layer* last = self->data + self->size - 1;
   dense_layer_backpropagate(last, output_layer);

   for (struct dense_layer* i = last - 1; i >= first; --i)
   {
      dense_layer_backpropagate(i, i + 1);
   }
   return;
}

/**************************************************************************************************
* dense_layer_vector_optimize: Justerar parametrar (bias och vikter) f�r noder i samtliga
*                              dense-lager, lagrade i angiven dense-lagervektor. Utdata fr�n
*                              f�reg�ende ing�ngslager passeras som ing�ngsdata f�r optimering 
*                              av det f�rsta dense-lagret. �vriga dense-lager justeras via utdata
*                              fr�n f�reg�ende dense-lager. L�rhastigheten avg�r hur mycket 
*                              parametrarna justeras, tillsammans med uppm�tt avvikelse.
*              
*                              - self         : Pekare till dense-lagervektorn.
*                              - input        : Utdata fr�n f�reg�ende ing�ngslager.  
*                              - learning_rate: L�rhastigheten, avg�r graden av justering.
**************************************************************************************************/
void dense_layer_vector_optimize(struct dense_layer_vector* self, 
                                 const struct double_vector* input,
                                 const double learning_rate)
{
   struct dense_layer* first = self->data;
   struct dense_layer* last = self->data + self->size - 1;

   for (struct dense_layer* i = last; i > first; --i)
   {
      const struct double_vector* previous_output = &(i - 1)->output;
      dense_layer_optimize(i, previous_output, learning_rate);
   }

   dense_layer_optimize(first, input, learning_rate);
   return;
}

/**************************************************************************************************
* dense_layer_vector_begin: Returnerar adressen till det f�rsta dense-lagret i angiven 
*                           dense-lagervektor.
* 
*                           - self: Pekare till dense-lagret.
**************************************************************************************************/
struct dense_layer* dense_layer_vector_begin(const struct dense_layer_vector* self)
{
   return self->data;
}

/**************************************************************************************************
* dense_layer_vector_end: Returnerar adressen direkt efter sista dense-lagret i angiven
*                         dense-lagervektor, allts� den adress d�r dense-lagervektorn upph�r.
* 
*                         - self: Pekare till dense-lagret.
**************************************************************************************************/
struct dense_layer* dense_layer_vector_end(const struct dense_layer_vector* self)
{
   return self->data + self->size;
}

/**************************************************************************************************
* dense_layer_vector_last: Returnerar adressen till det sista dense-lagret i givet f�lt.
* 
*                          - self: Pekare till dense-lagret.
**************************************************************************************************/
struct dense_layer* dense_layer_vector_last(const struct dense_layer_vector* self)
{
   return self->size ? self->data + self->size - 1 : 0;
}

/**************************************************************************************************
* dense_layer_vector_clear: T�mmer angiven dense-lagervektor. 
* 
*                           - self: Pekare till dense-lagervektorn.
**************************************************************************************************/
void (*dense_layer_vector_clear)(struct dense_layer_vector* self) = &dense_layer_vector_delete;