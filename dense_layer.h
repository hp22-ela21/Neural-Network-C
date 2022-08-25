/**************************************************************************************************
* dense_layer.h: Inneh�ller funktionalitet f�r implementering av dense-lager i neurala n�tverk
*                via strukten dense_layer samt motsvarande externa funktioner.
**************************************************************************************************/
#ifndef DENSE_LAYER_H_
#define DENSE_LAYER_H_

/* Inkluderingsdirektiv: */
#include "def.h"
#include "double_vector.h"
#include "double_2d_vector.h"

/**************************************************************************************************
* dense_layer: Implementering av ett dense-lager i ett neuralt n�tverk, kan anv�nda f�r dolda
*              lager samt det yttre lagret i ett regulj�rt neuralt n�tverk.
**************************************************************************************************/
struct dense_layer
{
   struct double_vector output;     /* Utsignaler fr�n respektive nod.. */
   struct double_vector bias;       /* Biasv�rden / vilov�rden f�r respektive nod. */
   struct double_vector error;      /* Aktuell fel f�r respektive nod. */
   struct double_2d_vector weights; /* Vikter f�r respektive nod. */
   size_t num_nodes;                /* Antalet noder i lagret. */
   size_t num_weights;              /* Antalet vikter per nod. */
};

/* Externa funktioner: */
void dense_layer_new(struct dense_layer* self, 
                     const size_t num_nodes, 
                     const size_t num_weights);
void dense_layer_delete(struct dense_layer* self);
struct dense_layer* dense_layer_ptr_new(const size_t num_nodes, 
                                        const size_t num_weights);
void dense_layer_ptr_delete(struct dense_layer** self);
void dense_layer_clear(struct dense_layer* self);
void dense_layer_reset(struct dense_layer* self);
void dense_layer_resize(struct dense_layer* self, 
                        const size_t num_nodes, 
                        const size_t num_weights);
void dense_layer_feedforward(struct dense_layer* self, 
                             const struct double_vector* input);
void dense_layer_compare_with_reference(struct dense_layer* self, 
                                        const struct double_vector* reference);
void dense_layer_backpropagate(struct dense_layer* self, 
                               const struct dense_layer* next_layer);
void dense_layer_optimize(struct dense_layer* self, 
                          const struct double_vector* input,
                          const double learning_rate);
void dense_layer_print(const struct dense_layer* self, 
                       FILE* ostream);

#endif /* DENSE_LAYER_H_ */