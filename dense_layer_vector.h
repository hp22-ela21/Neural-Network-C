/**************************************************************************************************
* dense_layer_vector.h: Innehåller funktionalitet för implementering av multipla dense-lager i
*                      ett dynamiskt fält, primärt för implementering av multipla dolda lager i
*                      ett neuralt nätverk, via strukten dense_layer_vector samt motsvarande
*                      externa funktioner.
**************************************************************************************************/
#ifndef DENSE_LAYER_VECTOR_H_
#define DENSE_LAYER_VECTOR_H_

/* Inkluderingsdirektiv: */
#include "def.h"
#include "dense_layer.h"

/**************************************************************************************************
* dense_layer_vector: Dynamiskt fält innehållande dense-lager.
**************************************************************************************************/
struct dense_layer_vector
{
   struct dense_layer* data; /* Pekare till fält innehållande dense-lager. */
   size_t size;             /* Antalet dense-lager i fältet. */
};

/* Externa funktioner: */
void dense_layer_vector_new(struct dense_layer_vector* self);
void dense_layer_vector_delete(struct dense_layer_vector* self);
struct dense_layer_vector* dense_layer_vector_ptr_new(void);
void dense_layer_vector_ptr_delete(struct dense_layer_vector** self);
int dense_layer_vector_resize(struct dense_layer_vector* self, 
                              const size_t new_size);
int dense_layer_vector_push(struct dense_layer_vector* self, 
                            const struct dense_layer* new_layer);
int dense_layer_vector_pop(struct dense_layer_vector* self);
int dense_layer_vector_add_layer(struct dense_layer_vector* self, 
                                 const size_t num_nodes, 
                                 const size_t num_weights);
int dense_layer_vector_add_layers(struct dense_layer_vector* self, 
                                  const size_t num_layers,
                                  const size_t num_nodes, 
                                  const size_t num_weights);
void dense_layer_vector_print(const struct dense_layer_vector* self, 
                              FILE* ostream);
void dense_layer_vector_feedforward(struct dense_layer_vector* self, 
                                    const struct double_vector* input);
void dense_layer_vector_backpropagate(struct dense_layer_vector* self, 
                                      const struct dense_layer* output_layer);
void dense_layer_vector_optimize(struct dense_layer_vector* self, 
                                 const struct double_vector* input, 
                                 const double learning_rate);
struct dense_layer* dense_layer_vector_begin(const struct dense_layer_vector* self);
struct dense_layer* dense_layer_vector_end(const struct dense_layer_vector* self);
struct dense_layer* dense_layer_vector_last(const struct dense_layer_vector* self);

/* Funktionspekare: */
extern void (*dense_layer_vector_clear)(struct dense_layer_vector* self);

#endif /* DENSE_LAYER_VECTOR_H_ */