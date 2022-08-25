/**************************************************************************************************
* ann.h: Innehåller funktionalitet för implementering av neurala nätverk innehållande multipla
*        dense-lager via strukten ann samt motsvarande externa funktioner.
**************************************************************************************************/
#ifndef ANN_H_
#define ANN_H_

/* Inkluderingsdirektiv: */
#include "def.h"
#include "double_vector.h"
#include "dense_layer.h"
#include "dense_layer_vector.h"
#include "training_data.h"

/**************************************************************************************************
* ann: Implementering av ett neuralt nätverk innehållande ett ingångslager, valfritt antal
*      dolda lager samt ett yttre lager. Antalet noder i respektive lager är valbart.
**************************************************************************************************/
struct ann
{ 
   struct dense_layer output_layer;         /* Yttre lager. */
   struct dense_layer_vector hidden_layers; /* Fält innehållande dolda lager. */
   struct training_data training_data;      /* Behållare för träningsdata. */
   const struct double_vector* input_layer; /* Pekare till insignaler i ingångslagret. */
   size_t num_inputs;                       /* Antalet insignaler. */
   size_t num_outputs;                      /* Antalet utsignaler. */
};

/* Externa funktioner: */
void ann_new(struct ann* self, 
             const size_t num_inputs, 
             const size_t num_hidden, 
             const size_t num_outputs);
void ann_delete(struct ann* self);
struct ann* ann_ptr_new(const size_t num_inputs, 
                        const size_t num_hidden,
                        const size_t num_outputs);
void ann_ptr_delete(struct ann** self);
int ann_add_hidden_layer(struct ann* self, 
                         const size_t num_nodes);
int ann_add_hidden_layers(struct ann* self, 
                          const size_t num_layers, 
                          const size_t num_nodes);
void ann_load_training_data(struct ann* self, 
                            const char* filepath);
void ann_set_training_data(struct ann* self, 
                           const struct double_2d_vector* train_in, 
                           const struct double_2d_vector* train_out);
void ann_train(struct ann* self,
               const size_t num_epochs,
               const double learning_rate);
double* ann_predict(struct ann* self, 
                    const struct double_vector* input);
void ann_predict_range(struct ann* self, 
                       const struct double_2d_vector* inputs, 
                       FILE* stream);

#endif /* ANN_H_ */