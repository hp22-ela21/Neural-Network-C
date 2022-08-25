/**************************************************************************************************
* training_data.h: Innehåller funktionalitet för inläsning och lagring av träningsdata till
*                  neurala nätverk. Träningsdatan kan både läsas in från en fil eller via
*                  tilldelning från tvådimensionella fält innehållande flyttal. Ordningen på
*                  träningsdatan kan också randomiseras, vilket bör göras vid träning.
**************************************************************************************************/
#ifndef TRAINING_DATA_H_
#define TRAINING_DATA_H_

/* Inkluderingsdirektiv: */
#include "def.h"
#include "double_2d_vector.h"
#include "uint_vector.h"

/**************************************************************************************************
* training_data: Strukt för lagring av träningsuppsättningar samt deras index för randomisering
*                av den inbördes ordningsföljden vid träning via träningsdatabehållare.
**************************************************************************************************/
struct training_data
{
   struct double_2d_vector in;  /* Indata. */
   struct double_2d_vector out; /* Utdata (referensvärden). */
   struct uint_vector order;    /* Ordningsföljd för träningsuppsättningarna. */
   size_t sets;                 /* Antalet träningsuppsättningar. */
   size_t num_inputs;           /* Antalet insignaler i nätverket. */
   size_t num_outputs;          /* Antalet utsignaler i nätverket. */
};

/* Externa funktioner: */
void training_data_new(struct training_data* self, 
                       const size_t num_inputs, 
                       const size_t num_outputs);
void training_data_delete(struct training_data* self);
struct training_data* training_data_ptr_new(const size_t inputs, 
                                            const size_t outputs);
void training_data_ptr_delete(struct training_data** self);
void training_data_clear(struct training_data* self);
void training_data_load(struct training_data* self,
                        const char* filepath);
void training_data_set(struct training_data* self, 
                       const struct double_2d_vector* train_in, 
                       const struct double_2d_vector* train_out);
void training_data_shuffle(struct training_data* self);
void training_data_print(const struct training_data* self, 
                         FILE* ostream);

#endif /* TRAINING_DATA_H_ */