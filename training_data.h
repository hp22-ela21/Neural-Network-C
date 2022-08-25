/**************************************************************************************************
* training_data.h: Inneh�ller funktionalitet f�r inl�sning och lagring av tr�ningsdata till
*                  neurala n�tverk. Tr�ningsdatan kan b�de l�sas in fr�n en fil eller via
*                  tilldelning fr�n tv�dimensionella f�lt inneh�llande flyttal. Ordningen p�
*                  tr�ningsdatan kan ocks� randomiseras, vilket b�r g�ras vid tr�ning.
**************************************************************************************************/
#ifndef TRAINING_DATA_H_
#define TRAINING_DATA_H_

/* Inkluderingsdirektiv: */
#include "def.h"
#include "double_2d_vector.h"
#include "uint_vector.h"

/**************************************************************************************************
* training_data: Strukt f�r lagring av tr�ningsupps�ttningar samt deras index f�r randomisering
*                av den inb�rdes ordningsf�ljden vid tr�ning via tr�ningsdatabeh�llare.
**************************************************************************************************/
struct training_data
{
   struct double_2d_vector in;  /* Indata. */
   struct double_2d_vector out; /* Utdata (referensv�rden). */
   struct uint_vector order;    /* Ordningsf�ljd f�r tr�ningsupps�ttningarna. */
   size_t sets;                 /* Antalet tr�ningsupps�ttningar. */
   size_t num_inputs;           /* Antalet insignaler i n�tverket. */
   size_t num_outputs;          /* Antalet utsignaler i n�tverket. */
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