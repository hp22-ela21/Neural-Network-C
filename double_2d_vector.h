/**************************************************************************************************
* double_2d_vector.h: Implementering av tvådimensionella vektorer innehållande flyttal via
*                     strukten double_2d_vector samt motsvarande externa funktioner.
**************************************************************************************************/
#ifndef DOUBLE_2D_VECTOR_H_
#define DOUBLE_2D_VECTOR_H_

/* Inkluderingsdirektiv: */
#include "def.h"
#include "double_vector.h"

/**************************************************************************************************
* double_2d_vector: Tvådimensionellt dynamiskt fält för lagring av flyttal. Antalet element
*                   som lagras i fältet räknas upp och uttrycks i form av vektorns storlek.
**************************************************************************************************/
struct double_2d_vector
{
   struct double_vector* data; /* Pekare till multipla dynamiska fält. */
   size_t size;                /* Antalet element i fältet. */
};

/* Externa funktioner: */
void double_2d_vector_new(struct double_2d_vector* self);
void double_2d_vector_delete(struct double_2d_vector* self);
struct double_2d_vector* double_2d_vector_ptr_new(const size_t size);
void double_2d_vector_ptr_delete(struct double_2d_vector** self);
int double_2d_vector_resize(struct double_2d_vector* self,   
                            const size_t new_size);
int  double_2d_vector_push(struct double_2d_vector* self, 
                           const struct double_vector* new_element);
int double_2d_vector_pop(struct double_2d_vector* self);
void double_2d_vector_print(const struct double_2d_vector* self, 
                            FILE* ostream);
struct double_vector* double_2d_vector_begin(const struct double_2d_vector* self);
struct double_vector* double_2d_vector_end(const struct double_2d_vector* self);

/* Funktionspekare: */
extern void (*double_2d_vector_clear)(struct double_2d_vector* self);

#endif /* DOUBLE_2D_VECTOR_H_ */