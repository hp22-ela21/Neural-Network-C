/**************************************************************************************************
* double_vector: Vektor inneh�llande ett dynamiskt f�lt f�r lagring av flyttal. Antalet element
*                som lagras i f�ltet r�knas upp och uttrycks i form av vektorns storlek.
**************************************************************************************************/
#ifndef DOUBLE_VECTOR_H_
#define DOUBLE_VECTOR_H_

/* Inkluderingsdirektiv: */
#include "def.h"

/**************************************************************************************************
* double_vector: Vektor inneh�llande ett dynamiskt f�lt f�r lagring av flyttal. Antalet element
*                som lagras i f�ltet r�knas upp och uttrycks i form av vektorns storlek.
**************************************************************************************************/
struct double_vector
{
   double* data; /* Pekare till dynamiskt f�lt f�r lagring av flyttal. */
   size_t size;  /* Vektorns storlek (antalet element i f�ltet). */
};

/* Externa funktioner: */
void double_vector_new(struct double_vector* self);
void double_vector_delete(struct double_vector* self);
struct double_vector* double_vector_ptr_new(const size_t size);
void double_vector_ptr_delete(struct double_vector** self);
int double_vector_resize(struct double_vector* self,
                         const size_t new_size);
int double_vector_push(struct double_vector* self,
                       const double new_element);
int double_vector_pop(struct double_vector* self);
void double_vector_print(const struct double_vector* self,
                         FILE* ostream);
double* double_vector_begin(const struct double_vector* self);
double* double_vector_end(const struct double_vector* self);

/* Funktionspekare: */
extern void (*double_vector_clear)(struct double_vector* self);

#endif /* DOUBLE_VECTOR_H_ */
