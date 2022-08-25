/**************************************************************************************************
* double_2d_vector.c: Innehåller funktionsdefinitioner som används för implementering av 
*                     tvådimensionella vektorer innehållande flyttal.
**************************************************************************************************/
#include "double_2d_vector.h"

/**************************************************************************************************
* double_2d_vector_new: Initierar angiven tvådimensionell vektor.
* 
*                       - self: Pekare till vektorn.
**************************************************************************************************/
void double_2d_vector_new(struct double_2d_vector* self)
{
   self->data = 0;
   self->size = 0;
   return;
}

/**************************************************************************************************
* double_2d_vector_delete: Tömmer innehållet lagrat av angiven tvådimensionell vektor.
* 
*                          - self: Pekare till den tvådimensionella vektorn.
**************************************************************************************************/
void double_2d_vector_delete(struct double_2d_vector* self)
{
   for (struct double_vector* i = self->data; i < self->data + self->size; ++i)
   {
      double_vector_delete(i);
   }

   free(self->data);
   self->data = 0;
   self->size = 0;
   return;
}

/**************************************************************************************************
* double_2d_vector_ptr_new: Returnerar pekare till en ny heapallokerad tvådimensionell vektor
*                           med angiven storlek.
* 
*                           - Vektorns storlek vid start (antalet element som ryms).
**************************************************************************************************/
struct double_2d_vector* double_2d_vector_ptr_new(const size_t size)
{
   struct double_2d_vector* self = (struct double_2d_vector*)malloc(sizeof(struct double_2d_vector));
   if (!self) return 0;
   self->data = 0;
   self->size = 0;
   double_2d_vector_resize(self, size);
   return self;
}

/**************************************************************************************************
* double_2d_vector_ptr_delete: Frigör minne för angiven tvådimensionell heapallokerad vektor och
*                              sätter motsvarande pekare till null. 
*
*                              - self: Adressen till vektorpekaren.
**************************************************************************************************/
void double_2d_vector_ptr_delete(struct double_2d_vector** self)
{
   double_2d_vector_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* double_2d_vector_resize: Ändrar storleken / kapaciteten på angiven tvådimensionell vektor.
* 
*                          - self    : Pekare till den tvådimensionella vektorn.
*                          - new_size: Vektorns nya storlek.
**************************************************************************************************/
int double_2d_vector_resize(struct double_2d_vector* self, 
                            const size_t new_size)
{
   struct double_vector* copy = (struct double_vector*)realloc(self->data,
      sizeof(struct double_vector) * new_size);
   if (!copy) return 1;
   self->data = copy;
   self->size = new_size;
   return 0;
}

/**************************************************************************************************
* double_2d_vector_push: Lägger till ett nytt element längst bak i angiven tvådimensionell vektor.
* 
*                        - self       : Pekare till den tvådimensionella vektorn.
*                        - new_element: Pekare till det nya element som skall läggas till. 
**************************************************************************************************/
int double_2d_vector_push(struct double_2d_vector* self, 
                          const struct double_vector* new_element)
{
   struct double_vector* copy = (struct double_vector*)realloc(self->data, 
      sizeof(struct double_vector) * (self->size + 1));
   if (!copy) return 1;
   copy[self->size++] = *new_element;
   self->data = copy;
   return 0;
}

/**************************************************************************************************
* double_2d_vector_pop: Tar bort det sista elementet i angiven tvådimensionell vektor, förutsatt 
*                       att denna inte är tom.
*                      
*                       - self: Pekare till den tvådimensionella vektorn.
**************************************************************************************************/
int double_2d_vector_pop(struct double_2d_vector* self)
{
   if (self->size <= 1)
   {
      double_2d_vector_delete(self);
      return 1;
   }
   else
   {
      struct double_vector* copy = (struct double_vector*)realloc(self->data,
         sizeof(struct double_vector) * (self->size - 1));
      if (!copy) return 1;
      self->data = copy;
      self->size--;
      return 0;
   }
}

/**************************************************************************************************
* double_2d_vector_print: Skriver ut innehållet lagrat i angiven tvådimensionell vektor via
*                         angiven utström, där standardutenhet stdout används som default för
*                         utskrift i terminalen.
* 
*                         - self   : Pekare till den tvådimensionella vektorn.
*                         - ostream: Pekare till aktuell utström (default = stdout).
**************************************************************************************************/
void double_2d_vector_print(const struct double_2d_vector* self, 
                            FILE* ostream)
{
   if (!self->size) return;
   if (!ostream) ostream = stdout;
   size_t num = 1;

   for (const struct double_vector* i = self->data; i < self->data + self->size; ++i)
   {
      fprintf(ostream, "Vector %zu:\n", num++);
      double_vector_print(i, ostream);
   }
   return;
}

/**************************************************************************************************
* double_2d_vector_begin: Returnerar adressen till det första elementet i angiven vektor.
* 
*                         - self: Pekare till den tvådimensionella vektorn.
**************************************************************************************************/
struct double_vector* double_2d_vector_begin(const struct double_2d_vector* self)
{
   return self->data;
}

/**************************************************************************************************
* double_2d_vector_end: Returnerar adressen direkt efter det sista elementet i angiven vektor.
* 
*                       - self: Pekare till den tvådimensionella vektorn.
**************************************************************************************************/
struct double_vector* double_2d_vector_end(const struct double_2d_vector* self)
{
   return self->data + self->size;
}

/**************************************************************************************************
* double_2d_vector_clear: Tömmer innehållet lagrat av angiven tvådimensionell vektor.
* 
*                         - self: Pekare till den tvådimensionella vektorn.
**************************************************************************************************/
void (*double_2d_vector_clear)(struct double_2d_vector* self) = &double_2d_vector_delete;
