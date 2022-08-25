/**************************************************************************************************
* double_2d_vector.c: Inneh�ller funktionsdefinitioner som anv�nds f�r implementering av 
*                     tv�dimensionella vektorer inneh�llande flyttal.
**************************************************************************************************/
#include "double_2d_vector.h"

/**************************************************************************************************
* double_2d_vector_new: Initierar angiven tv�dimensionell vektor.
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
* double_2d_vector_delete: T�mmer inneh�llet lagrat av angiven tv�dimensionell vektor.
* 
*                          - self: Pekare till den tv�dimensionella vektorn.
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
* double_2d_vector_ptr_new: Returnerar pekare till en ny heapallokerad tv�dimensionell vektor
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
* double_2d_vector_ptr_delete: Frig�r minne f�r angiven tv�dimensionell heapallokerad vektor och
*                              s�tter motsvarande pekare till null. 
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
* double_2d_vector_resize: �ndrar storleken / kapaciteten p� angiven tv�dimensionell vektor.
* 
*                          - self    : Pekare till den tv�dimensionella vektorn.
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
* double_2d_vector_push: L�gger till ett nytt element l�ngst bak i angiven tv�dimensionell vektor.
* 
*                        - self       : Pekare till den tv�dimensionella vektorn.
*                        - new_element: Pekare till det nya element som skall l�ggas till. 
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
* double_2d_vector_pop: Tar bort det sista elementet i angiven tv�dimensionell vektor, f�rutsatt 
*                       att denna inte �r tom.
*                      
*                       - self: Pekare till den tv�dimensionella vektorn.
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
* double_2d_vector_print: Skriver ut inneh�llet lagrat i angiven tv�dimensionell vektor via
*                         angiven utstr�m, d�r standardutenhet stdout anv�nds som default f�r
*                         utskrift i terminalen.
* 
*                         - self   : Pekare till den tv�dimensionella vektorn.
*                         - ostream: Pekare till aktuell utstr�m (default = stdout).
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
* double_2d_vector_begin: Returnerar adressen till det f�rsta elementet i angiven vektor.
* 
*                         - self: Pekare till den tv�dimensionella vektorn.
**************************************************************************************************/
struct double_vector* double_2d_vector_begin(const struct double_2d_vector* self)
{
   return self->data;
}

/**************************************************************************************************
* double_2d_vector_end: Returnerar adressen direkt efter det sista elementet i angiven vektor.
* 
*                       - self: Pekare till den tv�dimensionella vektorn.
**************************************************************************************************/
struct double_vector* double_2d_vector_end(const struct double_2d_vector* self)
{
   return self->data + self->size;
}

/**************************************************************************************************
* double_2d_vector_clear: T�mmer inneh�llet lagrat av angiven tv�dimensionell vektor.
* 
*                         - self: Pekare till den tv�dimensionella vektorn.
**************************************************************************************************/
void (*double_2d_vector_clear)(struct double_2d_vector* self) = &double_2d_vector_delete;
