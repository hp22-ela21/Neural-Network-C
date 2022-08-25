/**************************************************************************************************
* training_data.c: Inneh�ller funktionsdefinitioner som anv�nds f�r inl�sning samt lagring av
*                  tr�ningsdata f�r neurala n�tverk.
**************************************************************************************************/
#include "training_data.h"

/* Statiska funktioner: */
static void training_data_extract(struct training_data* self, const char* s);
static bool is_digit(const char c);
static void print_line(const double* data, const size_t size, FILE* ostream);

/**************************************************************************************************
* training_data_new: Initierar angiven tr�ningsdatabeh�llare f�r lagring av tr�ningsdata till
*                    ett neuralt n�tverk.
* 
*                    - self       : Pekare till tr�ningsdatabeh�llaren.
*                    - num_inputs : Antalet noder i ing�ngslagret p� tillh�rande neuralt n�tverk.
*                    - num_outputs: Antalet noder i utg�ngslagret p� tillh�rande neuralt n�tverk.
**************************************************************************************************/
void training_data_new(struct training_data* self, 
                       const size_t num_inputs, 
                       const size_t num_outputs)
{
   double_2d_vector_new(&self->in);
   double_2d_vector_new(&self->out);
   uint_vector_new(&self->order);
   self->sets = 0;
   self->num_inputs = num_inputs;
   self->num_outputs = num_outputs;
   return;
}

/**************************************************************************************************
* training_data_delete: T�mmer angiven tr�ningsdatabeh�llare.
* 
*                       - self: Pekare till tr�ningsdatabeh�llaren.
**************************************************************************************************/
void training_data_delete(struct training_data* self)
{
   double_2d_vector_delete(&self->in);
   double_2d_vector_delete(&self->out);
   uint_vector_delete(&self->order);
   self->sets = 0;
   self->num_inputs = 0;
   self->num_outputs = 0;
   return;
}

/**************************************************************************************************
* training_data_ptr_new: Returnerar en pekare till en ny heapallokerat tr�ningsdatabeh�llare f�r
*                        lagring av tr�ningsdata till ett neuralt n�tverk.
* 
*                        - num_inputs : Antalet noder i ing�ngslagret p� det neuralt n�tverket.
*                        - num_outputs: Antalet noder i utg�ngslagret p� det neuralt n�tverket.
**************************************************************************************************/
struct training_data* training_data_ptr_new(const size_t inputs, 
                                            const size_t outputs)
{
   struct training_data* self = (struct training_data*)malloc(sizeof(struct training_data));
   if (!self) return 0;
   training_data_new(self, inputs, outputs);
   return self;
}

/**************************************************************************************************
* training_data_ptr_delete: Raderar heapallokerad tr�ningsdatabeh�llare och s�tter motsvarande
*                           pekare till null.
* 
*                           - self: Adressen till pekaren som pekar p� tr�ningsdatabeh�llaren.
**************************************************************************************************/
void training_data_ptr_delete(struct training_data** self)
{
   training_data_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* training_data_clear: Nollst�ller aktuell tr�ningsdata inf�r inl�sning av ny tr�ningsdata.
*                      D�rmed bibeh�lls information om antalet noder i ing�ngslagret samt
*                      utg�ngslagret p� tillh�rande neuralt n�tverk.
* 
*                      - self: Pekare till tr�ningsdatabeh�llaren.
**************************************************************************************************/
void training_data_clear(struct training_data* self)
{
   double_2d_vector_delete(&self->in);
   double_2d_vector_delete(&self->out);
   uint_vector_delete(&self->order);
   self->sets = 0;
   return;
}

/**************************************************************************************************
* training_data_load: L�ser in tr�ningsdata till ett neuralt n�tverk fr�n en fil via angiven 
*                     fils�kv�g och lagrar i angiven tr�ningsdatabeh�llare.
* 
*                     - self    : Pekare till tr�ningsdatabeh�llaren.
*                     - filepath: Fils�kv�g som tr�ningsdatan skall l�sas fr�n.
**************************************************************************************************/
void training_data_load(struct training_data* self, const char* filepath)
{
   FILE* fstream = fopen(filepath, "r");

   if (!fstream)
   {
      fprintf(stderr, "Could not open file at path %s!\n\n", filepath);
   }
   else
   {
      char s[100] = { '\0 ' };
      while (fgets(s, sizeof(s), fstream))
      {
         training_data_extract(self, s);
      }
      fclose(fstream);
   }
   return;

}

/**************************************************************************************************
* training_data_set: L�gger till tr�ningsdata till neuralt n�tverk via data lagrat i var sin
*                    tv�dimensionella vektor och lagrar i angiven tr�ningsdatabeh�llare.
* 
*                    - self     : Pekare till tr�ningsdatabeh�llaren.
*                    - train_in : Pekare till vektor inneh�llande tr�ningsupps�ttningarnas indata.
*                    - train_out: Pekare till vektor inneh�llande tr�ningsupps�ttningarnas utdata.
**************************************************************************************************/
void training_data_set(struct training_data* self, 
                       const struct double_2d_vector* train_in, 
                       const struct double_2d_vector* train_out)
{
   training_data_clear(self);
   self->sets = train_in->size;

   double_2d_vector_resize(&self->in, self->sets);
   double_2d_vector_resize(&self->out, self->sets);
   uint_vector_resize(&self->order, self->sets);

   self->in = *train_in;
   self->out = *train_out;

   for (size_t i = 0; i < self->sets; ++i)
   {
      self->order.data[i] = i;
   }
   return;
}

/**************************************************************************************************
* training_data_shuffle: Randomiserar den inb�rdes ordningen p� tr�ningsupps�ttningarna lagrade
*                        i angiven tr�ningsdatabeh�llare via randomisering av deras index.
* 
*                        - self: Pekare till tr�ningsdatabeh�llaren.
**************************************************************************************************/
void training_data_shuffle(struct training_data* self)
{
   for (size_t i = 0; i < self->sets; ++i)
   {
      const size_t r = (size_t)(rand() % self->sets);
      const size_t temp = self->order.data[i];
      self->order.data[i] = self->order.data[r];
      self->order.data[r] = temp;
   }

   return;
}

/**************************************************************************************************
* training_data_print: Skriver ut tr�ningsupps�ttningar lagrade i angiven tr�ningsdatabeh�llare
*                      via angiven utstr�m, d�r standardutenheten stdout anv�nds som default f�r
*                      utskrift i terminalen.
*
*                      - self   : Pekare till tr�ningsdatabeh�llaren.
*                      - ostream: Pekare till angiven utstr�m (default = stdout).
**************************************************************************************************/
void training_data_print(const struct training_data* self, 
                         FILE* ostream)
{
   if (!ostream) ostream = stdout;

   if (!self->sets)
   {
      fprintf(ostream, "No training data!\n\n");
   }
   else
   {
      fprintf(ostream, "Number of training sets: %zu\n", self->sets);
      fprintf(ostream, "Inputs: %zu\n", self->num_inputs);
      fprintf(ostream, "Outputs: % zu\n", self->num_outputs);
      fprintf(ostream, "----------------------------------------------------------------------------\n");

      for (size_t i = 0; i < self->sets; ++i)
      {
         fprintf(ostream, "Set %zu\n", i + 1);
         fprintf(ostream, "Inputs: ");
         print_line(self->in.data[i].data, self->num_inputs, ostream);

         fprintf(ostream, "Outputs: ");
         print_line(self->out.data[i].data, self->num_outputs, ostream);
         if (i < self->sets - 1) fprintf(ostream, "\n");
      }
   }

   fprintf(ostream, "----------------------------------------------------------------------------\n\n");
   return;
}

/**************************************************************************************************
* training_data_extract: Extraherar tr�ningsdata ur ett textstycke och lagrar i angiven
*                        tr�ningsdatabeh�llare ifall angivet datapunkter �verensst�mmer med 
*                        antalet noder i ing�ngslagret samt utg�ngslagret p� tillh�rande neuralt
*                        n�tverk. OBS! Se till att textfilen avslutas med en blank rad, annars 
*                        sker inte inl�sning av sista raden.
* 
*                        - self: Pekare till tr�ningsdatabeh�llaren.
*                        - s   : Pekare till det textstycke som tr�ningsdata skall extraheras ur.
**************************************************************************************************/
static void training_data_extract(struct training_data* self, 
                                  const char* s)
{
   char num_str[20] = { '\0 ' };
   struct double_vector v = { .data = 0, .size = 0 };
   size_t index = 0;
   const size_t datapoints = self->num_inputs + self->num_outputs;

   for (const char* i = s; *i; ++i)
   {
      if (is_digit(*i) || (i == s && *i == '-'))
      {
         num_str[index++] = *i;
      }
      else
      {
         num_str[index] = '\0';
         const double number = atof(num_str);
         double_vector_push(&v, number);
         index = 0;
      }
   }

   if (v.data && v.size == datapoints)
   {
      struct double_vector in = { .data = 0, .size = 0 };
      struct double_vector out = { .data = 0, .size = 0 };
      double_vector_resize(&in, self->num_inputs);
      double_vector_resize(&out, self->num_outputs);

      for (size_t i = 0; i < self->num_inputs; ++i)
      {
         in.data[i] = v.data[i];
      }
      for (size_t i = 0; i < self->num_outputs; ++i)
      {
         out.data[i] = v.data[i + self->num_inputs];
      }

      double_2d_vector_push(&self->in, &in);
      double_2d_vector_push(&self->out, &out);
      uint_vector_push(&self->order, self->sets++);
   }
   else
   {
      fprintf(stderr, "Could not extract %zu datapoints out of current line!\n\n", datapoints);
   }

   double_vector_clear(&v);
   return;
}

/**************************************************************************************************
* is_digit: Indikerar ifall angivet tecken utg�r en siffra eller ett decimaltecken (punkt).
* 
*           - c: Det tecken som skall kontrolleras.
**************************************************************************************************/
static bool is_digit(const char c)
{
   const char* s = "0123456789.";
   for (const char* i = s; *i; ++i)
   {
      if (c == *i) return true;
   }
   return false;
}

/**************************************************************************************************
* print_line: Skriver ut flyttal lagrat i angiven vektor p� en enda rad via angiven utstr�m.
* 
*             - self   : Pekare till vektorn inneh�llande flyttalen som skall skrivas ut.
*             - ostream: Pekare till angiven utstr�m.
**************************************************************************************************/
static void print_line(const double* data, const size_t size, FILE* stream)
{
   for (const double* i = data; i < data + size; ++i)
   {
      fprintf(stream, "%g ", *i);
   }

   fprintf(stream, "\n");
   return;
}
