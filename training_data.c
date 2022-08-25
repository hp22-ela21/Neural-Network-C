/**************************************************************************************************
* training_data.c: Innehåller funktionsdefinitioner som används för inläsning samt lagring av
*                  träningsdata för neurala nätverk.
**************************************************************************************************/
#include "training_data.h"

/* Statiska funktioner: */
static void training_data_extract(struct training_data* self, const char* s);
static bool is_digit(const char c);
static void print_line(const double* data, const size_t size, FILE* ostream);

/**************************************************************************************************
* training_data_new: Initierar angiven träningsdatabehållare för lagring av träningsdata till
*                    ett neuralt nätverk.
* 
*                    - self       : Pekare till träningsdatabehållaren.
*                    - num_inputs : Antalet noder i ingångslagret på tillhörande neuralt nätverk.
*                    - num_outputs: Antalet noder i utgångslagret på tillhörande neuralt nätverk.
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
* training_data_delete: Tömmer angiven träningsdatabehållare.
* 
*                       - self: Pekare till träningsdatabehållaren.
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
* training_data_ptr_new: Returnerar en pekare till en ny heapallokerat träningsdatabehållare för
*                        lagring av träningsdata till ett neuralt nätverk.
* 
*                        - num_inputs : Antalet noder i ingångslagret på det neuralt nätverket.
*                        - num_outputs: Antalet noder i utgångslagret på det neuralt nätverket.
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
* training_data_ptr_delete: Raderar heapallokerad träningsdatabehållare och sätter motsvarande
*                           pekare till null.
* 
*                           - self: Adressen till pekaren som pekar på träningsdatabehållaren.
**************************************************************************************************/
void training_data_ptr_delete(struct training_data** self)
{
   training_data_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* training_data_clear: Nollställer aktuell träningsdata inför inläsning av ny träningsdata.
*                      Därmed bibehålls information om antalet noder i ingångslagret samt
*                      utgångslagret på tillhörande neuralt nätverk.
* 
*                      - self: Pekare till träningsdatabehållaren.
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
* training_data_load: Läser in träningsdata till ett neuralt nätverk från en fil via angiven 
*                     filsökväg och lagrar i angiven träningsdatabehållare.
* 
*                     - self    : Pekare till träningsdatabehållaren.
*                     - filepath: Filsökväg som träningsdatan skall läsas från.
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
* training_data_set: Lägger till träningsdata till neuralt nätverk via data lagrat i var sin
*                    tvådimensionella vektor och lagrar i angiven träningsdatabehållare.
* 
*                    - self     : Pekare till träningsdatabehållaren.
*                    - train_in : Pekare till vektor innehållande träningsuppsättningarnas indata.
*                    - train_out: Pekare till vektor innehållande träningsuppsättningarnas utdata.
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
* training_data_shuffle: Randomiserar den inbördes ordningen på träningsuppsättningarna lagrade
*                        i angiven träningsdatabehållare via randomisering av deras index.
* 
*                        - self: Pekare till träningsdatabehållaren.
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
* training_data_print: Skriver ut träningsuppsättningar lagrade i angiven träningsdatabehållare
*                      via angiven utström, där standardutenheten stdout används som default för
*                      utskrift i terminalen.
*
*                      - self   : Pekare till träningsdatabehållaren.
*                      - ostream: Pekare till angiven utström (default = stdout).
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
* training_data_extract: Extraherar träningsdata ur ett textstycke och lagrar i angiven
*                        träningsdatabehållare ifall angivet datapunkter överensstämmer med 
*                        antalet noder i ingångslagret samt utgångslagret på tillhörande neuralt
*                        nätverk. OBS! Se till att textfilen avslutas med en blank rad, annars 
*                        sker inte inläsning av sista raden.
* 
*                        - self: Pekare till träningsdatabehållaren.
*                        - s   : Pekare till det textstycke som träningsdata skall extraheras ur.
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
* is_digit: Indikerar ifall angivet tecken utgör en siffra eller ett decimaltecken (punkt).
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
* print_line: Skriver ut flyttal lagrat i angiven vektor på en enda rad via angiven utström.
* 
*             - self   : Pekare till vektorn innehållande flyttalen som skall skrivas ut.
*             - ostream: Pekare till angiven utström.
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
