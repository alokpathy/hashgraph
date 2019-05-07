#include <string.h>
#include <map>
using std::map;
using std::pair;

// Routines to read/write matrix.
// Modified from http://crd-legacy.lbl.gov/~yunhe/cs267/final/source/utils/convert/matrix_io.c

// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

int mm_read_banner(FILE *f, MM_typecode *matcode);
int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
int mm_read_mtx_array_size(FILE *f, int *M, int *N);

int mm_write_banner(FILE *f, MM_typecode matcode);
int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
int mm_write_mtx_array_size(FILE *f, int M, int N);


/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)	((typecode)[0]=='M')

#define mm_is_sparse(typecode) ( ((typecode)[1]=='C') || ((typecode)[1]=='S') )
#define mm_is_sparserow(typecode)	((typecode)[1]=='S')
#define mm_is_coordinate(typecode)((typecode)[1]=='C')
#define mm_is_dense(typecode)	((typecode)[1]=='A')
#define mm_is_array(typecode)	((typecode)[1]=='A')

#define mm_is_complex(typecode)	((typecode)[2]=='C')
#define mm_is_real(typecode)		((typecode)[2]=='R')
#define mm_is_pattern(typecode)	((typecode)[2]=='P')
#define mm_is_integer(typecode) ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)((typecode)[3]=='S')
#define mm_is_general(typecode)	((typecode)[3]=='G')
#define mm_is_skew(typecode)	((typecode)[3]=='K')
#define mm_is_hermitian(typecode)((typecode)[3]=='H')

int mm_is_valid(MM_typecode matcode);		/* too complex for a macro */


/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode)	((*typecode)[0]='M')
#define mm_set_coordinate(typecode)	((*typecode)[1]='C')
#define mm_set_sparserow(typecode)	((*typecode)[1]='S')
#define mm_set_array(typecode)	((*typecode)[1]='A')
#define mm_set_dense(typecode)	mm_set_array(typecode)

#define mm_set_complex(typecode)((*typecode)[2]='C')
#define mm_set_real(typecode)	((*typecode)[2]='R')
#define mm_set_pattern(typecode)((*typecode)[2]='P')
#define mm_set_integer(typecode)((*typecode)[2]='I')


#define mm_set_symmetric(typecode)((*typecode)[3]='S')
#define mm_set_general(typecode)((*typecode)[3]='G')
#define mm_set_skew(typecode)	((*typecode)[3]='K')
#define mm_set_hermitian(typecode)((*typecode)[3]='H')

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]= \
        (*typecode)[2]=' ',(*typecode)[3]='G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)


/********************* Matrix Market error codes ***************************/


#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF		12
#define MM_NOT_MTX				13
#define MM_NO_HEADER			14
#define MM_UNSUPPORTED_TYPE		15
#define MM_LINE_TOO_LONG		16
#define MM_COULD_NOT_WRITE_FILE	17


/******************** Matrix Market internal definitions ********************

  MM_matrix_typecode: 4-character sequence

  ojbect 		sparse/   	data        storage
  dense     	type        scheme

  string position:	 [0]        [1]			[2]         [3]

  Matrix typecode:  M(atrix)  C(oord)		R(eal)   	G(eneral)
  A(array)	C(omplex)   H(ermitian)
  P(attern)   S(ymmetric)
  I(nteger)	K(kew)

 ***********************************************************************/

#define MM_MTX_STR		"matrix"
#define MM_ARRAY_STR	"array"
#define MM_DENSE_STR	"array"
#define MM_COORDINATE_STR "coordinate"
#define MM_SPARSEROW_STR "sparserow"
#define MM_COMPLEX_STR	"complex"
#define MM_REAL_STR		"real"
#define MM_INT_STR		"integer"
#define MM_GENERAL_STR  "general"
#define MM_SYMM_STR		"symmetric"
#define MM_HERM_STR		"hermitian"
#define MM_SKEW_STR		"skew-symmetric"
#define MM_PATTERN_STR  "pattern"

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;


    mm_clear_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
                storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
    for (p=crd; *p!='\0'; *p=tolower(*p),p++);
    for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
    for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);


    /* second field describes whether this is a sparse matrix (in coordinate
       storgae) or a dense array */


    if (strcmp(crd, MM_SPARSEROW_STR) == 0)
        mm_set_sparserow(matcode);
    else
        if (strcmp(crd, MM_COORDINATE_STR) == 0)
            mm_set_coordinate(matcode);
        else
            if (strcmp(crd, MM_DENSE_STR) == 0)
                mm_set_dense(matcode);
            else
                return MM_UNSUPPORTED_TYPE;


    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
        if (strcmp(data_type, MM_COMPLEX_STR) == 0)
            mm_set_complex(matcode);
        else
            if (strcmp(data_type, MM_PATTERN_STR) == 0)
                mm_set_pattern(matcode);
            else
                if (strcmp(data_type, MM_INT_STR) == 0)
                    mm_set_integer(matcode);
                else
                    return MM_UNSUPPORTED_TYPE;


    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
        if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
            mm_set_symmetric(matcode);
        else
            if (strcmp(storage_scheme, MM_HERM_STR) == 0)
                mm_set_hermitian(matcode);
            else
                if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
                    mm_set_skew(matcode);
                else
                    return MM_UNSUPPORTED_TYPE;


    return 0;
}

#ifndef __NVCC__
char *strdup (const char *s) {
    char *d = (char*)my_malloc (strlen (s) + 1);   	// Allocate memory
    if (d != NULL) strcpy (d,s);         		// Copy string if okay
    return d;                            		// Return new memory
}
#endif

char  *mm_typecode_to_str(MM_typecode matcode)
{
    char buffer[MM_MAX_LINE_LENGTH];
    const char *types[4];

    /* check for MTX type */
    if (mm_is_matrix(matcode))
        types[0] = MM_MTX_STR;

    /* check for CRD or ARR matrix */
    if (mm_is_sparserow(matcode))
        types[1] = MM_SPARSEROW_STR;
    else
        if (mm_is_coordinate(matcode))
            types[1] = MM_COORDINATE_STR;
        else
            if (mm_is_dense(matcode))
                types[1] = MM_DENSE_STR;
            else
                return NULL;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else
        if (mm_is_complex(matcode))
            types[2] = MM_COMPLEX_STR;
        else
            if (mm_is_pattern(matcode))
                types[2] = MM_PATTERN_STR;
            else
                if (mm_is_integer(matcode))
                    types[2] = MM_INT_STR;
                else
                    return NULL;


    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else
        if (mm_is_symmetric(matcode))
            types[3] = MM_SYMM_STR;
        else
            if (mm_is_hermitian(matcode))
                types[3] = MM_HERM_STR;
            else
                if (mm_is_skew(matcode))
                    types[3] = MM_SKEW_STR;
                else
                    return NULL;

    sprintf(buffer,"%s %s %s %s", types[0], types[1], types[2], types[3]);
    return strdup(buffer);

}

/* generates random double in [low, high) */
double random_double (double low, double high)
{
    //return  ((high-low)*drand48()) + low;
    return  ((high-low)*rand()/RAND_MAX) + low;
}

void coo2csr_in(int n, int nz, double *a, int **i_idx, int *j_idx);	// in-place conversion, also replaces i_idx with new array of size (n+1) to save memory

/* write CSR format */
/* 1st line : % number_of_rows number_of_columns number_of_nonzeros
   2nd line : % base of index
   3rd line : row_number  nz_r(=number_of_nonzeros_in_the_row)
   next nz_r lines : column_index value(when a != NULL)
   next line : row_number  nz_r(=number_of_nonzeros_in_the_row)
   next nz_r lines : column_index value(when a != NULL)
   ...
   */

void write_csr (char *fn, int m, int n, int nz,
        int *row_start, int *col_idx, double *a)
{
    FILE *f;
    int i, j;

    if ((f = fopen(fn, "w")) == NULL){
        printf ("can't open file <%s> \n", fn);
        exit(1);
    }

    fprintf (f, "%s %d %d %d\n", "%", m, n, nz);

    for (i=0; i<m; i++){
        fprintf(f, "%d %d\n", i, row_start[i+1]-row_start[i]);

        for (j=row_start[i]; j<row_start[i+1]; j++){
            if (a)
                fprintf(f, "%d %g\n", col_idx[j], a[j]);
            else
                fprintf(f, "%d\n", col_idx[j]);
        }
    }

    fclose (f);

}


/* reads matrix market format (coordinate) and returns
   csr format */

void read_mm_matrix (char *fn, int *m, int *n, int *nz,
        int **i_idx, int **j_idx, double **a)
{
    MM_typecode matcode;
    FILE *f;
    int i,k;
    int base=1;

    if ((f = fopen(fn, "r")) == NULL) {
        printf ("can't open file <%s> \n", fn);
        exit(1);
    }
    if (mm_read_banner(f, &matcode) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (! (mm_is_matrix(matcode) && mm_is_sparse(matcode)) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* skip comments */
    unsigned long pos;
    char *line = NULL;
    size_t len = 0;
    size_t read;
    do {
      pos = ftell(f);
      read = getline(&line, &len, f);
    } while (read != -1 && line[0] == '%');
    fseek(f, pos, SEEK_SET);

    /* find out size of sparse matrix .... */
    if (fscanf(f, "%d %d %d", m, n, nz) != 3) {
        printf("Error reading matrix header: m n nz\n");
        exit(1);
    }


	//We always create back edges if doesnt exist

    /* reserve memory for matrices */
    //if (mm_is_symmetric(matcode)){
        *i_idx = (int *) my_malloc(*nz *2 * sizeof(int));
        *j_idx = (int *) my_malloc(*nz *2 * sizeof(int));
        *a = (double *) my_malloc(*nz *2 * sizeof(double));
    
    /*
    }
    else {
        *i_idx = (int *) my_malloc(*nz * sizeof(int));
        *j_idx = (int *) my_malloc(*nz * sizeof(int));
        *a = (double *) my_malloc(*nz * sizeof(double));
    }

    if (!(*i_idx) || !(*j_idx) || !(*a)){
        printf ("cannot allocate memory for %d, %d, %d sparse matrix\n", *m, *n, *nz);
        exit(1);
    }
    */
	
	map<pair<int,int>,double> raw_edges; // map edge(u,v) - > indice of edges in *a
    k=0;
    for (i=0; i<*nz; i++)  {
	int u,v;
	double d;

        if (mm_is_pattern(matcode)){
            if (fscanf(f, "%d %d", &u, &v) != 2) {
                printf("Error reading matrix entry %i\n", i);
                exit(1);
            }


            d = random_double(0.5, 1.0);
        }
        else if (mm_is_real(matcode)){
            if (fscanf(f, "%d %d %lg", &u, &v, &d) != 3) {
                printf("Error reading matrix entry %i\n", i);
                exit(1);
            }

        }
            u -= base;  /* adjust from 1-based to 0-based */
            v -= base;

	raw_edges.insert({{u,v}, d});
    }
	i=0;
	for(auto& e : raw_edges) {
	    int u = e.first.first, v = e.first.second;
	    (*i_idx)[i] = u;  
            (*j_idx)[i] = v;
	    (*a)[i] = e.second;
		++i;
	    if(u != v && raw_edges.count({v,u}) == 0) {
		 (*i_idx)[*nz+k] = v;
                 (*j_idx)[*nz+k] = u;
                 (*a)[*nz+k] = (mm_is_symmetric(matcode)) ? e.second : 0.0;
                k++;
	    }
	}

   
    *nz += k;

    fclose(f);

    coo2csr_in (*m, *nz, *a, i_idx, *j_idx);
}

void sort(int *col_idx, double *a, int start, int end)
{
    int i, j, it;
    double dt;

    for (i=end-1; i>start; i--)
        for(j=start; j<i; j++)
            if (col_idx[j] > col_idx[j+1]){

                if (a){
                    dt=a[j];
                    a[j]=a[j+1];
                    a[j+1]=dt;
                }
                it=col_idx[j];
                col_idx[j]=col_idx[j+1];
                col_idx[j+1]=it;

            }
}



/* converts COO format to CSR format, in-place,
   if SORT_IN_ROW is defined, each row is sorted in column index.
   On return, i_idx contains row_start position */

void coo2csr_in(int n, int nz, double *a, int **i_idx, int *j_idx)
{
    int *row_start;
    int i, j;
    int init, i_next, j_next, i_pos;
    double dt, a_next;

    row_start = (int *)my_malloc((n+1)*sizeof(int));
    if (!row_start){
        printf ("coo2csr_in: cannot allocate temporary memory\n");
        exit (1);
    }
    for (i=0; i<=n; i++) row_start[i] = 0;

    /* determine row lengths */
    for (i=0; i<nz; i++) row_start[(*i_idx)[i]+1]++;

    for (i=0; i<n; i++) row_start[i+1] += row_start[i];

    for (init=0; init<nz; ){
        dt = a[init];
        i = (*i_idx)[init];
        j = j_idx[init];
        (*i_idx)[init] = -1;
        while (1){
            i_pos = row_start[i];
            a_next = a[i_pos];
            i_next = (*i_idx)[i_pos];
            j_next = j_idx[i_pos];

            a[i_pos] = dt;
            j_idx[i_pos] = j;
            (*i_idx)[i_pos] = -1;
            row_start[i]++;
            if (i_next < 0) break;
            dt = a_next;
            i = i_next;
            j = j_next;

        }
        init++;
        while ((init < nz) && ((*i_idx)[init] < 0))  init++;
    }


    /* shift back row_start */
    for (i=0; i<n; i++) (*i_idx)[i+1] = row_start[i];
    (*i_idx)[0] = 0;


    for (i=0; i<n; i++){
        sort (j_idx, a, (*i_idx)[i], (*i_idx)[i+1]);
    }

    /* copy i_idx back to row_start, free old data, switch pointers */
    for (i=0; i<n+1; i++) row_start[i] = (*i_idx)[i];
    my_free(*i_idx);
    *i_idx = row_start;
}

