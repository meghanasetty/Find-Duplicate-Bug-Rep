#include<chrono>
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<string>
#include<sstream>
#include<fstream>
#include<vector>
#include<malloc.h>
#include<omp.h>
#include<immintrin.h>
#define LENGTH_DICTIONARY 57664
#define LENGTH_DOCS 10000
using namespace std;
double cosine_similarity(int *A, int *B, int size)
{
    double mul = 0.0, d_a = 0.0, d_b = 0.0 ;

    //int *mul1,*mul2,*mul3,*mul4,*mul5,*mul6,*mul7,*mul8;
    //int *d_a1,*d_a2,*d_a3,*d_a4,*d_a5,*d_a6,*d_a7,*d_a8;
    //int *d_b1,*d_b2,*d_b3,*d_b4,*d_b5,*d_b6,*d_b7,*d_b8;
    __m256i mul1 = _mm256_setzero_si256();
    __m256i d_a1 = _mm256_setzero_si256();
    __m256i d_b1 = _mm256_setzero_si256();

    __m256i mul2 = _mm256_setzero_si256();
    __m256i d_a2 = _mm256_setzero_si256();
    __m256i d_b2 = _mm256_setzero_si256();

    __m256i mul3 = _mm256_setzero_si256();
    __m256i d_a3 = _mm256_setzero_si256();
    __m256i d_b3 = _mm256_setzero_si256();

    __m256i mul4 = _mm256_setzero_si256();
    __m256i d_a4 = _mm256_setzero_si256();
    __m256i d_b4 = _mm256_setzero_si256();

    __m256i mul5 = _mm256_setzero_si256();
    __m256i d_a5 = _mm256_setzero_si256();
    __m256i d_b5 = _mm256_setzero_si256();

    __m256i mul6 = _mm256_setzero_si256();
    __m256i d_a6 = _mm256_setzero_si256();
    __m256i d_b6 = _mm256_setzero_si256();

    __m256i mul7 = _mm256_setzero_si256();
    __m256i d_a7 = _mm256_setzero_si256();
    __m256i d_b7 = _mm256_setzero_si256();

    __m256i mul8 = _mm256_setzero_si256();
    __m256i d_a8 = _mm256_setzero_si256();
    __m256i d_b8 = _mm256_setzero_si256();

    for(unsigned int i = 0; i < size/64; i=i+64) 
    {
	__m256i A_vector1 = _mm256_setr_epi32(A[i],A[i+1],A[i+2],A[i+3],A[i+4],A[i+5],A[i+6],A[i+7]);
	__m256i B_vector1 = _mm256_setr_epi32(B[i],B[i+1],B[i+2],B[i+3],B[i+4],B[i+5],B[i+6],B[i+7]);
	__m256i ab1 = _mm256_mul_epi32(A_vector1,B_vector1);
	__m256i aa1 = _mm256_mul_epi32(A_vector1,A_vector1);
	__m256i bb1 = _mm256_mul_epi32(B_vector1,B_vector1);
        mul1 = _mm256_add_epi32(mul1,ab1);
        d_a1 = _mm256_add_epi32(d_a1,aa1);
        d_b1 = _mm256_add_epi32(d_b1,bb1);

	__m256i A_vector2 = _mm256_setr_epi32(A[i+8],A[i+9],A[i+10],A[i+11],A[i+12],A[i+13],A[i+14],A[i+15]);
	__m256i B_vector2 = _mm256_setr_epi32(B[i+8],B[i+9],B[i+10],B[i+11],B[i+12],B[i+13],B[i+14],B[i+15]);
	__m256i ab2 = _mm256_mul_epi32(A_vector2,B_vector2);
	__m256i aa2 = _mm256_mul_epi32(A_vector2,A_vector2);
	__m256i bb2 = _mm256_mul_epi32(B_vector2,B_vector2);
        mul2 = _mm256_add_epi32(mul2,ab2);
        d_a2 = _mm256_add_epi32(d_a2,aa2);
        d_b2 = _mm256_add_epi32(d_b2,bb2);

	__m256i A_vector3 = _mm256_setr_epi32(A[i+16],A[i+17],A[i+18],A[i+19],A[i+20],A[i+21],A[i+22],A[i+23]);
	__m256i B_vector3 = _mm256_setr_epi32(B[i+16],B[i+17],B[i+18],B[i+19],B[i+20],B[i+21],B[i+22],B[i+23]);
	__m256i ab3 = _mm256_mul_epi32(A_vector3,B_vector3);
	__m256i aa3 = _mm256_mul_epi32(A_vector3,A_vector3);
	__m256i bb3 = _mm256_mul_epi32(B_vector3,B_vector3);
        mul3 = _mm256_add_epi32(mul3,ab3);
        d_a3 = _mm256_add_epi32(d_a3,aa3);
        d_b3 = _mm256_add_epi32(d_b3,bb3);

	__m256i A_vector4 = _mm256_setr_epi32(A[i+24],A[i+25],A[i+26],A[i+27],A[i+28],A[i+29],A[i+30],A[i+31]);
	__m256i B_vector4 = _mm256_setr_epi32(B[i+24],B[i+25],B[i+26],B[i+27],B[i+28],B[i+29],B[i+30],B[i+31]);
	__m256i ab4 = _mm256_mul_epi32(A_vector4,B_vector4);
	__m256i aa4 = _mm256_mul_epi32(A_vector4,A_vector4);
	__m256i bb4 = _mm256_mul_epi32(B_vector4,B_vector4);
        mul4 = _mm256_add_epi32(mul4,ab4);
        d_a4 = _mm256_add_epi32(d_a4,aa4);
        d_b4 = _mm256_add_epi32(d_b4,bb4);

	__m256i A_vector5 = _mm256_setr_epi32(A[i+32],A[i+33],A[i+34],A[i+35],A[i+36],A[i+37],A[i+38],A[i+39]);
	__m256i B_vector5 = _mm256_setr_epi32(B[i+32],B[i+33],B[i+34],B[i+35],B[i+36],B[i+37],B[i+38],B[i+39]);
	__m256i ab5 = _mm256_mul_epi32(A_vector5,B_vector5);
	__m256i aa5 = _mm256_mul_epi32(A_vector5,A_vector5);
	__m256i bb5 = _mm256_mul_epi32(B_vector5,B_vector5);
        mul5 = _mm256_add_epi32(mul5,ab5);
        d_a5 = _mm256_add_epi32(d_a5,aa5);
        d_b5 = _mm256_add_epi32(d_b5,bb5);

	__m256i A_vector6 = _mm256_setr_epi32(A[i+40],A[i+41],A[i+42],A[i+43],A[i+44],A[i+45],A[i+46],A[i+47]);
	__m256i B_vector6 = _mm256_setr_epi32(B[i+40],B[i+41],B[i+42],B[i+43],B[i+44],B[i+45],B[i+46],B[i+47]);
	__m256i ab6 = _mm256_mul_epi32(A_vector6,B_vector6);
	__m256i aa6 = _mm256_mul_epi32(A_vector6,A_vector6);
	__m256i bb6 = _mm256_mul_epi32(B_vector6,B_vector6);
        mul6 = _mm256_add_epi32(mul6,ab6);
        d_a6 = _mm256_add_epi32(d_a6,aa6);
        d_b6 = _mm256_add_epi32(d_b6,bb6);

	__m256i A_vector7 = _mm256_setr_epi32(A[i+48],A[i+49],A[i+50],A[i+51],A[i+52],A[i+53],A[i+54],A[i+55]);
	__m256i B_vector7 = _mm256_setr_epi32(B[i+48],B[i+49],B[i+50],B[i+51],B[i+52],B[i+53],B[i+54],B[i+55]);
	__m256i ab7 = _mm256_mul_epi32(A_vector7,B_vector7);
	__m256i aa7 = _mm256_mul_epi32(A_vector7,A_vector7);
	__m256i bb7 = _mm256_mul_epi32(B_vector7,B_vector7);
        mul7 = _mm256_add_epi32(mul7,ab7);
        d_a7 = _mm256_add_epi32(d_a7,aa7);
        d_b7 = _mm256_add_epi32(d_b7,bb7);

	__m256i A_vector8 = _mm256_setr_epi32(A[i+56],A[i+57],A[i+58],A[i+59],A[i+60],A[i+61],A[i+62],A[i+63]);
	__m256i B_vector8 = _mm256_setr_epi32(B[i+56],B[i+57],B[i+58],B[i+59],B[i+60],B[i+61],B[i+62],B[i+63]);
	__m256i ab8 = _mm256_mul_epi32(A_vector8,B_vector8);
	__m256i aa8 = _mm256_mul_epi32(A_vector8,A_vector8);
	__m256i bb8 = _mm256_mul_epi32(B_vector8,B_vector8);
        mul8 = _mm256_add_epi32(mul8,ab8);
        d_a8 = _mm256_add_epi32(d_a8,aa8);
        d_b8 = _mm256_add_epi32(d_b8,bb8);

    }
    int *result_mul1 = (int *)&mul1;
    int *result_da1 = (int *)&d_a1;
    int *result_db1 = (int *)&d_b1;

    int *result_mul2 = (int *)&mul2;
    int *result_da2 = (int *)&d_a2;
    int *result_db2 = (int *)&d_b2;

    int *result_mul3 = (int *)&mul3;
    int *result_da3 = (int *)&d_a3;
    int *result_db3 = (int *)&d_b3;

    int *result_mul4 = (int *)&mul4;
    int *result_da4 = (int *)&d_a4;
    int *result_db4 = (int *)&d_b4;

    int *result_mul5 = (int *)&mul5;
    int *result_da5 = (int *)&d_a5;
    int *result_db5 = (int *)&d_b5;

    int *result_mul6 = (int *)&mul6;
    int *result_da6 = (int *)&d_a6;
    int *result_db6 = (int *)&d_b6;

    int *result_mul7 = (int *)&mul7;
    int *result_da7 = (int *)&d_a7;
    int *result_db7 = (int *)&d_b7;

    int *result_mul8 = (int *)&mul8;
    int *result_da8 = (int *)&d_a8;
    int *result_db8 = (int *)&d_b8;

    for(int i=0;i<8;i++)
    {
	mul +=result_mul1[i]+result_mul2[i]+result_mul3[i]+result_mul4[i]+result_mul5[i]+result_mul6[i]+result_mul7[i]+result_mul8[i];
	d_a +=result_da1[i]+result_da2[i]+result_da3[i]+result_da4[i]+result_da5[i]+result_da6[i]+result_da7[i]+result_da8[i];
	d_b +=result_db1[i]+result_db2[i]+result_db3[i]+result_db4[i]+result_db5[i]+result_db6[i]+result_db7[i]+result_db8[i];
    }
    return mul / (sqrt(d_a) * sqrt(d_b)) ;
}

void getTokens(string line,vector<string>&tokens)
{
	istringstream tokenStream(line);
	string token;
	while (getline(tokenStream, token,' '))
   	{
      		tokens.push_back(token);
   	}
}
void printVector(vector<string> v)
{
	cout<<"size: "<<v.size()<<endl;
	for(int i=0;i<v.size();i++)
	{
		cout<<v[i]<<" ";
	}
	cout<<endl;
}

void feedTheMatrix(vector<string> tokens, int ** mat)
{
	for(int i=0;i<LENGTH_DICTIONARY;i++)
	{
		(*mat)[i] = 0;
		//cout<<i<<" "<<LENGTH_DICTIONARY<<endl;
	}
	for(int i=1;i<tokens.size();i++)
	{
		(*mat)[stoi(tokens[i])] +=1;
	}
}

void printTheMatrix(int **mat,int row,int col)
{
	for(int i=0;i<row;i++) {
		for(int j=0;j<col;j++)
			cout<<mat[i][j]<<" ";
		cout<<endl;
	}
}
void printTheCosineMatrix(double **mat,int row,int col)
{
	for(int i=0;i<row;i++) {
		for(int j=0;j<col;j++)
			cout<<mat[i][j]<<" ";
		cout<<endl;
	}
}
int findIndexofHighestSimilarity(double *cosinematrix)
{
	int index = 0;
	double maxvalue = -1;
	for(int i=0;i<LENGTH_DOCS;i++)
	{
		if(cosinematrix[i] > maxvalue)
		{
			maxvalue = cosinematrix[i];
			index = i;
		}
	}
	return index;
}
int main()
{
	int **sparsemat;
	sparsemat = (int **)malloc(LENGTH_DOCS*sizeof(int *));
	for(int i=0;i<LENGTH_DOCS;i++)
		sparsemat[i] = (int *)malloc(LENGTH_DICTIONARY*sizeof(int));
	//get the contents of file
	ifstream inFile;
	inFile.open("./sample10000.txt");
	if (!inFile) {
		cerr << "Unable to open file sample100.txt";
    	//	exit(1);   // call system to stop
		return -1;
	}
	string line;
	int linenum = 0;
	while (getline(inFile,line)) {
  		//cout<<line<<endl;
		vector<string> tokens;
		getTokens(line,tokens);
		//cout<<linenum<<" "<<LENGTH_DOCS<<endl;
		//printVector(tokens);
		feedTheMatrix(tokens,&sparsemat[linenum]);
		linenum++;
	}
	inFile.close();
	//printTheMatrix(sparsemat,LENGTH_DOCS,LENGTH_DICTIONARY);
	//create a docs*docs matrix
	double **cosinematrix;
	cosinematrix = (double **)malloc(LENGTH_DOCS*sizeof(double *));
	for(int i=0;i<LENGTH_DOCS;i++)
		cosinematrix[i] = (double *)malloc(LENGTH_DOCS*sizeof(double));
	chrono::time_point<chrono::system_clock> start = chrono::system_clock::now();
#pragma omp  parallel for collapse(2)
	for(int i=0;i<LENGTH_DOCS;i++)
	{
		for(int j =0;j<LENGTH_DOCS;j++)
		{
			if(i==j)
			{
				//obviously same docs have highest similarity so equating them to -1
				cosinematrix[i][j] = -1;
			}
			else
			{
				cosinematrix[i][j] = cosine_similarity(sparsemat[i],sparsemat[j],LENGTH_DICTIONARY);
			}
		}
	}
	chrono::time_point<chrono::system_clock> end = chrono::system_clock::now();	
	chrono::duration<double> elapsed_sec = end - start;
	double count_sec = elapsed_sec.count();
	cout<<"Time(cosine_similarity_calculations/sec): "<<(LENGTH_DOCS*LENGTH_DOCS)/count_sec<<endl;	
	//printTheCosineMatrix(cosinematrix,LENGTH_DOCS,LENGTH_DOCS);
	//sort the matrix
	for(int i=0;i<LENGTH_DOCS;i++)
	{
		int similardoc = findIndexofHighestSimilarity(cosinematrix[i]);
		cout<<"doc "<<i<<" is similart to doc "<<similardoc<<endl;
		
	}
	for(int i=0;i<LENGTH_DOCS;i++)
		free(sparsemat[i]);
	free(sparsemat);
	for(int i=0;i<LENGTH_DOCS;i++)
                free(cosinematrix[i]);
        free(cosinematrix);

}
