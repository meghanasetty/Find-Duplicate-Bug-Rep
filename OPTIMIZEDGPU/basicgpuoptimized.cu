#include<chrono>
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<string>
#include<sstream>
#include<fstream>
#include<vector>
#include<malloc.h>
#define LENGTH_DICTIONARY 57664
#define LENGTH_DOCS 10000
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
using namespace std;
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void cosine_similarity(int *sparsemat, double *cosinematrix, int dicsize,int docsize)
{

    double mul = 0.0, d_a = 0.0, d_b = 0.0 ;
    int doc1index = blockIdx.x;
    int doc2index = threadIdx.x;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(doc1index == doc2index)
	cosinematrix[index] = -1;
    else {
    int *A = &sparsemat[doc1index*dicsize];
    int *B = &sparsemat[doc2index*dicsize] ;
    for(unsigned int i = 0; i < dicsize; ++i) 
    {
        mul += A[i] * B[i] ;
        d_a += A[i] * A[i] ;
        d_b += B[i] * B[i] ;
    }
    cosinematrix[index] =  mul / (sqrt(d_a) * sqrt(d_b)) ;
    }
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

void feedTheMatrix(vector<string> tokens, int * mat)
{
	for(int i=0;i<LENGTH_DICTIONARY;i++)
	{
		mat[i] = 0;
		//cout<<i<<" "<<LENGTH_DICTIONARY<<endl;
	}
	for(int i=1;i<tokens.size();i++)
	{
		mat[stoi(tokens[i])] +=1;
	}
}

void printTheMatrix(int *mat,int row,int col)
{
	for(int i=0;i<row;i++) {
		for(int j=0;j<col;j++)
			cout<<mat[(i*LENGTH_DICTIONARY)+j]<<" ";
		cout<<endl;
	}
}
void printTheCosineMatrix(double *mat,int row,int col)
{
	for(int i=0;i<row;i++) {
		for(int j=0;j<col;j++)
			cout<<mat[(i*LENGTH_DOCS)+j]<<" ";
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
	int *sparsemat = NULL;
	cudaMallocHost((void**)&sparsemat,LENGTH_DOCS*LENGTH_DICTIONARY*sizeof(int));
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
		feedTheMatrix(tokens,&(sparsemat[linenum*LENGTH_DICTIONARY]));
		linenum++;
	}
	inFile.close();
	//printTheMatrix(sparsemat,LENGTH_DOCS,LENGTH_DICTIONARY);
	//create a docs*docs matrix
	double *cosinematrix=NULL;
	cudaMallocHost((void**)&cosinematrix,LENGTH_DOCS*LENGTH_DOCS*sizeof(double));

	//creating 8 streams for 2 GPUS with 4 streams each
	int *d_sparsemat1,*d_sparsemat2;
	double *d_cosinematrix1,*d_cosinematrix2,*d_cosinematrix3,*d_cosinematrix4,*d_cosinematrix5,*d_cosinematrix6,*d_cosinematrix7,*d_cosinematrix8;
	//Both the GPU devices will have all the data in them. Only the computation is done parallely.
	int chunkSize = LENGTH_DOCS/8;
	gpuErrchk( cudaMalloc((void **)&d_sparsemat1,LENGTH_DOCS*LENGTH_DICTIONARY*sizeof(int)));
	gpuErrchk( cudaMalloc((void **)&d_sparsemat2,LENGTH_DOCS*LENGTH_DICTIONARY*sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&d_cosinematrix1,chunkSize*LENGTH_DOCS*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&d_cosinematrix2,chunkSize*LENGTH_DOCS*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&d_cosinematrix3,chunkSize*LENGTH_DOCS*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&d_cosinematrix4,chunkSize*LENGTH_DOCS*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&d_cosinematrix5,chunkSize*LENGTH_DOCS*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&d_cosinematrix6,chunkSize*LENGTH_DOCS*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&d_cosinematrix7,chunkSize*LENGTH_DOCS*sizeof(double)));
	gpuErrchk(cudaMalloc((void **)&d_cosinematrix8,chunkSize*LENGTH_DOCS*sizeof(double)));
	cudaStream_t stream1,stream2,stream3,stream4,stream5,stream6,stream7,stream8;
	cudaSetDevice(0);
	gpuErrchk(cudaStreamCreate(&stream1));
	gpuErrchk(cudaStreamCreate(&stream2));
	gpuErrchk(cudaStreamCreate(&stream3));
	gpuErrchk(cudaStreamCreate(&stream4));
	gpuErrchk(cudaMemcpy(d_sparsemat1,sparsemat,LENGTH_DOCS*LENGTH_DICTIONARY*sizeof(int),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(d_cosinematrix1,&cosinematrix[0*chunkSize*LENGTH_DOCS],chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream1));
	gpuErrchk(cudaMemcpyAsync(d_cosinematrix2,&cosinematrix[1*chunkSize*LENGTH_DOCS],chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream2));
	gpuErrchk(cudaMemcpyAsync(d_cosinematrix3,&cosinematrix[2*chunkSize*LENGTH_DOCS],chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream3));
	gpuErrchk(cudaMemcpyAsync(d_cosinematrix4,&cosinematrix[3*chunkSize*LENGTH_DOCS],chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream4));
	gpuErrchk(cudaStreamSynchronize(stream1));
	gpuErrchk(cudaStreamSynchronize(stream2));
	gpuErrchk(cudaStreamSynchronize(stream3));
	gpuErrchk(cudaStreamSynchronize(stream4));

	cudaSetDevice(1);
	gpuErrchk(cudaStreamCreate(&stream5));
	gpuErrchk(cudaStreamCreate(&stream6));
	gpuErrchk(cudaStreamCreate(&stream7));
	gpuErrchk(cudaStreamCreate(&stream8));
	gpuErrchk(cudaMemcpy(d_sparsemat2,sparsemat,LENGTH_DOCS*LENGTH_DICTIONARY*sizeof(int),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(d_cosinematrix5,&cosinematrix[4*chunkSize*LENGTH_DOCS],chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream5));
	gpuErrchk(cudaMemcpyAsync(d_cosinematrix6,&cosinematrix[5*chunkSize*LENGTH_DOCS],chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream6));
	gpuErrchk(cudaMemcpyAsync(d_cosinematrix7,&cosinematrix[6*chunkSize*LENGTH_DOCS],chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream7));
	gpuErrchk(cudaMemcpyAsync(d_cosinematrix8,&cosinematrix[7*chunkSize*LENGTH_DOCS],chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream8));
	gpuErrchk(cudaStreamSynchronize(stream5));
	gpuErrchk(cudaStreamSynchronize(stream6));
	gpuErrchk(cudaStreamSynchronize(stream7));
	gpuErrchk(cudaStreamSynchronize(stream8));
	chrono::time_point<chrono::system_clock> start = chrono::system_clock::now();
	cudaSetDevice(0);	
	cosine_similarity<<<chunkSize,LENGTH_DOCS,0,stream1>>>(d_sparsemat1,d_cosinematrix1,LENGTH_DICTIONARY,LENGTH_DOCS);
	cosine_similarity<<<chunkSize,LENGTH_DOCS,0,stream2>>>(d_sparsemat1,d_cosinematrix2,LENGTH_DICTIONARY,LENGTH_DOCS);
	cosine_similarity<<<chunkSize,LENGTH_DOCS,0,stream3>>>(d_sparsemat1,d_cosinematrix3,LENGTH_DICTIONARY,LENGTH_DOCS);
	cosine_similarity<<<chunkSize,LENGTH_DOCS,0,stream4>>>(d_sparsemat1,d_cosinematrix4,LENGTH_DICTIONARY,LENGTH_DOCS);
	gpuErrchk(cudaMemcpyAsync(&cosinematrix[0*chunkSize*LENGTH_DOCS],d_cosinematrix1,chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream1));
	gpuErrchk(cudaMemcpyAsync(&cosinematrix[1*chunkSize*LENGTH_DOCS],d_cosinematrix2,chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream2));
	gpuErrchk(cudaMemcpyAsync(&cosinematrix[2*chunkSize*LENGTH_DOCS],d_cosinematrix3,chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream3));
	gpuErrchk(cudaMemcpyAsync(&cosinematrix[3*chunkSize*LENGTH_DOCS],d_cosinematrix4,chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream4));

	cudaSetDevice(1);
	cosine_similarity<<<chunkSize,LENGTH_DOCS,0,stream5>>>(d_sparsemat2,d_cosinematrix5,LENGTH_DICTIONARY,LENGTH_DOCS);
	cosine_similarity<<<chunkSize,LENGTH_DOCS,0,stream6>>>(d_sparsemat2,d_cosinematrix6,LENGTH_DICTIONARY,LENGTH_DOCS);
	cosine_similarity<<<chunkSize,LENGTH_DOCS,0,stream7>>>(d_sparsemat2,d_cosinematrix7,LENGTH_DICTIONARY,LENGTH_DOCS);
	cosine_similarity<<<chunkSize,LENGTH_DOCS,0,stream8>>>(d_sparsemat2,d_cosinematrix8,LENGTH_DICTIONARY,LENGTH_DOCS);
	gpuErrchk(cudaMemcpyAsync(&cosinematrix[4*chunkSize*LENGTH_DOCS],d_cosinematrix5,chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream5));
	gpuErrchk(cudaMemcpyAsync(&cosinematrix[5*chunkSize*LENGTH_DOCS],d_cosinematrix6,chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream6));
	gpuErrchk(cudaMemcpyAsync(&cosinematrix[6*chunkSize*LENGTH_DOCS],d_cosinematrix7,chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream7));
	gpuErrchk(cudaMemcpyAsync(&cosinematrix[7*chunkSize*LENGTH_DOCS],d_cosinematrix8,chunkSize*LENGTH_DOCS*sizeof(double),cudaMemcpyHostToDevice,stream8));
	cudaSetDevice(0);
	gpuErrchk(cudaStreamSynchronize(stream1));
	gpuErrchk(cudaStreamSynchronize(stream2));
	gpuErrchk(cudaStreamSynchronize(stream3));
	gpuErrchk(cudaStreamSynchronize(stream4));
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);
	
	cudaSetDevice(1);
	gpuErrchk(cudaStreamSynchronize(stream5));
	gpuErrchk(cudaStreamSynchronize(stream6));
	gpuErrchk(cudaStreamSynchronize(stream7));
	gpuErrchk(cudaStreamSynchronize(stream8));
	cudaStreamDestroy(stream5);
	cudaStreamDestroy(stream6);
	cudaStreamDestroy(stream7);
	cudaStreamDestroy(stream8);

/*
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
*/
	chrono::time_point<chrono::system_clock> end = chrono::system_clock::now();	
	chrono::duration<double> elapsed_sec = end - start;
	double count_sec = elapsed_sec.count();
	gpuErrchk( cudaDeviceSynchronize());
	cout<<"Time(cosine_similarity_calculations/sec): "<<(LENGTH_DOCS*LENGTH_DOCS)/count_sec<<endl;	
	//printTheCosineMatrix(cosinematrix,LENGTH_DOCS,LENGTH_DOCS);
	//sort the matrix
	for(int i=0;i<LENGTH_DOCS;i++)
	{
		int similardoc = findIndexofHighestSimilarity(&cosinematrix[i*LENGTH_DOCS]);
		cout<<"doc "<<i<<" is similart to doc "<<similardoc<<endl;
		
	}
	free(sparsemat);
        free(cosinematrix);
	cudaFree(d_sparsemat1);
	cudaFree(d_sparsemat2);
	cudaFree(d_cosinematrix1);
	cudaFree(d_cosinematrix2);
	cudaFree(d_cosinematrix3);
	cudaFree(d_cosinematrix4);
	cudaFree(d_cosinematrix5);
	cudaFree(d_cosinematrix6);
	cudaFree(d_cosinematrix7);
	cudaFree(d_cosinematrix8);
}
