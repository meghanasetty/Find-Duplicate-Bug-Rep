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
using namespace std;
double cosine_similarity(int *A, int *B, int size)
{
    double mul = 0.0, d_a = 0.0, d_b = 0.0 ;

    
    for(unsigned int i = 0; i < size; ++i) 
    {
        mul += A[i] * B[i] ;
        d_a += A[i] * A[i] ;
        d_b += B[i] * B[i] ;
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
