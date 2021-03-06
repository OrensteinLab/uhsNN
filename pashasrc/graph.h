/**
    Pasha: Parallel Algorithms for Approximating Compact Universal Hitting Sets
    graph.h
    Header file for graph-related attributes, as well as operations prior to the
    hitting set calculations.
    @author Baris Ekim
    @version 1.0 4/15/19
*/
#ifndef GRAPH_H
#define GRAPH_H
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <iomanip>
#include <cstdint>
#include <omp.h>
using namespace std;
using byte = uint8_t;
using unsigned_int = uint64_t;

struct record
{
    unsigned_int index;
    std::string kmer;
    float pred;
};


class graph {

	public:
    bool* finished;
    bool* pick;
    bool* used;
    bool* tried;
    double delta;
	double epsilon;
    double* hittingNumAnyArray;
    double* hittingNumArray;
    double* maxHittingNum;
    long double** D;
    long double** F;
    int ALPHABET_SIZE;
    double edgeCount;
    double edgeNum;
    int k;
    int l;
    int h;
    int tries;
    int total;
    int vertexCount; 
    int vertexExp;
    int vertexExp2;
    int vertexExp3;
    int vertexExpMask;
    int vertexExp_1;
    byte* edgeArray;
    byte* imaxHittingNum;
    byte* stageArray;
    byte* topoSort;
    static map<char, int> alphabetMap;
    string ALPHABET;
    vector<int> stageVertices;

	void generateGraph(int k) {  
    /**
    Generates a complete de Bruijn graph of order k.
    @param k: Desired k-mer length (order of complete graph).
    */
        for (int i = 0; i < edgeNum; i++) edgeArray[i] = 1;
		edgeCount = edgeNum;
		vertexCount = edgeNum / ALPHABET_SIZE; 
    }

    char getChar(int i) {
    /**
    Gets alphabet character from index.
    @param i: Index of character.
    @return The character in the alphabet.
    */
    	return ALPHABET[i];
    }

    string getLabel(int i) {
    /**
    Gets label of the input edge index.
    @param i: Index of edge.
    @return The label of the edge.
    */
		string finalString = "";
		for (int j = 0; j < k; j++) {
			finalString = getChar((i % ALPHABET_SIZE)) + finalString;
			i = i / ALPHABET_SIZE;
		}
		return finalString;
	}

    unsigned_int getIndex(string label) { //ABL
        /**
        Gets index of the input edge label.
        @param i: label of edge.
        @return The index of the edge.
        */
        map<char, unsigned_int> alphabetMap;
        for (unsigned_int i = 0; i < ALPHABET_SIZE; i++) alphabetMap.insert(pair<char,unsigned_int>(ALPHABET[i], i));
        unsigned_int index = 0;
        for (unsigned_int j = 0; j < k; j++) {
            //cout << alphabetMap[label[j]] << endl;
            index += alphabetMap[label[j]] * pow(4, k-j-1);
            //cout << index << endl;
        }
        return index;
    }

	graph(int argK) {
    /**
    Definition of a graph object. Generates a graph of order k, creates an empty
    edge index array, calculates number of edges, builds a character-index map.
    @param argK: Argument passed as k-mer length.
    */
		ALPHABET = "ACGT";
		ALPHABET_SIZE = 4;
		k = argK;
        edgeNum = pow(ALPHABET_SIZE, k);
        edgeArray = new byte[(int)edgeNum];
		generateGraph(k);
		map<char, int> alphabetMap;
		for (int i = 0; i < ALPHABET_SIZE; i++) alphabetMap.insert(pair<char,int>(ALPHABET[i], i));
    }

    void calculateForEach(int i, int L);
    void calculateForEachAny(int i);
    int calculateHittingNumber(int L);
    byte* calculateHittingNumberAny(int x);
    int calculateHittingNumberParallel(int L, bool random, int threads);
    int calculatePaths(int L, int threads);
    int calculatePathsSeq(int L);
    int calculatePathsAny();
    byte* findMaxAny(int x);
    int findLog(double base, double x);
    vector<int> getAdjacent(int v);
    int Hitting(int L, string hittingFile);
    int HittingAny(int L, int x, string hittingFile);
    int HittingParallel(int L, string hittingFile, int threads);
    int HittingRandomParallel(int L, string hittingFile, int threads);
    int maxLength();
    vector<int> pushBackVector();
    void removeEdge(int i);
    int stageOps(int l, double maxPtr, string hittingFile);
    void topologicalSort();
    private:
	int depthFirstSearch(int index, int u);

};
#endif