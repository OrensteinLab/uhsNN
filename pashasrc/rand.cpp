	#include <iostream>
    #include <sys/resource.h> 
    #include <sys/time.h> 
    #include <unistd.h> 
    #include <stdio.h> 
    #include <time.h>
    //#include<bits/stdc++.h>
    #include <sys/stat.h> 
    #include <sys/types.h> 
    #include "decycling.h"
    #include "pasha.h"
    int main(int argc, char* argv[])
    {
        struct timespec start, finish;
        double elapsed;
        size_t pos;
        unsigned_int k = stoi(argv[1], &pos);
        unsigned_int L = stoi(argv[2], &pos);
        unsigned_int threads = stoi(argv[3], &pos);
        const char *decyclingPath = argv[4];
        const char *hittingPath = argv[5];
        double threshold = stod(argv[6]);
        string decyclingFile;
        string hittingFile;
        const double PI = 3.14159;
        string directory;
        ofstream decyclingStream(decyclingPath);
        PASHA pasha = PASHA(k);
        cout << "Graph OK" << endl;
        decycling newDecycling;
        vector<unsigned int> decyclingSet = newDecycling.computeDecyclingSet(k);
        //cout << "Decycling max size: " << decyclingSet.max_size() << "Decycling size: " << decyclingSet.size() << endl;
        for (unsigned_int i = 0; i < decyclingSet.size(); i++) {
            //cout << "Removing decycling edge " << decyclingSet[i] << endl;
            string label = pasha.getLabel(decyclingSet[i]);
            //cout << label << " Index: " << decyclingSet[i] << endl;
            pasha.removeEdge(decyclingSet[i]);
            //cout << label << endl;
            decyclingStream << label << "\n";
        }
        unsigned_int decyclingSize = decyclingSet.size();
        cout << "Decycling set size: " << decyclingSize << endl;
        decyclingStream.close();
        string predsPath = "/data/liorkot/gen/xpreds/preds_" + to_string(k) + "_" + to_string(L) + ".txt";
        string line;
        int preds = 0;
        std::ifstream file(predsPath);
        while (std::getline(file, line)) {
            std::istringstream is(line);
            record r;
            std::string s;
            std::getline(is, r.kmer, '\t' );
            r.index = pasha.getIndex(r.kmer);
            std::getline( is, s, '\t' );
            r.pred = std::stof(s);
            if (r.pred >= threshold && r.pred < 2.0 && pasha.isEdgeinGraph(r.index)) {
                preds++;
                //            decyclingStream << r.kmer << "\n"; CBL
                pasha.removeEdge(r.index);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &start);
        unsigned_int hittingSize = pasha.HittingRandomParallel(L, hittingPath, threads);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        cout << "PASHA(RAW) took " << elapsed << " seconds." << endl;
        cout << "Prediction set size: " << preds << endl;
        cout << "PASHA set size: " << hittingSize << endl << endl;
        cout << "Hitting set size: " << preds + hittingSize + decyclingSize << endl << endl;
    }
