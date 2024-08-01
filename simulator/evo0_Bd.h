//  Evograph
//
//  Created by Yang Ping Kuo on 5/22/19.
//  Copyright © 2019 Yang Ping Kuo. All rights reserved.
//

#pragma once

#include <random>
#include <ctime>
#include <cmath>

#include <iostream>
#include <fstream>
#include <string>

#include <vector>
#include <stdexcept>

using namespace std;

void permute(int *iton, int *ntoi, int index1, int index2)
{
    int node1 = iton[index1];
    int node2 = iton[index2];
    
    iton[index1] = node2;
    iton[index2] = node1;
    
    ntoi[node1] = index2;
    ntoi[node2] = index1;
}

void fitness_t(double *fitness, int *populations, double s2, double a11, double a12, double a21, double a22)
{
    double c1 = 1 / (2 * (populations[0] * a11 + populations[1] * a21));
    double c2 = 1 / (2 * (populations[0] * a12 + populations[1] * a22));
    fitness[0] = a11 * c1 + a12 * c2;
    fitness[1] = (1 + s2) * a21 * c1 + (1 + s2) * a22 * c2;
}

class Simulator {
private:
	int popsize;
	int max_gen;
	int run_idx;
	double s2;
	double a1, a2;
	int *degrees, **edgelist;
	double *freq2;
	int *tfix;
    ofstream file;
	mt19937 generator;

public:
	Simulator(int, string, string);
    ~Simulator();
    void simulate(int, double, double, double);
    void simulate(int, int, double, double, double);
    //void print();
    void save(int);
};

Simulator::Simulator(int trials, string input_name, string output_name) {
    ifstream input(input_name);
    file.open(output_name);
    
    vector<int> out, in;
    int node;
    int i = 0;
    popsize = 0;
    while (input >> node)
    {
        popsize = popsize < node ? node: popsize;
        if (i % 2 == 0)
            out.push_back(node);
        else
            in.push_back(node);
        ++i;
    }
    if (out.size() != in.size())
        throw invalid_argument("in and out should have same length");
    ++popsize;
    generator = mt19937((unsigned int)time(NULL));
    
    degrees = new int[popsize];
    int *temp = new int[popsize];
    
    for (int node = 0; node < popsize; ++node) {
        temp[node] = 0;
        degrees[node] = 0;
    }
    
    for (auto node : out)
        ++degrees[node];
    
    for (auto node : in)
        ++degrees[node];
    
    edgelist = new int*[popsize];
    for (int node = 0; node < popsize; ++node) {
        edgelist[node] = new int[degrees[node]];
    }
    
    for (int i = 0; i < in.size(); ++i) {
        int node1 = in[i];
        int node2 = out[i];
        edgelist[node1][temp[node1]] = node2;
        edgelist[node2][temp[node2]] = node1;
        ++temp[node1];
        ++temp[node2];
    }
	
	freq2 = new double[trials];
	tfix = new int[trials];
		
    delete[] temp;
}

Simulator::~Simulator()
{
    for (int node = 0; node < popsize; ++node) {
        delete [] edgelist[node];
    }
    delete [] edgelist;
    delete [] degrees;
	delete [] freq2;
	delete [] tfix;
    file.close();
}


void Simulator::simulate(int max_gen = 500, double s2 = 0, double a1 = 1.0, double a2 = 1.0)
{
	this->max_gen = max_gen;
    this->s2 = s2;
	this->a1 = a1;
	this->a2 = a2;
		
	double a11 = a1;
	double a12 = 1 - a1;
	double a21 = a2;
	double a22 = 1 - a2;
			
    uniform_real_distribution<double> rand(0.0, 1.0);
    
    double fitness[] = { 0, 0};
    int populations[] = { popsize - 1, 1};
		
    int *ntoi = new int[popsize];
    int *iton = new int[popsize];
    int *mutant = new int[popsize];
    
    for (int i = 0; i < popsize; ++i)
    {
        ntoi[i] = i;
        iton[i] = i;
        mutant[i] = 0;
    }
    	
    int index2 = (int)(rand(generator) * popsize);
    	
    mutant[iton[index2]] = 1;
    permute(iton, ntoi, populations[0], index2);
	
    int t = 0;
    while (populations[0] != popsize && populations[1] != popsize && t < (max_gen * popsize))
    {
		
		++t;
		fitness_t(fitness, populations, s2, a11, a12, a21, a22);
		
        double bar1 = populations[0] * fitness[0];
		double totalFitness = bar1 + populations[1] * fitness[1];
        
        double birth = totalFitness * rand(generator);
        int birthIndex, birthNode, deathIndex, deathNode;
       
        if (birth > bar1)
        {
            birthIndex = populations[0] + (int)((birth - bar1) / fitness[1]);
            birthNode = iton[birthIndex];
        }
        else
        {
            birthIndex = (int)(birth / fitness[0]);
            birthNode = iton[birthIndex];
        }
			
        deathIndex = (int)(degrees[birthNode] * rand(generator));
        deathNode = edgelist[birthNode][deathIndex];
		
        if (mutant[deathNode] == mutant[birthNode])
            continue;
		
        if (mutant[birthNode] == 0 && mutant[deathNode] == 1)
        {
            index2 = populations[0];
            permute(iton, ntoi, ntoi[deathNode], index2);
			++populations[0];
        }
        else
        {
            index2 = populations[0] - 1;
            permute(iton, ntoi, ntoi[deathNode], index2);
			++populations[1];
        }
		
        if (mutant[deathNode] == 0)
		{
            --populations[0];
		}
        else
		{
            --populations[1];
		}
        
        mutant[deathNode] = mutant[birthNode];
		
    }
	
	this->freq2[run_idx] = populations[1];
	this->tfix[run_idx] = t;
	this->run_idx += 1;
	
    delete[] mutant;
    delete[] ntoi;
    delete[] iton;
}

// simulate birth-death processes for input trial number of times
void Simulator::simulate(int trials, int max_gen = 500, double s2 = 0.0, double a1 = 1.0, double a2 = 1.0)
{
    generator = mt19937((unsigned int)time(NULL));
	this->run_idx = 0;

    for (int i = 0; i < trials; ++i)
    {
		simulate(max_gen, s2, a1, a2);
    }

}

void Simulator::save(int trials)
{
	file << "$" << "\t";
	        file << s2 << "\t";
	        file << a1 << "\t";
	        file << a2 << endl;
	    for (int i = 0; i < trials; ++i)
	    {
	        file << freq2[i] << endl;
	    }
	    for (int i = 0; i < trials; ++i)
	    {
	        file << tfix[i] << endl;
	    }
}
