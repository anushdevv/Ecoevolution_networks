//  Evograph
//
//  Created by Yang Ping Kuo on 5/22/19.
//  Copyright © 2019 Yang Ping Kuo. All rights reserved.
//

#include <iostream>
#include "evo0_Bd.h"
#include <random>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    if (argc < 8) {
        cout << "Not enough arguments!" << endl;
        exit(1);
    }
			
	int runs = atoi(argv[3]);
    Simulator sim(runs, argv[1], argv[2]);
    
	int max_gen = atoi(argv[4]);
    for(int i = 5; i < argc; i+=3) {
		double a1 = atof(argv[i]);
		double a2 = atof(argv[i + 1]);
        double s2 = atof(argv[i + 2]);
        
        sim.simulate(runs, max_gen, s2, a1, a2);
        sim.save(runs);
    }  
    
    return 1;
}
