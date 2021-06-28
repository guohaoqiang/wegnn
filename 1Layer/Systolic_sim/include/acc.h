// ******************************************************
// Author       : Haoqiang Guo
// Last modified: 2021-04-10 13:36
// Email        : ghaoqi1@lsu.edu
// Filename     : acc.h
// Description  : 
// ******************************************************
#ifndef _ACC_H_
#define _ACC_H_ 
#include "pec.h"
#include <glog/logging.h>
class Acc{
    public:
        //l2, l1, total pes, #pes per cluster
        Acc(const int, const int, const int, const int); 
        void check_pe_total();
    //protected:
        int l1_support;
        int l2_support;
        int pe_total;
        int pec_total;
        int pec_size;
        int l2;
        int l1;
        //const float pro_latency = 0.00000000214; //in the unit of (unit:s)
        //const float pro_energy = 0.0000000000037;
        
        const float pro_latency = 0.000000002; //in the unit of (unit:s)
        const float pro_energy = 0.0000000000585336;
        
        
        const float l1_read_latency = 0.000000004419;  //L1 read latency (unit:s)
        const float l1_write_latency = 0.000000010865;  // L1 write latency (unit:s)
        
        const float in_pec_com_latency = l1_write_latency; //Asuming identiocal to pro_latency
        
        const float l1_read_energy = 0.000000000416;  //L1 read energy (unit:J)
        const float l1_write_energy = 0.000000001224;  // L1 write energy (unit:J)
        
        const float in_pec_com_energy = l1_write_energy; //Asuming identiocal to pro_latency
        
        const int l1_width = 1; // in the unit of 4-Bytes
        const float l2_read = 0.000000004902;  //L2 read latency
        const float l2_write =0.000000011068 ;  // L2 write latency
        const float l2_read_energy = 0.000000000487;  //L2 read energy
        const float l2_write_energy = 0.000000001302;  // L2 write energy
        const int l2_load_store_width = 1; // in the unit of 4-Bytes
        const float dram_energy = 0.00000000326;
        const float dram_latency = 0.0000001044;
        const int dram_width = 1; // == l2_load_store_width
        const float l1_leakage_power = 0.600882;
        const float l2_leakage_power = 1.195658;
        
        std::vector<Pec> pec_arr;
    public:
        std::vector<unsigned long> l2_read_2dram;  // == dram_write
        std::vector<unsigned long> l2_write_from_dram; // == dram_read
        std::vector<unsigned long> dram_read;
        std::vector<unsigned long> dram_write;

};
#endif /* _ACC_H_ */
