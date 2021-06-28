#include "../include/acc.h"
Acc::Acc(const int l2_, const int l1_, const int pe_t, const int pec_sz):l2(l2_),\
                                                                        pe_total(pe_t),\
                                                                        pec_size(pec_sz){
        pec_total = (int)pe_t/pec_sz;
        l1 = l1_*1024/pec_total; // KB
        l1_support = (int)l1_*1024*1024/pec_total/4; // word
        l2_support = (int)l2_*1024*1024/4; // word
        check_pe_total();
        for(int i=0; i<pec_total; i++){
            pec_arr.emplace_back(pec_sz,l1_);
        }
}
void Acc::check_pe_total(){
    if(pe_total%pec_size != 0 )
        LOG(FATAL)<<"The given #PEs cannot be divided by #PE_SIZE";
}
