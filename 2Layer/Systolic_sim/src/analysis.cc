#include "../include/analysis.h"

Analysis::Analysis(std::shared_ptr<Acc> accelerator,\
        std::shared_ptr<Graph> input_data):\
        hw(accelerator),dat(input_data){
}  

void Analysis::print(){
    for(auto pec:hw->pec_arr){
       if(pec.opt_counts.size()!=pec.opt_delay.size() || \
               pec.opt_counts.size()!=pec.com_delay.size() ||\
               pec.opt_counts.size()!=pec.l1_read.size() ||\
               pec.opt_counts.size()!=pec.l1_write.size())
           LOG(FATAL)<<"Un..."<<std::endl;
       std::cout<<std::endl;
       std::cout<<"-----------------------"<<std::endl;
       //LOG(INFO)<<"total_delay: "<<pec.total_delay.size()<<std::endl;
       pec.print();
    }
}
void Analysis::run_wegnn(){
    
    VLOG(0)<<"Accelerator: ";
    VLOG(0)<<"Total PEs: "<<hw->pe_total;
    VLOG(0)<<"PE clusters: "<<hw->pec_total;
    VLOG(0)<<"#PEs in each PE cluster: "<<hw->pec_size;
    VLOG(0)<<"L1 in each PE cluster: "<<hw->l1<<" KB";
    VLOG(0)<<"L2 on chip: "<<hw->l2<<" MB";
    //VLOG(0)<<"Dataset: "<<dat->name<<"\-"<<dat->nodes<<","<<dat->max_num<<","<<dat->sorted_data.at(0).at(1)-dat->sorted_data.at(0).at(0);
    VLOG(0)<<"Dataset: "<<dat->name<<"\-"<<dat->nodes<<","<<dat->feature_size<<","<<dat->hiden;

    double minimum_delay = DBL_MAX;
    double minimum_edp = DBL_MAX;
    double minimum_energy = DBL_MAX;
    int tm_tag; 
    int tk_tag; 
    int tn_tag; 
    int tc_tag; 
    long update_flag = 0;
    int total_pec = hw->pec_total;
    //for(int tm=1;tm<dat->sorted_data.at(0).size()-1;++tm){
    for(int tm=1; tm<dat->sorted_data.at(0).size()-1; ++tm){
        for(int tn=1;tn<dat->feature_size;tn++){
            hw->pec_total = total_pec;
            if(tm*tn>hw->l1_support) break;

            row_divider = 1;
            while(!(tn*hw->pec_total <= row_divider*dat->get_feature_size())){
                row_divider += 1;
                if(row_divider>=hw->pec_total || row_divider>=dat->sorted_data.at(0).at(1)-dat->sorted_data.at(0).at(0)){
                    break;
                }
            }

            //Either tn OR pec_total is too large, and row_divider gets the maximum value
            if(tn*hw->pec_total > row_divider*dat->get_feature_size()){
                hw->pec_total = dat->get_feature_size()*row_divider/tn;
            }
            
            
            int eff_pec = hw->pec_total/row_divider; 
            if(tm<eff_pec) break;
            

            float uti = (float)hw->pec_total/total_pec;
            PCHECK(tn*hw->pec_total <= row_divider*dat->get_feature_size())<<"row_divider failure";
            //PCHECK(hw->pec_total%row_divider==0)<<"row_divider failure";
            //PCHECK(row_divider<=dat->hiden)<<"row_divider is too large";
            

            int flag = 0; 

            for(int tk=dat->sorted_data.at(0).at(1)-dat->sorted_data.at(0).at(0); tk>0; --tk){
                if(tm*tk*tn<row_divider*hw->pec_size) break;
                
                if(tm*tk + tk*tn > hw->l1_support || \
                    tm*tk+eff_pec*tk*tn > hw->l2_support){
                    continue;
                }
                
                if(flag == 1){
                    flag = 0;
                    goto LOOP_N; 
                }
                flag = 1; 
                for(int tc=1;tc<dat->hiden;++tc){
                    VLOG(2)<<"tm = "<<tm<<" tc = "<<tc; 
                    if(tm*tn+tn*tc>hw->l1_support || \
                        tm*tc>hw->l1_support || \
                        eff_pec*tm*tc>hw->l2_support) goto FINAL_END;
                    height = dat->sorted_data.at(0).size()-1;
                    len = ceil((float)height/tm);
                    VLOG(0)<<"tm = "<<tm<<", tk = "<<tk<<", tn = "<<tn<<", tc = "<<tc\
                        <<", row_divider = "<<row_divider<<", uti = "<<uti;
                    run_w_order_wo_move(tm,tk,tn,tc);     
                    if(global_delay<minimum_delay){
                        std::cout<<"tm="<<tm<<" tk="<<tk<<" tn="<<tn<<" tc="<<tc<<\
                            " minimum_delay = "<<minimum_delay<<\
                            " minimum_edp = "<<minimum_edp<<\
                            " minimum_energy = "<<minimum_energy<<\
                            " uti = "<<uti<<std::endl;
                        tm_tag = tm;tk_tag = tk;tn_tag = tn;tc_tag = tc;
                        minimum_delay = global_delay;
                        minimum_edp = global_edp;
                        minimum_energy = global_energy;
                        update_flag += 1;
                        VLOG(0)<<"tm="<<tm<<" tk="<<tk<<" tn="<<tn<<" tc="<<tc<<\
                            " minimum_delay = "<<minimum_delay<<\
                            " minimum_edp = "<<minimum_edp<<\
                            " minimum_energy = "<<minimum_energy<<\
                            " uti = "<<uti;
                        print_final("wegnn"); 
                  //      VLOG(0)<<"delay0="<<onchip_delay0<<" delay1="<<onchip_delay1<<" delay2="<<onchip_delay2<<
                    }
                    
                }
            }
            LOOP_N:
                VLOG(2)<<"continue";
        }            
        VLOG(1)<<"tm = "<<tm<<" flag = "<<update_flag;
    }
    FINAL_END:
        global_delay = minimum_delay;
        global_edp = minimum_edp;
        global_energy = minimum_energy;
    // to be identical with paper ("tn" and "tk")
    //std::cout<<"tm = "<<tm_tag<<"  tn = "<<tk_tag<<"  tk = "<<tn_tag<<"  tc = "<<tc_tag<<std::endl;
}
void Analysis::run_baseline(){
    
    VLOG(0)<<"Accelerator: ";
    VLOG(0)<<"Total PEs: "<<hw->pe_total;
    VLOG(0)<<"PE clusters: "<<hw->pec_total;
    VLOG(0)<<"#PEs in each PE cluster: "<<hw->pec_size;
    VLOG(0)<<"L1 in each PE cluster: "<<hw->l1<<" KB";
    VLOG(0)<<"L2 on chip: "<<hw->l2<<" MB";
    //VLOG(0)<<"Dataset: "<<dat->name<<"\-"<<dat->nodes<<","<<dat->max_num;
    VLOG(0)<<"Dataset: "<<dat->name<<"\-"<<dat->nodes<<","<<dat->feature_size<<","<<dat->hiden;

    double minimum_delay = DBL_MAX;
    double minimum_edp = DBL_MAX;
    double minimum_energy = DBL_MAX;
    int tm_tag = 0; 
    int tk_tag = 0; 
    int tn_tag = 0; 
    int tc_tag = 0; 
    long update_flag = 0;
    //iterate row_index of X first
    int total_pec = hw->pec_total;
    for(int tk=1;tk<dat->nodes;++tk){ //dat->max_num refers the maximum degrees of one node. Note here A has been condensed
        for(int tc=1;tc<dat->hiden;tc++){
            hw->pec_total = total_pec;
            VLOG(2)<<"tk = "<<tk<<",tc = "<<tc;
            // I tile fits in L1
            if(tk*tc>hw->l1_support) break;
            //VLOG(0)<<"tk = "<<tk<<",tc = "<<tc;
            
            //distribute colums of W
            row_divider = 1;
            while(!(tc*hw->pec_total <= row_divider*dat->hiden)){
                row_divider += 1;
                //VLOG(0)<<"row_divider = "<<row_divider;
                if(row_divider>=hw->pec_total || row_divider>=dat->feature_size){
                    break;
                }
            }
            VLOG(2)<<"#pec = "<<hw->pec_total<<\
                ",row_divider = "<<row_divider<<\
                ",tc*hw->pec_total = "<<tc*hw->pec_total<<\
                ", row_divider*dat->hiden = "<<row_divider*dat->hiden;
            //Either tc OR pec_total is large, and row_divider equals maximum value
            if(tc*hw->pec_total > row_divider*dat->hiden){
                hw->pec_total = dat->hiden*row_divider/tc;
            }
            
            
            //if((row_divider == hw->pec_total || row_divider==dat->feature_size) && \
                   tc*hw->pec_total > row_divider*dat->hiden) break;
            
            //if(hw->pec_total%row_divider!=0){
            float uti = (float)hw->pec_total/total_pec;
                //hw->pec_total = hw->pec_total - hw->pec_total%row_divider;
            //}
            VLOG(2)<<"tk = "<<tk<<",tc = "<<tc;
            VLOG(2)<<"#pec = "<<hw->pec_total<<\
                ",row_divider = "<<row_divider<<\
                ",tc*hw->pec_total = "<<tc*hw->pec_total<<\
                ", row_divider*dat->hiden = "<<row_divider*dat->hiden;
            PCHECK(tc*hw->pec_total <= row_divider*dat->hiden)<<"row_divider failure";
            //PCHECK(hw->pec_total%row_divider==0)<<"row_divider failure";
            int eff_pec = hw->pec_total/row_divider; 
            

            int flag = 0; 
            for(int tn=dat->feature_size;tn>0;--tn){
                VLOG(2)<<", tk = "<<tk<<", tn = "<<tn<<", tc = "<<tc\
                        <<", row_divider = "<<row_divider;
                if(tn%row_divider!=0) continue;
                if(tk*tn*tc<row_divider*hw->pec_size){
                    VLOG(2)<<"tn out";
                    goto LOOP_C;
                }
                //a row-tile of X fits in L1 && #eff_pec W colum-tile fits in L2
                if(tk*tn + tn*tc >hw->l1_support || \
                    tk*tn+eff_pec*tn*tc>hw->l2_support){
                    continue;
                }
                // a smaller tn leads to a low arithmetic density (#op/#IO) 
                // reduce exploration space dramatically
                if(flag == 1){
                    flag = 0;
                    VLOG(0)<<"flag_out";
                    goto LOOP_C; 
                }
                flag = 1; 
                
                //for(int tm=1;tm<dat->csr_diag.at(0).size()-1;++tm){
                for(int tm=row_divider; tm<dat->csr_diag.at(0).size()-1; tm+=row_divider){
            //        VLOG(0)<<"tm = "<<tm<<", tk = "<<tk<<", tn = "<<tn<<", tc = "<<tc\
                        <<", row_divider = "<<row_divider;
                    //column-tile of A fits in L1. O tile fits int L1. O tiles fit in L2
                    if(tm*tk + tk*tc>hw->l1_support || \
                            tm*tc>hw->l1_support || \
                        eff_pec*tm*tc>hw->l2_support) goto FINAL_END;
                    
                    height = dat->csr_diag.at(0).size()-1;
                    len = ceil((float)height/tk); // determined by #row_tiles of X
                    VLOG(0)<<"tm = "<<tm<<", tk = "<<tk<<", tn = "<<tn<<", tc = "<<tc\
                        <<", row_divider = "<<row_divider<<",uti = "<<uti;
                    run_baseline_awbgcn_gcnax(tm,tk,tn,tc);  // note that "tk" and "tm"   
                    if(global_delay<minimum_delay){
                        tm_tag = tk;tk_tag = tm;tn_tag = tn;tc_tag = tc;
                        minimum_delay = global_delay;
                        minimum_edp = global_edp;
                        minimum_energy = global_energy;
                        update_flag += 1;
                        // note that "tk" and "tm"
                        std::cout<<"tm="<<tk<<" tk="<<tm<<" tn="<<tn<<" tc="<<tc<<\
                            " minimum_delay = "<<minimum_delay<<\
                            " minimum_edp = "<<minimum_edp<<\
                            " minimum_energy = "<<minimum_energy<<"  uti = "<<uti<<std::endl;
                        VLOG(0)<<"tm="<<tk<<" tk="<<tm<<" tn="<<tn<<" tc="<<tc<<\
                            " minimum_delay = "<<minimum_delay<<\
                            " minimum_edp = "<<minimum_edp<<\
                            " minimum_energy = "<<minimum_energy<<"  uti = "<<uti;
                        print_final("Baseline"); 
                    }
                    
                }
            }
            LOOP_C:
                VLOG(2)<<"continue";
        }            
        //VLOG(1)<<"tm = "<<tk<<" flag = "<<flag;
    }
    FINAL_END:
        global_delay = minimum_delay;
        global_edp = minimum_edp;
        global_energy = minimum_energy;
    // to be identical with paper ("tm" and "tk")
    //std::cout<<"tm = "<<tk_tag<<"  tn = "<<tn_tag<<"  tk = "<<tm_tag<<"  tc = "<<tc_tag<<std::endl;
}
void Analysis::run_w_order_wo_move(int tm,int tk,int tn,int tc){
    //if(tm%10==0 && tk%30==0 && tn%50==0 && tc==1)
    //    VLOG(0)<<"(wegnn)tm = "<<tm<<",tk = "<<tk<<",tn = "<<tn<<",tc = "<<tc;
    //L1:
    //spmm: tm*tk \times tk*tn = tm*tn -----> tn*(tm-tk)<=alpha*tm*tk
    //dense: tm*tn \times tn*tc = tm*tc ------> 
    //L2:
    //sparse: tm*tk  (shared across PECs)
    //dense: hw->pec_total*(tk*tn), hw->pec_total*(tn*tc+tm*tc)
    global_delay = 0;
    global_edp = 0;
    global_energy = 0;
    onchip_delay = 0;
    offchip_delay = 0;
    pec_read_delay = 0;  // pec read from L2
    pec_write_delay = 0; // pec write to L2
    for(auto pec=hw->pec_arr.begin();pec!=hw->pec_arr.end();++pec){
        (*pec).opt_counts.clear();
        (*pec).opt_delay.clear();
        (*pec).com_delay.clear();
        (*pec).total_delay.clear();
        (*pec).l1_read.clear();
        (*pec).l1_write.clear();
        (*pec).l2_read_2pe.clear();
        (*pec).l2_write_from_pe.clear();
    }
    hw->l2_read_2dram.clear();
    hw->l2_write_from_dram.clear();
    hw->dram_read.clear();
    hw->dram_write.clear();
    tiles->clear();
    tiles_r->clear();
    //fetch
    for(int i=0; i<height; i+=tm){ //iterate over row-tiles of A
        int up_1 = 0;
        int up_2 = 0;
        double l1_write_temp = 0;
        double l1_write_temp0 = 0;
        long opt_counts_temp = 0; 
        long opt_counts_temp0 = 0; 
        long l1_read_temp = 0;
        long l1_read_temp1 = 0;
        long l1_read_temp2 = 0;
        double delay = 0;
        long cols_in_row_tiles = 0;
        long rows_in_row_tiles = 0;
        long entrys_in_row_tiles = 0;
        int row_in_a_tile = 0;
        std::shared_ptr<std::set<int>> unique_col = std::make_shared<std::set<int>>();
        int max_col_in_a_tile = 0; // l2_2_pe, load X
        for(int j=0; j<dat->sorted_data.at(0).at(i+1)-dat->sorted_data.at(0).at(i); j+=tk){  //iterate over tiles of a specific row-tile of A
            //VLOG(1)<<"i = "<<i<<" j =  "<<j<<std::endl;
            //std::shared_ptr<std::vector<int>> temp(new std::vector<int>());
            std::shared_ptr<std::vector<int>> temp = std::make_shared<std::vector<int>>();
            std::shared_ptr<std::vector<int>> temp_r = std::make_shared<std::vector<int>>();
            row_in_a_tile = 0;
            //fetch a specific A tile of the current row-tile
            for(int tm_i=i; tm_i<std::min(i+tm,height); ++tm_i){
                row_in_a_tile += 1;
                VLOG(2)<<"tm_i = "<<tm_i<<std::endl;
                int width = std::min(j+tk,dat->sorted_data.at(0).at(tm_i+1)-dat->sorted_data.at(0).at(tm_i));
                VLOG(2)<<"width = "<<width<<std::endl;
                int offset = *(dat->sorted_data.at(0).begin()+tm_i);
                VLOG(2)<<"offset = "<<offset<<std::endl;
                int cols_in_a_tile = 0;
                for(int tk_j=j; tk_j<width; ++tk_j){
                    cols_in_a_tile += 1;
                    temp->push_back(dat->sorted_data.at(1).at(offset+tk_j));
                    unique_col->insert(dat->sorted_data.at(1).at(offset+tk_j));
                    temp_r->push_back(tm_i);
                    VLOG(2)<<"Push r: "<<tm_i<<std::endl;
                    VLOG(2)<<"Push c: "<<dat->sorted_data.at(1).at(offset+tk_j)<<std::endl;
                }
                //if(cols_in_a_tile > max_col_in_a_tile)
                //    max_col_in_a_tile = cols_in_a_tile;
            }
            //cols_in_row_tiles += max_col_in_a_tile; 
            //till the last tile of the row-tile,gets the value
            max_col_in_a_tile = unique_col->size(); //l1_read,l2_2_l1, read X, needs *tn later
            rows_in_row_tiles += row_in_a_tile; // l1_write: AX=I
            entrys_in_row_tiles += (*temp).size();
            //sum reduction 
            if(max_col_in_a_tile != 0){
                float real_1 = log2((float)max_col_in_a_tile);
                up_1 += ceil(real_1)*row_in_a_tile; //one PEC, One colum, need *tn later
            }
            //pec #
            if(row_in_a_tile != 0){
                //l1_write_temp = row_in_a_tile*tc;
                l1_write_temp0 = row_in_a_tile; //I,l1_write, one PEC, One column, needs *tn later
            }
            //int opt_counts_temp = (*temp).size()*tn + tm*tn*tc; // scatter has no impact on #ops of the last item
            // feature size can be divided by tn*#pec
            
            tiles->push_back(temp);
            tiles_r->push_back(temp_r);
            //pec_id = (++pec_id)%(hw->pec_total); 
            VLOG(2)<<"Complete one tile! "<<std::endl;
        }// end a row-tile of A
        
        opt_counts_temp0 = entrys_in_row_tiles; // One PEC, One column, needs *tn later
        
        //left (A)
        l1_read_temp1 += ceil((float)entrys_in_row_tiles/row_divider);// One PEC, tn colum 
        //output (I) , right (X)
        l1_read_temp2 += (row_in_a_tile + ceil((float)max_col_in_a_tile/row_divider));// One PEC, One colum, need *tn later 
        
        int eff_pec = hw->pec_total/row_divider;
        int mod1_tmp = dat->get_feature_size()%(tn*eff_pec);
        int mod1 = mod1_tmp/tn; // <hw->pec_total
        int res_last_pec = mod1_tmp - mod1*tn; // <tn
        long up1 = 0; 
        //for(int pec_id=0;pec_id<hw->pec_total;++pec_id){
        for(int pec_id=0;pec_id<eff_pec;++pec_id){
            long l2_2_pe = ceil((float)entrys_in_row_tiles/row_divider); // A row-tile
            if(pec_id<mod1){
                up1 = up_1*tn*ceil((float)dat->get_feature_size()/(tn*eff_pec));
                l1_read_temp = (l1_read_temp1+l1_read_temp2*tn)*ceil((float)dat->get_feature_size()/(tn*eff_pec));
                
                opt_counts_temp = opt_counts_temp0*tn*ceil((float)dat->get_feature_size()/(tn*eff_pec));
                l1_write_temp = ceil((float)max_col_in_a_tile/row_divider/tk)*l1_write_temp0*tn*ceil((float)dat->get_feature_size()/(tn*eff_pec));
                
                l2_2_pe += ceil((float)max_col_in_a_tile/row_divider)*tn*ceil((float)dat->get_feature_size()/(tn*eff_pec));// X
            }else if(pec_id==mod1 && res_last_pec!=0){
                up1 = up_1*tn*floor((float)dat->get_feature_size()/(tn*eff_pec)) + \
                        up_1*res_last_pec;
                l1_read_temp = \
                        (l1_read_temp1+l1_read_temp2*tn)*floor((float)dat->get_feature_size()/(tn*eff_pec)) + \
                        l1_read_temp1+l1_read_temp2*res_last_pec;
                opt_counts_temp = opt_counts_temp0*tn*floor((float)dat->get_feature_size()/(tn*eff_pec)) + \
                                 opt_counts_temp0*res_last_pec;
                l1_write_temp = ceil((float)max_col_in_a_tile/row_divider/tk)*(l1_write_temp0*tn*floor((float)dat->get_feature_size()/(tn*eff_pec)) + \
                                l1_write_temp0*res_last_pec);
                
                l2_2_pe += ceil((float)max_col_in_a_tile/row_divider)*tn*floor((float)dat->get_feature_size()/(tn*eff_pec)) + \
                           cols_in_row_tiles*res_last_pec;;
            
            }else{
                up1 = up_1*tn*floor((float)dat->get_feature_size()/(tn*eff_pec));
                l1_read_temp = (l1_read_temp1+l1_read_temp2*tn)*floor((float)dat->get_feature_size()/(tn*eff_pec));
                opt_counts_temp = opt_counts_temp0*tn*floor((float)dat->get_feature_size()/(tn*eff_pec));
                l1_write_temp = ceil((float)max_col_in_a_tile/row_divider/tk)*(l1_write_temp0*tn*floor((float)dat->get_feature_size()/(tn*eff_pec)));
                
                l2_2_pe += ceil((float)max_col_in_a_tile/row_divider)*tn*floor((float)dat->get_feature_size()/(tn*eff_pec));
            }
            //step1: accumulate partial sums of a output tile (I)
            double sub_com0 = l1_write_temp;//inter-PEC exchange
            double sub_accu0 = l1_write_temp;//accumulation // #additions 
            //scatter
            double sub_com = ceil(l1_write_temp*(eff_pec-1)/eff_pec) + sub_com0;//inter-PEC communication 
            hw->pec_arr.at(pec_id).set_com_delay(sub_com);//One PEC
            
            //IW;
            float real_2 = log2(dat->get_feature_size());
            //here indicates requirements : tm%hw->pec_total == 0 
            up_2 = ceil(row_in_a_tile/hw->pec_total)*ceil(real_2)*dat->hiden; // One PEC
            l1_read_temp += ceil((float)row_in_a_tile/eff_pec)*dat->hiden +\
                            ceil((float)row_in_a_tile/eff_pec)*dat->get_feature_size() + \
                            ceil((float)dat->get_feature_size()*dat->hiden/row_divider);// One PEC //output + I + W 
            l1_write_temp += ceil(row_in_a_tile/eff_pec)*tc*ceil(dat->hiden/tc)/row_divider; // one PEC
            
            opt_counts_temp += (sub_accu0 + \
                    ceil((float)row_in_a_tile/eff_pec)*dat->get_feature_size()*ceil((float)dat->hiden/row_divider)); //One PEC, scatter has no impact on #ops
            int opt_delay_temp = ceil((float)opt_counts_temp/hw->pec_size);
            hw->pec_arr.at(pec_id).set_opt_counts(opt_counts_temp);//One PEC 
            hw->pec_arr.at(pec_id).set_opt_delay(opt_delay_temp);//One PEC 
           
            double delay = opt_delay_temp*hw->pro_latency\
                    + sub_com*hw->in_pec_com_latency\
                    + l1_read_temp*hw->l1_read_latency\
                    + l1_write_temp*hw->l1_write_latency;
            
            
            //Only right W
            l2_2_pe += ceil((float)dat->get_feature_size()*dat->hiden/row_divider);
            hw->pec_arr.at(pec_id).l2_read_2pe.push_back(l2_2_pe); //One PEC
            
            //here row_in_a_tile >= eff_pec
            long pe_2_l2 = ceil((float)ceil((float)row_in_a_tile/eff_pec)*dat->hiden/row_divider);
            hw->pec_arr.at(pec_id).l2_write_from_pe.push_back(pe_2_l2); //One PEC
            
            hw->pec_arr.at(pec_id).set_l1_read(l1_read_temp);//One PEC 
            hw->pec_arr.at(pec_id).set_l1_write(l1_write_temp);//One PEC
            hw->pec_arr.at(pec_id).set_total_delay(delay);//One PEC 
            
        }// end pec_id
        hw->dram_read.push_back(entrys_in_row_tiles+max_col_in_a_tile*dat->get_feature_size()\
                              + dat->get_feature_size()*dat->hiden); // A row tile of A + X + W
        hw->l2_write_from_dram.push_back(entrys_in_row_tiles+max_col_in_a_tile*dat->get_feature_size()\
                              + dat->get_feature_size()*dat->hiden);
        
        
        hw->dram_write.push_back(row_in_a_tile*dat->hiden); // A row tile of final output 
        hw->l2_read_2dram.push_back(row_in_a_tile*dat->hiden); 
    
    }
    VLOG(1)<<"All tiles have been feched!"<<std::endl; 
    long count = 0;
    for(auto tile:(*tiles)){
       count += tile->size();
    }
    
    VLOG(1)<<"Now extract the global results...";
    extract_timing();
    VLOG(1)<<"Extraction has completed!";
    VLOG(2)<<"#tile: "<<tiles_r->size();
    VLOG(2)<<"#entry: "<<count;
    //print_final("wegnn");
}
//Since the involved sparse matrix is symmetric sparse, so CSR and CSC is the same.
void Analysis::run_baseline_awbgcn_gcnax(int tm,int tk,int tn,int tc){
    //if(tm%10==0 && tk%30==0 && tn%50==0 && tc==1)
    //if(tc==1)
    VLOG(2)<<"(gcnax)tm = "<<tm<<",tk = "<<tk<<",tn = "<<tn<<",tc = "<<tc;
    //L1:
    //spmm: tm*tk \times tk*tn = tm*tn -----> tn*(tm-tk)<=alpha*tm*tk
    //dense: tm*tn \times tn*tc = tm*tc ------> 
    //L2:
    //sparse: tm*tk  (shared across PECs)
    //dense: hw->pec_total*(tk*tn), hw->pec_total*(tn*tc+tm*tc)

    //height = dat->csr_diag.at(0).size()-1;
    //len = ceil(height/tk);
    
    global_delay = 0;
    global_edp = 0;
    global_energy = 0;
    onchip_delay = 0;
    offchip_delay = 0;
    pec_read_delay = 0;  // pec read from L2
    pec_write_delay = 0; // pec write to L2
    for(auto pec=hw->pec_arr.begin();pec!=hw->pec_arr.end();++pec){
        (*pec).opt_counts.clear();
        (*pec).opt_delay.clear();
        (*pec).com_delay.clear();
        (*pec).total_delay.clear();
        (*pec).l1_read.clear();
        (*pec).l1_write.clear();
        (*pec).l2_read_2pe.clear();
        (*pec).l2_write_from_pe.clear();
    }
    VLOG(2)<<"(gcnax)tm = "<<tm<<",tk = "<<tk<<",tn = "<<tn<<",tc = "<<tc;
    hw->l2_read_2dram.clear();
    hw->l2_write_from_dram.clear();
    hw->dram_read.clear();
    hw->dram_write.clear();
    tiles->clear();
    tiles_c->clear();
    //fetch
    //int height = dat->csr_diag.at(0).size()-1;
    auto pec_id = 0;
    //XW: row tile (X) shared, so round_trips_1
    //AI: column tile (A) shared, so round_trips_2
    //"tm" and "tk" have the reverse meaning to wegnn
    //Perform XW first, height here equals #nodes
    //int tc_counts = dat->hiden/(hw->pec_total*tc);
    int eff_pec = hw->pec_total/row_divider;
    VLOG(2)<<"(gcnax)eff_pec = "<<eff_pec;
    int tc_counts = dat->hiden/(eff_pec*tc);
    VLOG(2)<<"(gcnax)tc_counts = "<<tc_counts<<",eff_pec = "<<eff_pec;
    std::set<int> tile_row_id;
    //Since the involved sparse matrix (A) is symmetric sparse, so CSR and CSC is the same.
    for(int x_row=0; x_row<height; x_row+=tk){ // iterate ove row-tile of X
        //std::shared_ptr<std::vector<int>> x_row_tile = std::make_shared<std::vector<int>>();
        int up_2 = 0;
        long l1_read_xw = 0;
        long l1_write_xw = 0;
        long xw_ops = 0;
        long l2_2_pe = 0;
        VLOG(2)<<"";
        for(size_t x_r=x_row; x_r<std::min(x_row+tk,height); ++x_r){ // A specific row-tile of X
            tile_row_id.insert(x_r);
            l1_read_xw += dat->get_feature_size();//read X from l1, needs *(# I tiles)
            float real_2 = log2((float)dat->get_feature_size());
            up_2 += ceil(real_2); //One PEC, One column, needs *tc later 
            xw_ops += dat->get_feature_size(); //One PEC, One column. needs *tc later
            l2_2_pe += dat->get_feature_size(); //One PEC, read X from L2
        }
        VLOG(2)<<"";
        //fetch A row-tile by row-tile (identical to column-tile because A is symmetric)
        long ops_ai = 0;
        int rows_in_a_tile = 0; //l1_write
        int max_row_in_a_tile = 0;// l1_write,pe_2_l2,l2_2_dram
        long entrys_in_column_tile = 0;
        long l1_read_a = 0;
        long l1_read_i = 0;
        int col_in_a_tile = 0;
        std::shared_ptr<std::set<int>> unique_rows = std::make_shared<std::set<int>>();
        for(int i=0; i<height; i+=tm){ //iterate over row tiles of A
            bool cflag = 1;
            int row_length = 0;
            int ct = i;
            while(ct<std::min(i+tm,height)){
                if(row_length<dat->csr_diag.at(0).at(ct+1)-dat->csr_diag.at(0).at(ct))
                    row_length = dat->csr_diag.at(0).at(ct+1)-dat->csr_diag.at(0).at(ct);
                ct++;
            }
            //fetch one row-tile of A (identical to column-tile because A is symmetric)
            unique_rows->clear();
            for(int j=0; j<row_length; j+=tk){ //iterate over tiles of a specific row-tile 
                //VLOG(1)<<"i = "<<i<<" j =  "<<j<<std::endl;
                //std::shared_ptr<std::vector<int>> temp(new std::vector<int>());
                std::shared_ptr<std::vector<int>> temp = std::make_shared<std::vector<int>>();
                std::shared_ptr<std::vector<int>> temp_c = std::make_shared<std::vector<int>>();
                //fetch one tile of a row-tile
                bool rflag = 1;
                for(int tm_i=i; tm_i<std::min(i+tm,height); ++tm_i){
                    // only colum_ID is identical to the completed row_ID of XW tiles, ...\
                    // ... perform the MatMul
                    if(tile_row_id.find(tm_i) != tile_row_id.end()){
                        VLOG(2)<<"tile_row_id.size() = "<<tile_row_id.size();
                        if(cflag == 1){  // One column-tile of A, once 
                            col_in_a_tile += 1; // determines sum reduction: up_1, needs to *max_row_in_a_tile later
                            rows_in_a_tile = dat->csr_diag.at(0).at(tm_i+1)-dat->csr_diag.at(0).at(tm_i);
                            //if(rows_in_a_tile > max_row_in_a_tile)
                            //    max_row_in_a_tile = rows_in_a_tile; //One PEC, l1_write, pe_2_l2,l2_2_dram
                            ops_ai += rows_in_a_tile; //One PEC, needs to *tc later
                            l1_read_a += rows_in_a_tile; //One PEC, needs to *(#I_tiles per PEC) later
                            entrys_in_column_tile = ops_ai; // total entrys in the corresponding column-tile of A
                            VLOG(2)<<"rows_in_a_tile = "<<rows_in_a_tile;
                            VLOG(2)<<"max_row_in_a_tile = "<<max_row_in_a_tile;
                        }
                        if(rflag == 1){ // ensure one tile of A, load once I tile
                            l1_read_i += tile_row_id.size();// One PEC.  needs to *tc later 
                            rflag = 0;
                        }
                        VLOG(2)<<"tm_i = "<<tm_i<<std::endl;
                        int width = std::min(j+tk,dat->csr_diag.at(0).at(tm_i+1)-dat->csr_diag.at(0).at(tm_i));
                        VLOG(2)<<"width = "<<width<<std::endl;
                        int offset = *(dat->csr_diag.at(0).begin()+tm_i);
                        VLOG(2)<<"offset = "<<offset<<std::endl;
                        int cols_in_a_tile = 0;
                        for(int tk_j=j; tk_j<width; ++tk_j){
                            temp->push_back(dat->csr_diag.at(1).at(offset+tk_j));
                            temp_c->push_back(tm_i);
                            
                            unique_rows->insert(dat->csr_diag.at(1).at(offset+tk_j));
                            
                            VLOG(2)<<"Push r: "<<tm_i<<std::endl;
                            VLOG(2)<<"Push c: "<<dat->csr_diag.at(1).at(offset+tk_j)<<std::endl;
                        }
                        //till the last tile of the row-tile,...
                        //... max_row_in_a_tile gets the required value
                        max_row_in_a_tile = unique_rows->size();
                    } // end if 
                    VLOG(2)<<"Complete one tile! "<<std::endl;
                }// end fetch one tile of a row-tile
                if(rflag == 0){
                    tiles->push_back(temp);
                    tiles_c->push_back(temp_c);
                }
                cflag = 0;
            } // end fetch one row-tile of A
        }// end fetch A
        
        VLOG(2)<<"";
        VLOG(2)<<"max_row_in_a_tile = "<<max_row_in_a_tile;
        
        //sum reduction timing in AI
        int up_1 = 0;
        float real_1 = log2((float)col_in_a_tile); // refer row#: 410
        up_1 = ceil(real_1)*max_row_in_a_tile; //One PEC, needs to *tc later
        float l1_write_temp = 0;
        l1_write_temp = max_row_in_a_tile; // refer row#521, write final, //One PEC, needs to *tc later
            
        long opt_counts_temp = xw_ops + ops_ai; // refer row#453 & row#494,  //One PEC, needs to *tc later
        long l1_read_temp = 0; //l1_read_a & l1_read_i; // refer row#495 & row#501
        long up = 0;
        //for(int pec_id=0;pec_id<hw->pec_total;++pec_id){
        for(int pec_id=0;pec_id<eff_pec;++pec_id){
            long reduce_l1_write_temp = 0;
            long i_l1_write_temp = 0;
            long pe_2_l2 = 0;
            PCHECK(tc_counts>0)<<"error for divider";
            if(tc_counts>0){// read W from l1
               //int mod0_tmp = dat->hiden%(hw->pec_total*tc);
               int mod0_tmp = dat->hiden%(eff_pec*tc);
               int mod0 = mod0_tmp/tc;
               int res_last_pec1 = mod0_tmp-mod0*tc;
               if(pec_id<mod0){
                    VLOG(2)<<"pec_id = "<<pec_id<<" pec_id<mod0";
                    l1_read_temp = ceil((float)l1_read_xw/row_divider) + tile_row_id.size()*(tc_counts+1)*tc+\
                            ceil((float)(tc_counts*tc*dat->get_feature_size() + tc*dat->get_feature_size())/row_divider) + \
                            ceil((float)l1_read_a*(tc_counts+1)/row_divider) + l1_read_i*(tc_counts+1)*tc;//output + right (W)
                    up = (up_1+up_2)*(tc_counts+1)*tc;
                    opt_counts_temp = ceil((float)opt_counts_temp*(tc_counts+1)*tc/row_divider);
                    // I
                    i_l1_write_temp = (tile_row_id.size())*(tc_counts+1)*tc;//write (I) to l1
                    l1_write_temp = ceil((float)dat->get_feature_size()/row_divider/tn)*(tile_row_id.size())*(tc_counts+1)*tc;//write (I) to l1
                    l2_2_pe += ceil((float)(tc_counts*tc*dat->get_feature_size() + \
                            tc*dat->get_feature_size() + l1_read_a)/row_divider); //right (W) + A
                    pe_2_l2 = ceil((float)2*max_row_in_a_tile*(tc_counts+1)*tc/row_divider); // partial sums + final results, they share identical size
                    //final sum reduction
                    opt_counts_temp += ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
                    l1_read_temp += ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
                    reduce_l1_write_temp = ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);//write (partial+I) to l1
                    l1_write_temp += ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);//write partial to l1
                    l2_2_pe += ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
               }else if(pec_id==mod0){
                    VLOG(2)<<"pec_id = "<<pec_id<<" pec_id=mod0";
                    l1_read_temp = ceil((float)l1_read_xw/row_divider) + tile_row_id.size()*(tc_counts*tc+res_last_pec1)+\
                            ceil((float)(tc_counts*tc*dat->get_feature_size() + res_last_pec1*dat->get_feature_size())/row_divider)+\ 
                            ceil((float)l1_read_a*(tc_counts+1)/row_divider) + l1_read_i*(tc_counts+1)*tc;
                    up = (up_1+up_2)*(tc_counts*tc+res_last_pec1);
                    opt_counts_temp = ceil((float)opt_counts_temp*(tc_counts*tc+res_last_pec1)/row_divider);
                    i_l1_write_temp = (tile_row_id.size())*(tc_counts*tc+res_last_pec1);//write I to l1
                    l1_write_temp = ceil((float)dat->get_feature_size()/row_divider/tn)*(tile_row_id.size())*(tc_counts*tc+res_last_pec1);//write I to l1
                    l2_2_pe += ceil((float)(tc_counts*tc*dat->get_feature_size() + \
                            res_last_pec1*dat->get_feature_size()+ l1_read_a)/row_divider); 
                    pe_2_l2 = ceil((float)2*max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
                    //final sum reduction
                    opt_counts_temp += ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
                    l1_read_temp += ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
                    reduce_l1_write_temp = ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
                    l1_write_temp += ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
                    l2_2_pe += ceil((float)max_row_in_a_tile*(tc_counts+1)*tc/row_divider);
               }else{
                    VLOG(2)<<"pec_id = "<<pec_id<<" pec_id>mod0";
                    l1_read_temp = ceil((float)l1_read_xw/row_divider) + tile_row_id.size()*tc_counts*tc+\
                            ceil((float)tc_counts*tc*dat->get_feature_size()/row_divider) +\ 
                            ceil((float)l1_read_a*tc_counts/row_divider) + l1_read_i*tc_counts*tc;
                    up = (up_1+up_2)*tc_counts*tc;
                    opt_counts_temp = ceil((float)opt_counts_temp*tc_counts*tc/row_divider);
                    i_l1_write_temp = (tile_row_id.size())*tc_counts*tc;//write I to l1
                    l1_write_temp = ceil((float)dat->get_feature_size()/row_divider/tn)*(tile_row_id.size())*tc_counts*tc;//write I to l1
                    l2_2_pe += ceil((float)(tc_counts*tc*dat->get_feature_size()+ l1_read_a)/row_divider);
                    pe_2_l2 = ceil((float)2*max_row_in_a_tile*tc_counts*tc/row_divider);
                    //final sum reduction
                    opt_counts_temp += ceil((float)max_row_in_a_tile*tc_counts*tc/row_divider);
                    l1_read_temp += ceil((float)max_row_in_a_tile*tc_counts*tc/row_divider);
                    reduce_l1_write_temp = ceil((float)max_row_in_a_tile*tc_counts*tc/row_divider);
                    l1_write_temp += ceil((float)max_row_in_a_tile*tc_counts*tc/row_divider);
                    l2_2_pe += ceil((float)max_row_in_a_tile*tc_counts*tc/row_divider);
               }
            }else{ // some PECs are idle
               int mod0 = dat->hiden/tc;
               int res_last_pec2 = dat->hiden - mod0*tc;
               if(pec_id<mod0){
                    VLOG(2)<<"pec_id = "<<pec_id<<" tc_counts = "<<tc_counts<<" pec_id<mod0";
                    l1_read_temp = ceil((float)l1_read_xw/row_divider) + tile_row_id.size()*tc+ceil((float)tc*dat->get_feature_size()/row_divider) +\
                                  ceil((float)l1_read_a*1/row_divider) + l1_read_i*1*tc;
                    up = (up_1+up_2)*tc;
                    opt_counts_temp = ceil((float)opt_counts_temp*tc/row_divider);
                    i_l1_write_temp = (tile_row_id.size())*tc;//write I to l1
                    l1_write_temp = ceil((float)dat->get_feature_size()/row_divider/tn)*(tile_row_id.size())*tc;//write I to l1
                    l2_2_pe += ceil((float)(tc*dat->get_feature_size()+ l1_read_a)/row_divider);
                    pe_2_l2 = ceil((float)2*max_row_in_a_tile*1*tc/row_divider);
                    VLOG(2)<<"max_row_in_a_tile = "<<max_row_in_a_tile;
                    //final sum reduction
                    opt_counts_temp += ceil((float)max_row_in_a_tile*1*tc/row_divider);
                    l1_read_temp += ceil((float)max_row_in_a_tile*1*tc/row_divider);
                    reduce_l1_write_temp = ceil((float)max_row_in_a_tile*1*tc/row_divider);
                    l1_write_temp += ceil((float)max_row_in_a_tile*1*tc/row_divider);
                    l2_2_pe += ceil((float)max_row_in_a_tile*1*tc/row_divider);
               }else if(pec_id==mod0){
                    VLOG(2)<<"pec_id = "<<pec_id<<" tc_counts = "<<tc_counts<<" pec_id=mod0";
                    l1_read_temp = ceil((float)l1_read_xw/row_divider) + tile_row_id.size()*res_last_pec2 + \
                            ceil((float)res_last_pec2*dat->get_feature_size()/row_divider) +\ 
                                  ceil((float)l1_read_a*1/row_divider) + l1_read_i*1*tc;
                    up = (up_1+up_2)*res_last_pec2;
                    opt_counts_temp = ceil((float)opt_counts_temp*res_last_pec2/row_divider);
                    i_l1_write_temp = tile_row_id.size()*res_last_pec2;//write I to l1
                    l1_write_temp = ceil((float)dat->get_feature_size()/row_divider/tn)*(tile_row_id.size())*res_last_pec2;//write I to l1
                    l2_2_pe += ceil((float)(res_last_pec2*dat->get_feature_size()+ l1_read_a)/row_divider); 
                    pe_2_l2 = ceil((float)2*max_row_in_a_tile*1*res_last_pec2/row_divider);
                    VLOG(2)<<"max_row_in_a_tile = "<<max_row_in_a_tile;
                    //final sum reduction
                    opt_counts_temp += ceil((float)max_row_in_a_tile*1*res_last_pec2/row_divider);
                    l1_read_temp += ceil((float)max_row_in_a_tile*1*res_last_pec2/row_divider);
                    reduce_l1_write_temp = ceil((float)max_row_in_a_tile*1*res_last_pec2/row_divider);
                    l1_write_temp += ceil((float)max_row_in_a_tile*1*res_last_pec2/row_divider);
                    l2_2_pe += ceil((float)max_row_in_a_tile*1*res_last_pec2/row_divider);
               }
            }
            
            hw->pec_arr.at(pec_id).set_opt_counts(opt_counts_temp); // one A tile leads to #pec ... 
            int opt_delay_temp = ceil((float)opt_counts_temp/hw->pec_arr.at(pec_id).pe_counts);
            hw->pec_arr.at(pec_id).set_opt_delay(opt_delay_temp); 
            
            long sub_com = (row_divider-1)*i_l1_write_temp;//exchange the partial sum tile (I) within #row_divider PEC
            if(row_divider>1){
                hw->pec_arr.at(pec_id).set_com_delay(sub_com);
            }else{
                hw->pec_arr.at(pec_id).set_com_delay(0);
            }
           
            l1_read_temp += sub_com; 
            hw->pec_arr.at(pec_id).set_l1_read(l1_read_temp); 
            
            double delay = opt_delay_temp*hw->pro_latency\
                        + sub_com*hw->in_pec_com_latency\
                        + l1_read_temp*hw->l1_read_latency\
                        + l1_write_temp*hw->l1_write_latency;
            //VLOG(0)<<"pec_id = "<<pec_id<<" total_delay.size() = "<<hw->pec_arr.at(pec_id).total_delay.size();
            hw->pec_arr.at(pec_id).set_total_delay(delay); 
            //VLOG(0)<<"delay = "<<delay;
            
            hw->pec_arr.at(pec_id).set_l1_write(l1_write_temp);
            //hw->pec_arr.at(pec_id).set_com_delay(up);
               
            hw->pec_arr.at(pec_id).l2_write_from_pe.push_back(pe_2_l2);
            hw->pec_arr.at(pec_id).l2_read_2pe.push_back(l2_2_pe);
           
            VLOG(2)<<"pe_2_l2 = "<<pe_2_l2<<" pec_id = "<<pec_id;
        }//end pec id
        VLOG(2)<<"";
        hw->dram_read.push_back(l1_read_xw + dat->get_feature_size()*dat->hiden + \
                entrys_in_column_tile +\
                max_row_in_a_tile*dat->hiden); // X + W + A + partial sums
        hw->l2_write_from_dram.push_back(l1_read_xw + dat->get_feature_size()*dat->hiden + \
                entrys_in_column_tile +\
                max_row_in_a_tile*dat->hiden); // sum reduction
        hw->dram_write.push_back(max_row_in_a_tile*dat->hiden); 
        hw->l2_read_2dram.push_back(max_row_in_a_tile*dat->hiden); 
        
        tile_row_id.clear();
    }
/*
    VLOG(2)<<"Tile fetch...";
    for(int i=0; i<height; i+=tm){ //#row tiles
        int row_length = 0;
        int ct = i;
        while(ct<std::min(i+tm,height)){
            if(row_length<dat->csr_diag.at(0).at(ct+1)-dat->csr_diag.at(0).at(ct))
                row_length = dat->csr_diag.at(0).at(ct+1)-dat->csr_diag.at(0).at(ct);
            ct++;
        }
        for(int j=0; j<row_length; j+=tk){  //#column tiles
            //VLOG(1)<<"i = "<<i<<" j =  "<<j<<std::endl;
            //std::shared_ptr<std::vector<int>> temp(new std::vector<int>());
            std::shared_ptr<std::vector<int>> temp = std::make_shared<std::vector<int>>();
            std::shared_ptr<std::vector<int>> temp_c = std::make_shared<std::vector<int>>();
            int max_row_in_a_tile = 0;
            int col_in_a_tile = 0;
            int rows_in_a_tile = 0;
            for(int tm_i=i; tm_i<std::min(i+tm,height); ++tm_i){
                col_in_a_tile += 1;
                VLOG(2)<<"tm_i = "<<tm_i<<std::endl;
                int width = std::min(j+tk,dat->csr_diag.at(0).at(tm_i+1)-dat->csr_diag.at(0).at(tm_i));
                VLOG(2)<<"width = "<<width<<std::endl;
                int offset = *(dat->csr_diag.at(0).begin()+tm_i);
                VLOG(2)<<"offset = "<<offset<<std::endl;
                int cols_in_a_tile = 0;
                for(int tk_j=j; tk_j<width; ++tk_j){
                    rows_in_a_tile += 1;
                    temp->push_back(dat->csr_diag.at(1).at(offset+tk_j));
                    temp_c->push_back(tm_i);
                    VLOG(2)<<"Push r: "<<tm_i<<std::endl;
                    VLOG(2)<<"Push c: "<<dat->csr_diag.at(1).at(offset+tk_j)<<std::endl;
                }
                if(rows_in_a_tile > max_row_in_a_tile)
                    max_row_in_a_tile = rows_in_a_tile;
            }
            
            int up_1 = 0;
            int up_3 = 0;
            //sum reduction timing
            if(col_in_a_tile != 0){
                float real_1 = log2((float)col_in_a_tile);
                up_1 = ceil(real_1)*max_row_in_a_tile;
                //accumulate partial sums (dm/tm * dc/tc tiles)
                //int reduction_rounds = ceil(ceil((height+1)/tm)*ceil(dat->hiden/tc)/hw->pec_total);
                up_3 = ceil(hw->pec_total*log2(tk*tc))/ceil((height+1)/tm); // one A tile brings about #pec sum reductions 
                hw->pec_arr.at(pec_id).set_com_delay(up_1+up_2+up_3);
            }
            //pec #
            float l1_write_temp = 0;
            if(max_row_in_a_tile != 0){
                //each tile (tm*tc) write round_trips_2 counts, another one for sum reduction
                l1_write_temp = max_row_in_a_tile*tc; // partial sums
                //l1_write_temp = ceil((height+1)/tk)*ceil(dat->hiden/tc)/hw->pec_total*tk*tc; //  sum reduction
                hw->pec_arr.at(pec_id).set_l1_write(l1_write_temp*hw->pec_total);
            }
            // each tk*tn is reused ceil() times
            int opt_counts_temp = (*temp).size()*tc+tm*tn*tc;
            hw->pec_arr.at(pec_id).set_opt_counts(opt_counts_temp*hw->pec_total); // one A tile leads to #pec ... 
            int opt_delay_temp = ceil((float)opt_counts_temp/hw->pec_arr.at(pec_id).pe_counts);
            hw->pec_arr.at(pec_id).set_opt_delay(opt_delay_temp); 
            int l1_read_temp = (*temp).size()\
                               + tm*tn*ceil((height+1)/tm)\
                               + tn*tc\
                               + tk*tc; //sum reduction
            hw->pec_arr.at(pec_id).set_l1_read(l1_read_temp*hw->pec_total); 
            double delay = opt_delay_temp*hw->pro_latency\
                        + (up_1 + up_1 + up_3)*hw->in_pec_com_latency\
                        + l1_read_temp*hw->l1_read_latency\
                        + l1_write_temp*hw->l1_write_latency;
            //VLOG(0)<<"pec_id = "<<pec_id<<" total_delay.size() = "<<hw->pec_arr.at(pec_id).total_delay.size();
            hw->pec_arr.at(pec_id).set_total_delay(delay); 
            //VLOG(0)<<"delay = "<<delay;
            
            VLOG(2)<<"pec_id = "<<pec_id<<" total_delay.size() = "<<hw->pec_arr.at(pec_id).total_delay.size();
            
            
            tiles->push_back(temp);
            tiles_c->push_back(temp_c);
            pec_id = (++pec_id)%(hw->pec_total); 
            VLOG(2)<<"Complete one tile! "<<std::endl;
        }
    }
    VLOG(1)<<"All tiles have been feched!"<<std::endl; 
    
    //compute # for acc
    std::set<int> wo_repeat_big_tile;
    std::set<int> wo_repeat_big_tile_c;
    //std::pair<std::set<int>::iterator,std::bool> flag;
    int pec_batch = 0;
    int total_in_big_tile = 0;
    int tile_counts = 0;
    // No right matrix reuse leads to redundant load
    for(auto tile_c:(*tiles_c)){
       tile_counts += 1;
       if(pec_batch<hw->pec_total && tile_counts<(*tiles_c).size()){
           for(auto ele:(*tile_c)){
                wo_repeat_big_tile_c.insert(ele);
                //if(!flag.second){LOG(FATAL)<<"insert failed! "<<std::endl;}
           }
           pec_batch++;
           total_in_big_tile += (*tile_c).size();  //total columns
       }else{
           //tile + X + W
                                  + wo_repeat_big_tile_c.size()*tn\
                                  + tn*tc*hw->pec_total\
                                  + tk*tc*hw->pec_total); // sum reduction
           VLOG(2)<<"read_l2_write_l1 = "<<total_in_big_tile\
                                  + wo_repeat_big_tile_c.size()*tn\
                                  + tn*tc*hw->pec_total\
                                  + tk*tc*hw->pec_total; // sum reduction
               
           VLOG(2)<<"hw->l2_read_2pe(end) = "<<*(hw->l2_read_2pe.end()-1);
           pec_batch = 0;
           total_in_big_tile = 0;
           wo_repeat_big_tile_c.clear();
       }
    }
    VLOG(2)<<"hw->l2_read_2pe.size() = "<<hw->l2_read_2pe.size();
    VLOG(2)<<"hw->l2_read_2pe.at(0) = "<<hw->l2_read_2pe.at(0);
    VLOG(2)<<"hw->l2_read_2pe.at(1) = "<<hw->l2_read_2pe.at(1);
    VLOG(2)<<"hw->l2_read_2pe.at(2) = "<<hw->l2_read_2pe.at(2);
    VLOG(2)<<"hw->l2_read_2pe.at(3) = "<<hw->l2_read_2pe.at(3);
    VLOG(1)<<"The first part of L2 I/O has completed!";
    pec_batch = 0;
    tile_counts = 0;
*/
    long count = 0;
    for(auto tile:(*tiles)){
       count += tile->size();
    }
    
    //print();
    VLOG(2)<<"Now extract the global results...";
    extract_timing();
    VLOG(2)<<"";
    VLOG(2)<<"Extraction has completed!";
    VLOG(2)<<"#tile: "<<tiles->size();
    VLOG(2)<<"#entry: "<<count;
    //print_final("baseline");   
}
void Analysis::extract_timing(){
    long long pec_ = 0;
    double temp_delay = 0;
    double  pec_delay = 0;
    double  l2_2_l1 = 0;
    double  l1_2_l2 = 0;
    long long  total_op = 0;
    long long total_l1_read = 0 ;
    long long total_l1_write = 0;
    long long total_op_delay = 0;
    long long total_in_pec_com_delay = 0;
    //VLOG(1)<<"hw->pec_total = "<<hw->pec_total;
    onchip_delay0 = 0;
    onchip_delay1 = 0;
    onchip_delay2 = 0;
    VLOG(1)<<"Timing within PECs...";
    VLOG(2)<<"(*tiles).size() = "<<(*tiles).size();
    VLOG(2)<<24911%4096<<"th  ="<<hw->pec_arr.at(24911%4096).total_delay.size();
    VLOG(2)<<24911/4096;
    double read_dram_write_l2 = 0;
    double read_l2_write_l1 = 0;
    double temp_read_l2_write_l1 = 0;
    double write_dram_read_l2 = 0;
    double write_l2_read_l1 = 0;
    double temp_write_l2_read_l1 = 0;
    
    onchip_delay = 0;
    l2_l1 = 0;
    l1_l2 = 0;
    for(size_t it=0; it<len; ++it){
        pec_delay = 0;
        l2_2_l1 = 0;
        l1_2_l2 = 0;
        for(int pec_id = 0; pec_id<hw->pec_total/row_divider; pec_id++){
            temp_delay = \
                hw->pec_arr.at(pec_id).total_delay.at(it);
            if(temp_delay>pec_delay)
                pec_delay = temp_delay;

            temp_read_l2_write_l1 = \
                                hw->pec_arr.at(pec_id).l2_read_2pe.at(it);
            if(temp_read_l2_write_l1>l2_2_l1)
                l2_2_l1 = temp_read_l2_write_l1;
            temp_write_l2_read_l1 = \
                                hw->pec_arr.at(pec_id).l2_write_from_pe.at(it);
            if(temp_write_l2_read_l1>l1_2_l2)
                l1_2_l2 = temp_write_l2_read_l1;
       


            total_op += hw->pec_arr.at(pec_id).opt_counts.at(it);
            total_l1_read += hw->pec_arr.at(pec_id).l1_read.at(it);
            total_l1_write += hw->pec_arr.at(pec_id).l1_write.at(it);
            total_op_delay += hw->pec_arr.at(pec_id).opt_delay.at(it);
            total_in_pec_com_delay += hw->pec_arr.at(pec_id).com_delay.at(it);

            read_l2_write_l1 += hw->pec_arr.at(pec_id).l2_read_2pe.at(it);
            write_l2_read_l1 += hw->pec_arr.at(pec_id).l2_write_from_pe.at(it);
        }
        onchip_delay += pec_delay;
        //use for delay
        l2_l1 += l2_2_l1;
        l1_l2 += l1_2_l2;
    
        VLOG(2)<<"temp_delay = "<<temp_delay;
        
        //use for energy and edp
        read_dram_write_l2 += hw->l2_write_from_dram.at(it)/hw->l2_load_store_width;
        write_dram_read_l2 += hw->l2_read_2dram.at(it)/hw->dram_width;
    }
    
    offchip_delay0 = read_dram_write_l2*(hw->l2_write+hw->dram_latency);
    offchip_delay1 = write_dram_read_l2*(hw->l2_read+hw->dram_latency);
    offchip_delay = read_dram_write_l2*(hw->l2_write+hw->dram_latency)\
                    + write_dram_read_l2*(hw->l2_read+hw->dram_latency);
    
    onchip_delay0 += onchip_delay;
    //onchip_delay1 += read_l2_write_l1*(hw->l2_read+hw->l1_write_latency);
    onchip_delay1 += l2_l1*(hw->l2_read+hw->l1_write_latency);
    onchip_delay2 += l1_l2*(hw->l1_read_latency+hw->l2_write);
    onchip_delay += onchip_delay+(l2_l1*(hw->l2_read+hw->l1_write_latency)\
            + l1_l2*(hw->l1_read_latency+hw->l2_write));
    
    global_delay = onchip_delay + offchip_delay;
    
    global_edp0 = read_dram_write_l2*(hw->dram_latency+hw->l2_write)*(hw->dram_energy+hw->l2_write_energy);
    global_edp1 = write_dram_read_l2*(hw->dram_latency+hw->l2_read)*(hw->dram_energy+hw->l2_read_energy);
    global_edp2 = row_divider*write_l2_read_l1*(hw->l2_write+hw->l1_read_latency)*(hw->l2_write_energy+hw->l1_read_energy);
    global_edp3 = row_divider*read_l2_write_l1*(hw->l2_read+hw->l1_write_latency)*(hw->l2_read_energy+hw->l1_write_energy);
    global_edp4 = row_divider*total_l1_read*hw->l1_read_energy;
    global_edp5 = row_divider*total_l1_write*hw->l1_write_energy;
    global_edp6 = row_divider*total_op*hw->pro_latency*hw->pro_energy;// excluded intra-pec reduce and inter pec scatter
    global_edp7 = row_divider*total_in_pec_com_delay*hw->in_pec_com_latency*hw->in_pec_com_energy;
    
    
    global_edp = read_dram_write_l2*(hw->dram_latency+hw->l2_write)*(hw->dram_energy+hw->l2_write_energy)\
                   +write_dram_read_l2*(hw->dram_latency+hw->l2_read)*(hw->dram_energy+hw->l2_read_energy)\
                   +row_divider*write_l2_read_l1*(hw->l2_write+hw->l1_read_latency)*(hw->l2_write_energy+hw->l1_read_energy)\
                   +row_divider*read_l2_write_l1*(hw->l2_read+hw->l1_write_latency)*(hw->l2_read_energy+hw->l1_write_energy)\
                   + global_edp7 + global_edp4 + global_edp5\
                   +row_divider*total_op*hw->pro_latency*hw->pro_energy;// excluded intra-pec reduce and inter pec scatter
    
    global_energy0 = read_dram_write_l2*(hw->dram_energy+hw->l2_write_energy);
    global_energy1 = write_dram_read_l2*(hw->dram_energy+hw->l2_read_energy);
    global_energy2 = row_divider*write_l2_read_l1*(hw->l2_write_energy+hw->l1_read_energy);
    global_energy3 = row_divider*read_l2_write_l1*(hw->l2_read_energy+hw->l1_write_energy);
    global_energy4 = row_divider*total_l1_read*hw->l1_read_energy;
    global_energy5 = row_divider*total_l1_write*hw->l1_write_energy;
    global_energy6 = row_divider*total_op*hw->pro_energy;
    global_energy7 = row_divider*total_in_pec_com_delay*hw->in_pec_com_energy;
    global_energy8 = global_delay*(hw->l1_leakage_power+hw->l2_leakage_power); // excluded intra-pec reduce and inter pec scatter
    
    global_energy = read_dram_write_l2*(hw->dram_energy+hw->l2_write_energy)\
                   +write_dram_read_l2*(hw->dram_energy+hw->l2_read_energy)\
                   +row_divider*write_l2_read_l1*(hw->l2_write_energy+hw->l1_read_energy)\
                   +row_divider*read_l2_write_l1*(hw->l2_read_energy+hw->l1_write_energy)\
                  +row_divider*total_op*hw->pro_energy\
                 +row_divider*total_l1_read*hw->l1_read_energy\
                +row_divider*total_l1_write*hw->l1_write_energy\
                +global_energy7\
                +global_delay*(hw->l1_leakage_power+hw->l2_leakage_power); // excluded intra-pec reduce and inter pec scatter
    VLOG(2)<<"total_op = "<<total_op<<std::endl;
}
void Analysis::print_final(std::string name_tag){
    std::cout<<name_tag<<"---"<<dat->name<<"----"<<hw->pec_size<<"----"<<hw->pec_total<<std::endl;
    std::cout<<"Delay: "<<global_delay<<std::endl;
    std::cout<<"            "<<" pec delay: "<<onchip_delay0<<std::endl;
    std::cout<<"            "<<" L1_2_L2: "<<onchip_delay2<<" L2_2_l1: "<<onchip_delay1<<std::endl;
    std::cout<<"            "<<" l2_2_dram: "<<offchip_delay1<<" dram_2_l2: "<<offchip_delay0<<std::endl;
    std::cout<<"EDP: "<<global_edp<<std::endl;
    std::cout<<"            "<<" Ops: "<<global_edp6<<std::endl;
    std::cout<<"            "<<" inter-PEC Com.: "<<global_edp7<<std::endl;
    std::cout<<"            "<<" PE_2_L1: "<<global_edp5<<" L1_2_PE: "<<global_edp4<<std::endl;
    std::cout<<"            "<<" L1_2_L2: "<<global_edp2<<" L2_2_l1: "<<global_edp3<<std::endl;
    std::cout<<"            "<<" l2_2_dram: "<<global_edp1<<" dram_2_l2: "<<global_edp0<<std::endl;
    std::cout<<"Energy: "<<global_energy<<std::endl;
    std::cout<<"            "<<" Ops: "<<global_energy6<<std::endl;
    std::cout<<"            "<<" Leakage: "<<global_energy8<<std::endl;
    std::cout<<"            "<<" inter-PEC Com.: "<<global_energy7<<std::endl;
    std::cout<<"            "<<" PE_2_L1: "<<global_energy5<<" L1_2_PE: "<<global_energy4<<std::endl;
    std::cout<<"            "<<" L1_2_L2: "<<global_energy2<<" L2_2_l1: "<<global_energy3<<std::endl;
    std::cout<<"            "<<" l2_2_dram: "<<global_energy1<<" dram_2_l2: "<<global_energy0<<std::endl;
}
