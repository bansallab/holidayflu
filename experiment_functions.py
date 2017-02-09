# January 25, 2017
import numpy as np
import sys
import operator

### functions ###
import population_parameters as pop_func

#######################################
def set_time_start_and_end (model_peak, data_peak, data_holiday_start, dis_len_before, dis_len_after, trav_len_before, trav_len_after):
    '''set intervention start and end time steps
    '''
    
    epi_start = (model_peak - data_peak)
    model_holiday = (epi_start + data_holiday_start)
    dis_start = (model_holiday - dis_len_before)
    dis_end = (model_holiday + dis_len_after)
    travel_start = (model_holiday - trav_len_before)
    travel_end = (model_holiday + trav_len_after)
    
    return dis_start, dis_end, travel_start, travel_end
      
#######################################
def reduce_C_cc (C):
    ''' reduce child to child contacts only (sensitivity analysis)
    '''
    
    C_cc = C.item((0, 0))
    C_ca = C.item((0, 1))
    C_ac = C.item((1, 0))
    C_aa = C.item((1, 1))
    red_percent = ((27.7 - 11.6) / 27.7) # 58% reduction
    #red_percent = .4
    red_C_cc = (C_cc - (C_cc * red_percent))
    C_exp = np.matrix([[red_C_cc, C_ca], [C_ac, C_aa]])
    #C.item((0,0)) = red_C_cc #reassign C_cc in contact matrix
    
    return C_exp
    
#######################################
def reduce_C_aa (C):
    ''' reduce adult to adult contacts only (sensitivity analysis)
    '''
    
    C_cc = C.item((0, 0))
    C_ca = C.item((0, 1))
    C_ac = C.item((1, 0))
    C_aa = C.item((1, 1))
    red_percent = ((27.7 - 11.6) / 27.7) # 58% reduction
    #red_percent = .4
    red_C_aa = (C_aa - (C_aa * red_percent))
    C_exp = np.matrix([[C_cc, C_ca], [C_ac, red_C_aa]])
    #C.item((0,0)) = red_C_cc #reassign C_cc in contact matrix
    
    return C_exp

#######################################
def reduce_C_all (C):
    ''' all elements in the contact matrix are updated according to holiday behavioral changes
    '''

    C_cc = C.item((0, 0))
    C_ca = C.item((0, 1))
    C_ac = C.item((1, 0))
    C_aa = C.item((1, 1))
    C_cc_red_percent = ((27.7 - 11.6) / 27.7) # 58% reduction
    C_ca_red_percent = ((3.8 - 2.3) / 3.8) # child contacts reported by adults
    C_ac_red_percent = ((11.2 - 11.7) / 11.2) # adult contacts reported by children
    C_aa_red_percent = ((14.8 - 15) / 14.8)
    red_C_cc = (C_cc - (C_cc * C_cc_red_percent))
    red_C_ca = (C_ca - (C_ca * C_ca_red_percent))
    red_C_ac = (C_ac - (C_ac * C_ac_red_percent))
    red_C_aa = (C_aa - (C_aa * C_aa_red_percent))
    C_exp = np.matrix([[red_C_cc, red_C_ca], [red_C_ac, red_C_aa]])
    
    return C_exp

#######################################
def reduce_C_ageROnly (C):
    ''' change contact matrix according to reductions for all child and adult contacts (sensitivity analysis)
    '''

    C_cc = C.item((0, 0))
    C_ca = C.item((0, 1))
    C_ac = C.item((1, 0))
    C_aa = C.item((1, 1))
    C_c_percent = ((7.78+5.83) / (C_cc+C_ac)) # 0.56 of total child contacts in orig C (scaled by C_red_all child contact colsum)
    C_a_percent = ((2.55+8.15) / (C_ca+C_aa)) # 0.87 of total adult contacts in orig C (scaled by C_red_all adult contact colsum)
    red_C_cc = (C_cc * C_c_percent)
    red_C_ca = (C_ca * C_a_percent)
    red_C_ac = (C_ac * C_c_percent)
    red_C_aa = (C_aa * C_a_percent)
    C_exp = np.matrix([[red_C_cc, red_C_ca], [red_C_ac, red_C_aa]])
    
    return C_exp

#######################################
def reduce_C_ageROnly_less (C):
    ''' 90% of reduce_C_ageROnly contact matrix (sensitivity analysis)
    '''

    C1 = reduce_C_ageROnly(C)
    C_exp = 0.9*C1 # 0.9 of partial school closure intervention
    
    return C_exp    

#######################################
def reduce_C_ageROnly_more (C):
    '''110% of reduce_C_ageROnly contact matrix (sensitivity analysis)
    '''

    C1 = reduce_C_ageROnly(C)
    C_exp = 1.1*C1 # 1.1 of partial school closure intervention
    
    return C_exp    

#######################################
