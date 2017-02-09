# updated January 25, 2017
import networkx as nx 
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import operator

### functions ###
import population_parameters as pop_func
import experiment_functions as exp_func
import intervention_settings as settings

###################################################    
def infected_contact_child(C, Infc_A, Infc_C, metro_id, d_metro_age_pop):
    '''returns infected degree based on contact probabilities with children
    '''

    C_cc = C.item((0, 0)) #((row, column))
    C_ac = C.item((1, 0)) # number of adult contacts with children
    child_pop = d_metro_age_pop[(metro_id, 'child')]
    adult_pop = d_metro_age_pop[(metro_id, 'adult')]
    
    return ((C_cc*(Infc_C/child_pop))+(C_ac*(Infc_A/adult_pop)))
        
###################################################    
def infected_contact_adult(C, Infc_A, Infc_C, metro_id, d_metro_age_pop):
    '''returns infected degree based on contact probabilities for adults
    '''

    C_ca = C.item((0, 1))
    C_aa = C.item((1, 1))
    child_pop = d_metro_age_pop[(metro_id, 'child')]
    adult_pop = d_metro_age_pop[(metro_id, 'adult')]

    return ((C_ca*(Infc_C/child_pop))+(C_aa*(Infc_A/adult_pop)))

###################################################
def lambda_child_calc(C, Infc_A, Infc_C, metro_id, d_metro_age_pop, beta):
    '''calculates force of infection (lambda) from children
    '''

    infc_cont_child = infected_contact_child(C, Infc_A, Infc_C, metro_id, d_metro_age_pop)
    
    lambda_child = ((beta) * (infc_cont_child))
    
    return lambda_child

###################################################    
def lambda_adult_calc(C, Infc_A, Infc_C, metro_id, d_metro_age_pop, beta):
    '''calculates force of infection (lambda) from adults
    '''

    infc_cont_adult = infected_contact_adult(C, Infc_A, Infc_C, metro_id, d_metro_age_pop)

    lambda_adult = ((beta) * (infc_cont_adult))
    
    return lambda_adult

###################################################
def SIR_initial_pops(metro_ids, d_metro_age_pop):
    '''set initial populations for each state
    '''
        
    d_Susc, d_Infc, d_Recv = {}, {}, {}
    for met_id in metro_ids:
        child_pop = d_metro_age_pop[(met_id, 'child')] 
        adult_pop = d_metro_age_pop[(met_id, 'adult')] 
        Susc_C = child_pop #value = pop size
        Susc_A = adult_pop
        Infc_C = 0
        Infc_A = 0 # number of infected adults
        Recv_C = 0
        Recv_A = 0 # number of recovered adults (empty for now)
        d_Susc[(met_id, 'C')] = Susc_C
        d_Susc[(met_id, 'A')] = Susc_A
        d_Infc[(met_id, 'C')] = Infc_C
        d_Infc[(met_id, 'A')] = Infc_A
        d_Recv[(met_id, 'C')] = Recv_C
        d_Recv[(met_id, 'A')] = Recv_A    
        
    return d_Susc, d_Infc, d_Recv
        
###################################################
def update_SI(metro_zero, met_id, d_Susc, d_Infc, num_new_infc_child, num_new_infc_adult, d_metro_infected_child, d_metro_infected_adult, time_step):
    '''track current age-specific susceptible & infected individuals in each metro area after adding new infections
    '''
    
    #children
    d_Susc[(met_id, 'C')] -= num_new_infc_child
    d_Infc[(met_id, 'C')] += num_new_infc_child
    num_infc_child = d_Infc[(met_id, 'C')]
    
    #adults
    d_Susc[(met_id, 'A')] -= num_new_infc_adult
    d_Infc[(met_id, 'A')] += num_new_infc_adult
    num_infc_adult = d_Infc[(met_id, 'A')]

    # record number of current infecteds (not just newly infected) in metro area at this time step 
    d_metro_infected_child[(metro_zero, met_id, time_step)] = num_infc_child
    d_metro_infected_adult[(metro_zero, met_id, time_step)] = num_infc_adult
    
###################################################
def update_IR(metro_zero, met_id, d_Infc, d_Recv, new_recov_child, new_recov_adult, d_metro_infected_child, d_metro_infected_adult, time_step):
    ''' track current age-specific infected and recovered individuals in each metro area after adding new recoveries
    '''
     
    #child
    d_Infc[(met_id, 'C')] -= new_recov_child
    d_Recv[(met_id, 'C')] += new_recov_child
    num_infc_child = d_Infc[(met_id, 'C')] # number of children infected at this time step in this metro area

    #adults
    d_Infc[(met_id, 'A')] -= new_recov_adult
    d_Recv[(met_id, 'A')] += new_recov_adult
    num_infc_adult = d_Infc[(met_id, 'A')]
    
    # record number of current infecteds (not just new infections) in metro area at this time step 
    d_metro_infected_child[(metro_zero, met_id, time_step)] = num_infc_child
    d_metro_infected_adult[(metro_zero, met_id, time_step)] = num_infc_adult
    
###################################################
def travel_btwn_metros(air_network, d_Susc, d_Infc, d_Recv, d_prob_travel_C, d_prob_travel_A, theta_susc, theta_infc, theta_recv):
    
    edges = air_network.edges() 
    for (i, j) in edges:
        ###### Travel of susceptibles ######
        
        #### children ####
        
        # travel metro i --> j
        ch_susc_i = (d_Susc[(i, 'C')]) # binomial number of trials
        ch_susc_prob_i_j = ((d_prob_travel_C[(i, j)]) * (theta_susc)) # binomial probability
        # select number of children who travel
        ch_travel_i_j = ((ch_susc_i) * (ch_susc_prob_i_j))
        # multiply ch_susc_i by prob = number of children travel from i to j
        
        # travel metro j --> i
        ch_susc_j = (d_Susc[(j, 'C')])
        ch_susc_prob_j_i = ((d_prob_travel_C[(j, i)]) * (theta_susc))
        # select number of children who travel
        ch_travel_j_i = ((ch_susc_j) * (ch_susc_prob_j_i))
        
        # update pop sizes
        net_travel = ((ch_travel_j_i) - (ch_travel_i_j)) # difference - if (+), j pop decreases, and i pop increases, vice versa for (-)
        d_Susc[(i, 'C')] = ((d_Susc[(i, 'C')]) + net_travel)
        d_Susc[(j, 'C')] = ((d_Susc[(j, 'C')]) - net_travel)
	        
        #### adults #####
        
        # travel metro i --> j
        ad_susc_i = (d_Susc[(i, 'A')])
        ad_susc_prob_i_j = ((d_prob_travel_A[(i, j)]) * (theta_susc))
        # select number of adults who travel
        ad_travel_i_j = ((ad_susc_i) * (ad_susc_prob_i_j))
        
        # travel metro j --> i
        ad_susc_j = (d_Susc[(j, 'A')])
        ad_susc_prob_j_i = ((d_prob_travel_A[(j, i)]) * (theta_susc))
        # select number of adults who travel
        ad_travel_j_i = ((ad_susc_j) * (ad_susc_prob_j_i))
        
        # update pop sizes
        net_travel = ((ad_travel_j_i) - (ad_travel_i_j)) # difference - if (+), j pop decreases, and i pop increases, vice versa for (-)	        
        d_Susc[(i, 'A')] = ((d_Susc[(i, 'A')]) + net_travel)
        d_Susc[(j, 'A')] = ((d_Susc[(j, 'A')]) - net_travel)
               
               
        ###### Travel of infecteds ######
        
        #### children ####
        
        # travel metro i --> j
        ch_infc_i = (d_Infc[(i, 'C')]) # binomial number of trials
        ch_infc_prob_i_j = ((d_prob_travel_C[(i, j)]) * (theta_infc)) # binomial probability
        # select number of children who travel
        ch_travel_i_j = ((ch_infc_i) * (ch_infc_prob_i_j))
        
        # travel metro j --> i
        ch_infc_j = (d_Infc[(j, 'C')])
        ch_infc_prob_j_i = ((d_prob_travel_C[(j, i)]) * (theta_infc))
        # select number of children who travel
        ch_travel_j_i = ((ch_infc_j) * (ch_infc_prob_j_i))
        
        # update pop sizes
        net_travel = ((ch_travel_j_i) - (ch_travel_i_j)) # difference - if (+), j pop decreases, and i pop increases, vice versa for (-)
        d_Infc[(i, 'C')] = ((d_Infc[(i, 'C')]) + net_travel)
        d_Infc[(j, 'C')] = ((d_Infc[(j, 'C')]) - net_travel)
	        
        #### adults #####
        
        # travel metro i --> j
        ad_infc_i = (d_Infc[(i, 'A')])
        ad_infc_prob_i_j = ((d_prob_travel_A[(i, j)]) * (theta_infc))
        # select number of adults who travel
        ad_travel_i_j = ((ad_infc_i) * (ad_infc_prob_i_j))
        
        # travel metro j --> i
        ad_infc_j = (d_Infc[(j, 'A')])
        ad_infc_prob_j_i = ((d_prob_travel_A[(j, i)]) * (theta_infc))
        # select number of adults who travel
        ad_travel_j_i = ((ad_infc_j) * (ad_infc_prob_j_i))
        
        # update pop sizes
        net_travel = ((ad_travel_j_i) - (ad_travel_i_j)) # difference - if (+), j pop decreases, and i pop increases, vice versa for (-)	        
        d_Infc[(i, 'A')] = ((d_Infc[(i, 'A')]) + net_travel)
        d_Infc[(j, 'A')] = ((d_Infc[(j, 'A')]) - net_travel)


        ###### Travel of recovereds ######
        
        #### children #####
        
        # travel metro i --> j
        ch_recv_i = (d_Recv[(i, 'C')]) # binomial number of trials
        ch_recv_prob_i_j = ((d_prob_travel_C[(i, j)]) * (theta_recv))
        # select number of children who travel
        ch_travel_i_j = ((ch_recv_i) * (ch_recv_prob_i_j))
        
        # travel metro j --> i
        ch_recv_j = (d_Recv[(j, 'C')])
        ch_recv_prob_j_i = ((d_prob_travel_C[(j, i)]) * (theta_recv))
        # select number of children who travel
        ch_travel_j_i = ((ch_recv_j) * (ch_recv_prob_j_i))
        
        # update pop sizes
        net_travel = ((ch_travel_j_i) - (ch_travel_i_j)) # difference - if (+), j pop decreases, and i pop increases, vice versa for (-)
        d_Recv[(i, 'C')] = ((d_Recv[(i, 'C')]) + net_travel)
        d_Recv[(j, 'C')] = ((d_Recv[(j, 'C')]) - net_travel)
	        
        #### adults #####
        
        # travel metro i --> j
        ad_recv_i = (d_Recv[(i, 'A')])
        ad_recv_prob_i_j = ((d_prob_travel_A[(i, j)]) * (theta_recv))
        # select number of adults who travel
        ad_travel_i_j = ((ad_recv_i) * (ad_recv_prob_i_j))
        
        # travel metro j --> i
        ad_recv_j = (d_Recv[(j, 'A')])
        ad_recv_prob_j_i = ((d_prob_travel_A[(j, i)]) * (theta_recv))
        # select number of adults who travel
        ad_travel_j_i = ((ad_recv_j) * (ad_recv_prob_j_i))
        
        # update pop sizes
        net_travel = ((ad_travel_j_i) - (ad_travel_i_j)) # difference - if (+), j pop decreases, and i pop increases, vice versa for (-)	        
        d_Recv[(i, 'A')] = ((d_Recv[(i, 'A')]) + net_travel)
        d_Recv[(j, 'A')] = ((d_Recv[(j, 'A')]) - net_travel)
        
        
###################################################
def run_one_simulation(d_metro_infected_child, d_metro_infected_adult, d_metro_tot_infected_child, d_metro_tot_infected_adult, metro_zero, d_metro_age_pop, d_metropop, metro_ids, gamma, alpha, theta_susc, theta_infc, theta_recv, time_end, ch_travelers_r, ad_travelers_s, num_metro_zeros, num_child_zeros, num_adult_zeros, params_to_calc_C, manip_exp_params, intervention_time_params, base_param_dict, exp_param_dict):
    '''simulate one epidemic on the metapopulation
    '''
        
    #create dicts with initial pops of S, I, and R for each metro and age
    # keys: (metro, 'age'), value: pop in #
    d_Susc, d_Infc, d_Recv = SIR_initial_pops(metro_ids, d_metro_age_pop)
    
    time_step = 0 # clock counter keeping track of current time
    d_nat_infected_child, d_nat_infected_adult = {}, {} # this dictionary will keep track of how many infected in current time step
    #record number currently infected at each metro at each time step


    d_new_cases_child, d_new_cases_adult = {}, {}
    # record new cases for each time step (agg over metro areas)
    d_tot_new_cases_child, d_tot_new_cases_adult = {}, {}
    
    # infect patient_zeros and upated Susc and Infc lists
    metro_zeros = []
    metro_zeros.append(metro_zero)
    #second patient_zero selection for indv - select fixed number of patient_zeros - #children, #adults
    for met_id in metro_zeros:
        update_SI(met_id, met_id, d_Susc, d_Infc, num_child_zeros, num_adult_zeros, d_metro_infected_child, d_metro_infected_adult, time_step)
        num_infc_child = d_Infc[(met_id, 'C')]
        num_infc_adult = d_Infc[(met_id, 'A')]
        d_metro_tot_infected_child[(met_id, met_id, time_step)] = num_infc_child
        d_metro_tot_infected_adult[(met_id, met_id, time_step)] = num_infc_adult
        d_new_cases_child[(met_id, time_step)] = num_infc_child
        d_new_cases_adult[(met_id, time_step)] = num_infc_adult
       
    metros_not_zeros = [metro for metro in metro_ids if metro not in metro_zeros] 
    for met_id in metros_not_zeros:
        d_metro_infected_child[(metro_zero, met_id, time_step)] = 0
        d_metro_infected_adult[(metro_zero, met_id, time_step)] = 0
        d_metro_tot_infected_child[(metro_zero, met_id, time_step)] = 0
        d_metro_tot_infected_adult[(metro_zero, met_id, time_step)] = 0
        d_new_cases_child[(met_id, time_step)] = 0
        d_new_cases_adult[(met_id, time_step)] = 0
    
    num_infc_child = sum([d_Infc[(met_id, 'C')] for met_id in metro_ids])
    num_infc_adult = sum([d_Infc[(met_id, 'A')] for met_id in metro_ids])
    d_nat_infected_child[(time_step)] = num_infc_child
    d_nat_infected_adult[(time_step)] = num_infc_adult
    
    # assign parameters
    experiment, disease_intervention, travel_intervention, beta, C, ch_travelers_r, ad_travelers_s, theta_susc, theta_infc, theta_recv = manip_exp_params
    # calculate intervention time_start and time_end from outside function
    model_peak, data_peak, data_holiday_start, dis_len_before, dis_len_after, trav_len_before, trav_len_after = intervention_time_params
    dis_start, dis_end, travel_start, travel_end = exp_func.set_time_start_and_end(model_peak, data_peak, data_holiday_start, dis_len_before, dis_len_after, trav_len_before, trav_len_after)
    if experiment == 'yes':
        if disease_intervention != 'none':
            dis_intervention_time = range(dis_start, dis_end + 1)
        elif disease_intervention == 'none':
            dis_intervention_time = range(0, 0)
        if travel_intervention != 'none':
            trav_intervention_time = range(travel_start, travel_end + 1)
        elif travel_intervention == 'none':
            trav_intervention_time = range(0, 0)
    elif experiment == 'no':
        dis_intervention_time = range(0, 0)
        trav_intervention_time = range(0, 0)  
    
    # while there are infected individuals
    # go to next time step
    num_infected = ((d_nat_infected_child[time_step]) + (d_nat_infected_adult[time_step])) 
    while num_infected >= 1 and time_step < time_end:
        
        print "seed %s time %s has %s child, %s adult, %s total infections, C = %s, beta = %s, frac_ch_travelers = %s, frac_ad_travelers = %s, theta_s = %s, theta_i = %s, theta_r = %s" % (metro_zero, time_step, d_nat_infected_child[time_step], d_nat_infected_adult[time_step], num_infected, C, beta, ch_travelers_r, ad_travelers_s, theta_susc, theta_infc, theta_recv)
        
        time_step += 1 #update clock
        
        if time_step in trav_intervention_time:
            # use one set of parameters when travel intervention is implemented
            param_dict = exp_param_dict
        elif time_step in dis_intervention_time:
            # use one set of parameters when contact intervention is implemented
            param_dict = disease_exp_param_dict 
        else:
            # otherwise, use baseline parameters
            param_dict = base_param_dict
        
        ###### TRAVEL STEP ######
        
        #assign child travelers and adult travelers
        ch_travelers_r = param_dict['r']
        ad_travelers_s = param_dict['s']
        theta_susc = param_dict['theta_susc']
        theta_infc = param_dict['theta_infc']
        theta_recv = param_dict['theta_recv']
        air_network = param_dict['trav_network']
        
    
        # create two dictionaries with probabilities of travel for each age group, keys being tuples of cities: (i, j) and (j, i)
        d_prob_travel_C, d_prob_travel_A = pop_func.calc_prob_travel(air_network, alpha, ch_travelers_r, ad_travelers_s, d_metropop)
        
        #update population sizes for S, I, R for each metro
        travel_btwn_metros(air_network, d_Susc, d_Infc, d_Recv, d_prob_travel_C, d_prob_travel_A, theta_susc, theta_infc, theta_recv)
        
        ###### DISEASE PROGRESSION ######
        
        #assign contact matrix and beta values
        C = param_dict['C']
        beta = param_dict['beta'] 
           
        for met_id in metro_ids:

            # Ii --> Ri
            # determine how many child / adult infected will recover in each metro area
            
            #child
            Infc_C = d_Infc[(met_id, 'C')] # number of infected children in metro area
            prob = gamma # probability of recovery = gamma
            new_recov_child = ((Infc_C) * (prob))
            
            #adult
            Infc_A = d_Infc[(met_id, 'A')]
            prob = gamma
            new_recov_adult = ((Infc_A) * (prob))
            
            # subtract from Ii, add to Ri            
            update_IR(metro_zero, met_id, d_Infc, d_Recv, new_recov_child, new_recov_adult, d_metro_infected_child, d_metro_infected_adult, time_step)
               
            # Si --> Ii
            # determine how many susceptibles get infected in each metro area
            
            # child
            Susc_C = d_Susc[(met_id, 'C')] # number of susc children in metro area
            prob = lambda_child_calc(C, Infc_A, Infc_C, met_id, d_metro_age_pop, beta)     
                                        
            # determine how many are infected (coin flip 'Susc_C' number of times, with probability 'prob' each flip will result in infected)
            new_cases_child = ((Susc_C) * (prob)) # determine how many are infected
            d_new_cases_child[(met_id, time_step)] = new_cases_child
            previous_time_step = (time_step - 1)
            previous_cases = d_metro_tot_infected_child[(metro_zero, met_id, previous_time_step)]
            d_metro_tot_infected_child[(metro_zero, met_id, time_step)] = previous_cases + new_cases_child
                         
            #adult
            Susc_A = d_Susc[(met_id, 'A')] # number of susc adults in metro area
            prob = lambda_adult_calc(C, Infc_A, Infc_C, met_id, d_metro_age_pop, beta) # calc probability of infection
            new_cases_adult = ((Susc_A) * (prob))
            d_new_cases_adult[(met_id, time_step)] = new_cases_adult
            previous_time_step = (time_step - 1)
            previous_cases = d_metro_tot_infected_adult[(metro_zero, met_id, previous_time_step)]
            d_metro_tot_infected_adult[(metro_zero, met_id, time_step)] = previous_cases + new_cases_adult
                    
            #subtract from Si, add to Ii
            update_SI(metro_zero, met_id, d_Susc, d_Infc, new_cases_child, new_cases_adult, d_metro_infected_child, d_metro_infected_adult, time_step)

        #record how many total infected across metro_ids at this time step
        num_infc_child = sum([d_Infc[(met_id, 'C')] for met_id in metro_ids])
        num_infc_adult = sum([d_Infc[(met_id, 'A')] for met_id in metro_ids])
        d_nat_infected_child[(time_step)] = num_infc_child
        d_nat_infected_adult[(time_step)] = num_infc_adult
        num_infected = ((d_nat_infected_child[time_step]) + (d_nat_infected_adult[time_step])) 
        d_tot_new_cases_adult[(time_step)] = sum([d_new_cases_adult[(met_id, time_step)] for met_id in metro_ids])
        d_tot_new_cases_child[(time_step)] = sum([d_new_cases_child[(met_id, time_step)] for met_id in metro_ids])        
        # go back thru while loop
        # next time step
        # travel again
        # S --> I --> R
        
    # Note num_newly_infected is the incidence time series      
    return d_new_cases_child, d_new_cases_adult, d_metro_infected_child, d_metro_infected_adult, d_metro_tot_infected_child, d_metro_tot_infected_adult, sum(d_tot_new_cases_child.values()), sum(d_tot_new_cases_adult.values()) # return total number infected in outbreak

###################################################
def run_multiple_simulations(beta, gamma, alpha, theta_susc, theta_infc, theta_recv, time_end, num_metro_zeros, num_child_zeros, num_adult_zeros, d_metropop, metro_ids, filename_metropop, ch_travelers_r, ad_travelers_s, params_to_calc_C, manip_exp_params, intervention_time_params, abrv_metro_ids):
    '''run multiple simulations'''   

    d_metro_infected_child, d_metro_infected_adult = {}, {}
    #record number total infected at each metro area
    d_metro_tot_infected_child, d_metro_tot_infected_adult = {}, {}
    # record previous new cases plus current new cases at each time step and each metro
    
    population_size = sum([d_metropop[x] for x in metro_ids])
    threshold = 0.10 # 10% of population size is our threshold for a large epidemic

    d_metro_age_pop = pop_func.calc_csv_metro_age_pop(filename_metropop, alpha)
    
    _, disease_intervention, travel_intervention, _, _, _, _, _, _, _ = manip_exp_params

    epidemic_sizes, adult_epidemic_sizes, child_epidemic_sizes = [], [], [] # will keep list of outbreak sizes
    for metro in abrv_metro_ids: 
        new_cases_incidence_time_series_metro_child, new_cases_incidence_time_series_metro_adult, incidence_time_series_metro_child, incidence_time_series_metro_adult, tot_incidence_time_series_child, tot_incidence_time_series_adult, outbreak_size_child, outbreak_size_adult = run_one_simulation(d_metro_infected_child, d_metro_infected_adult, d_metro_tot_infected_child, d_metro_tot_infected_adult, metro, d_metro_age_pop, d_metropop, metro_ids, gamma, alpha, theta_susc, theta_infc, theta_recv, time_end, ch_travelers_r, ad_travelers_s, num_metro_zeros, num_child_zeros, num_adult_zeros, params_to_calc_C, manip_exp_params, intervention_time_params, base_param_dict, exp_param_dict)

        outbreak_size = (outbreak_size_child + outbreak_size_adult)

        # save size of outbreaks
        epidemic_sizes.append(outbreak_size)
        adult_epidemic_sizes.append(outbreak_size_adult)
        child_epidemic_sizes.append(outbreak_size_child)
            
    # calculate average epidemic size, and how frequent they were
    if epidemic_sizes:
        average_epidemic_size = np.mean(epidemic_sizes)/float(population_size)
        epi_size_fractions = [x / float(population_size) for x in epidemic_sizes] #divide cases by pop to get percent of pop infected
        standard_deviation = np.std(epi_size_fractions)
        #probability_epidemic = len(epidemic_sizes)/float(num_sims)
    else:
        average_epidemic_size = 0
        #probability_epidemic = 0
    
    # calculate average adult epi size
    if adult_epidemic_sizes:
        avg_adult_epi_size = np.mean(adult_epidemic_sizes)/float(population_size-(population_size*alpha)) # adult pop = total pop - child pop
    else:
        avg_adult_epi_size = 0
    
    # calculate average child epi size
    if child_epidemic_sizes:
        avg_child_epi_size = np.mean(child_epidemic_sizes)/float(population_size*alpha) #child pop = alpha * total pop
    else:
        avg_child_epi_size = 0
            
    return average_epidemic_size, standard_deviation, avg_adult_epi_size, avg_child_epi_size, incidence_time_series_metro_child, incidence_time_series_metro_adult, tot_incidence_time_series_child, tot_incidence_time_series_adult
   
    
###################################################
def read_edgelist (filename):
    '''read metro area contact network from edgelist contained in file (filename)
    '''
    
    G = nx.Graph()
    file = open(filename, 'rU')
    reader = csv.reader(file)
    for row in reader:
        #data = row[0].split('\t') #for air_network
        data = row[0].split(' ') #for baseline_air_network and holiday_air_network
        G.add_edge(int(data[0]),int(data[1]), weight=float(data[2]))
        
    return G
    
####################################################
def write_csv_file (incidence_time_series_metro_child, incidence_time_series_metro_adult, tot_incidence_time_series_child, tot_incidence_time_series_adult, disease_intervention, travel_intervention, timing):
    
    #time_series = range(0, time_end)
    csvfile = csv.writer(open(pathname + '/model_outputs/output_nummetroseeds_%s_nummetrozeros_%s_numchildzeros_%s_numadultzeros_%s_disease_%s_travel_%s_timing_%s.csv' % ((len(abrv_metro_ids)), num_metro_zeros, num_child_zeros, num_adult_zeros, disease_intervention, travel_intervention, timing), 'wb'), delimiter = ',')
    csvfile.writerow(['metro_zero', 'time_step', 'metro_id', 'age', 'currently_infected', 'total_infected'])
    child_list_tuples, adult_list_tuples = [], [] # create separate lists for child and adult keys
    
    # grab tuples from child time series
    for (met_zero, met_id, time_step) in incidence_time_series_metro_child:
        child_list_tuples.append((met_zero, met_id, time_step))
    # grab tuples from adult time series
    for (met_zero, met_id, time_step) in incidence_time_series_metro_adult:
        adult_list_tuples.append((met_zero, met_id, time_step))
    # combine all unique tuples into one list
    tuples = list(set(child_list_tuples + adult_list_tuples))
    
    # loop thru unique tuples
    for (met_zero, met_id, time_step) in tuples:
        # assign keys not in child a value of 0
        if (met_zero, met_id, time_step) not in incidence_time_series_metro_child: 
            incidence_time_series_metro_child[(met_zero, met_id, time_step)] = incidence_time_series_metro_child.get((met_zero, met_id, time_step), 0)      
        # assign keys not in adult a value of 0
        if (met_zero, met_id, time_step) not in incidence_time_series_metro_adult:
            incidence_time_series_metro_adult[(met_zero, met_id, time_step)] = incidence_time_series_metro_adult.get((met_zero, met_id, time_step), 0)
    
    # sort all unique tuples for csv writing (csv order: metro_zero, time, met_id) - want metro zeros in order, and time steps in order
    sort_by_time_list_tuples = sorted(tuples, key=operator.itemgetter(2)) # sort by time
    sort_by_sim_list_tuples = sorted(sort_by_time_list_tuples, key=operator.itemgetter(0)) # now sort by metro_zero
    
    # write child rows
    for (met_zero, met_id, time_step) in sort_by_sim_list_tuples:
        csvfile.writerow([met_zero, time_step, met_id, 'C', (incidence_time_series_metro_child[(met_zero, met_id, time_step)]), (tot_incidence_time_series_child[(met_zero, met_id, time_step)])])          
    # write adult rows
    for (met_zero, met_id, time_step) in sort_by_sim_list_tuples:
        csvfile.writerow([met_zero, time_step, met_id, 'A', (incidence_time_series_metro_adult[(met_zero, met_id, time_step)]), (tot_incidence_time_series_adult[(met_zero, met_id, time_step)])])            
    
###################################################  
def def_orig_params (disease_intervention, travel_intervention, beta, C, ch_travelers_r, ad_travelers_s, theta_susc, theta_infc, theta_recv, baseline_air_network):
    '''create dictionary for baseline parameters
    '''
    #experiment = 'no'
    #intervention = 'none'
    
    base_param_dict = {'dis_int': disease_intervention, 'trav_int': travel_intervention, 'beta': beta, 'C': C, 'r': ch_travelers_r, 's': ad_travelers_s, 'theta_susc': theta_susc, 'theta_infc': theta_infc, 'theta_recv': theta_recv, 'trav_network': baseline_air_network}
    
    return base_param_dict
            
###################################################    
def def_disease_exp_params (disease_intervention, travel_intervention, beta, C, ch_travelers_r, ad_travelers_s, theta_susc, theta_infc, theta_recv, baseline_air_network):
    '''experiment parameters for school closure only
    '''
    
    if disease_intervention == 'red_C_cc':
        C_red = exp_func.reduce_C_cc(C)
        beta_exp = beta
    
    elif disease_intervention == 'red_C_aa':
        C_red = exp_func.reduce_C_aa(C)
        beta_exp = beta
    
    elif disease_intervention == 'red_C_all':
        C_red = exp_func.reduce_C_all(C)
        beta_exp = beta    
    # 8/11/16 reduce age-specific r
    elif disease_intervention == 'red_C_ageROnly':
    	C_red = exp_func.reduce_C_ageROnly(C)
    	beta_exp = beta
    # sequence to reduce age-specific r (lower)
    elif disease_intervention == 'red_C_ageROnly_less':
        C_red = exp_func.reduce_C_ageROnly_less(C)
        beta_exp = beta
    # sequence to reduce age-specific r (upper)
    elif disease_intervention == 'red_C_ageROnly_more':
        C_red = exp_func.reduce_C_ageROnly_more(C)
        beta_exp = beta    	
    elif disease_intervention == 'red_beta':
        C_red = C
        beta_exp = (beta * 0.6666667)
        
    elif disease_intervention == 'none':
        C_red = C
        beta_exp = beta
        
    ch_trav = ch_travelers_r
    ad_trav = ad_travelers_s
    tsusc = theta_susc
    tinfc = theta_infc
    trecv = theta_recv
    travel_net = baseline_air_network
    
    exp_param_dict = {'dis_int': disease_intervention, 'trav_int': travel_intervention, 'beta': beta_exp, 'C': C_red, 'r': ch_trav, 's': ad_trav, 'theta_susc': tsusc, 'theta_infc': tinfc, 'theta_recv': trecv, 'trav_network': travel_net}

    return exp_param_dict
    
###################################################
def def_exp_params (disease_intervention, travel_intervention, beta, C, ch_travelers_r, ad_travelers_s, theta_susc, theta_infc, theta_recv, baseline_air_network, holiday_air_network):
    '''experiment parameters for school closure and travel interventions
    '''

    if disease_intervention == 'red_C_cc':
        C_red = exp_func.reduce_C_cc(C)
        beta_exp = beta
        
    elif disease_intervention == 'red_C_aa':
        C_red = exp_func.reduce_C_aa(C)
        beta_exp = beta   
         
    elif disease_intervention == 'red_C_all':
        C_red = exp_func.reduce_C_all(C)
        beta_exp = beta    
    # reduce age-specific r
    elif disease_intervention == 'red_C_ageROnly':
    	C_red = exp_func.reduce_C_ageROnly(C)
    	beta_exp = beta
    # sequence to reduce age-specific r (lower)
    elif disease_intervention == 'red_C_ageROnly_less':
        C_red = exp_func.reduce_C_ageROnly_less(C)
        beta_exp = beta
    # sequence to reduce age-specific r (upper)
    elif disease_intervention == 'red_C_ageROnly_more':
        C_red = exp_func.reduce_C_ageROnly_more(C)
        beta_exp = beta               
    elif disease_intervention == 'red_beta':
        C_red = C
        beta_exp = (beta * 0.6666667)
        
    elif disease_intervention == 'none':
        C_red = C
        beta_exp = beta
        
    if travel_intervention == 'inc_child_trav':
        ch_trav = 0.15 # based on Kucharski increase for children fig. 2 (>30 mile travel)
        ad_trav = (1 - ch_trav)
        tsusc = theta_susc
        tinfc = theta_infc
        trecv = theta_recv
        travel_net = baseline_air_network
        
    elif travel_intervention == 'inc_all_trav':
        ch_trav = 0.15 
        ad_trav = (1 - ch_trav)
        tsusc = (theta_susc * 1.23)
        tinfc = (theta_infc * 1.23)
        trecv = (theta_recv * 1.23)
        travel_net = baseline_air_network
        
    elif travel_intervention == 'swap_networks':
        ch_trav = 0.15 
        ad_trav = (1 - ch_trav)
        tsusc = theta_susc
        tinfc = theta_infc
        trecv = theta_recv
        travel_net = holiday_air_network
    
    elif travel_intervention == 'none':
        ch_trav = ch_travelers_r
        ad_trav = ad_travelers_s
        tsusc = theta_susc
        tinfc = theta_infc
        trecv = theta_recv
        travel_net = baseline_air_network

    exp_param_dict = {'dis_int': disease_intervention, 'trav_int': travel_intervention, 'beta': beta_exp, 'C': C_red,'r': ch_trav,'s': ad_trav, 'theta_susc': tsusc, 'theta_infc': tinfc, 'theta_recv': trecv, 'trav_network': travel_net}
    
    return exp_param_dict
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
###################################################
if __name__ == "__main__":
    
    # SET WORKING DIRECTORY PATH
    pathname = os.getcwd()

    # IMPORT metro population data
    filename_metropop = pathname + '/model_inputs/metro_travel_data/metro_pop.csv'
    d_metropop, metro_ids = pop_func.import_csv_metropop(filename_metropop, 1, 2)
    sorted_metro_ids = sorted(metro_ids)
    abrv_metro_ids = sorted_metro_ids # number of seeds

    ## IMPORT baseline and holiday network travel data
    filename_baseline_air_network = pathname + '/model_inputs/metro_travel_data/baseline_flight_network_undirected.txt'
    filename_holiday_air_network = pathname + '/model_inputs/metro_travel_data/holiday_flight_network_undirected.txt'
    baseline_air_network = read_edgelist(filename_baseline_air_network)
    holiday_air_network = read_edgelist(filename_holiday_air_network)
    
    # IMPORT US population data
    us_popdata = csv.reader(open(pathname + '/model_inputs/contact_matrix_data/totalpop_age.csv', 'r'),delimiter = ',')
    dict_popdata, ages, years = pop_func.import_popdata(us_popdata, 0, 1, 2)
    dict_childpop, dict_adultpop = pop_func.pop_child_adult (dict_popdata, years)
    
    # IMPORT Germany age-specific contact data (Mossong, et al. 2008)
    filename_germ_within_group_contact_data = pathname + '/model_inputs/contact_matrix_data/within_group_polymod_germany_contact_matrix_Mossong_2008.csv'
    filename_germ_all_contact_data = pathname + '/model_inputs/contact_matrix_data/all_ages_polymod_germany_contact_matrix_Mossong_2008.csv'

    # IMPORT Germany population data (to normalize contact matrix)
    filename_germ_pop_data = pathname + 'model_inputs/contact_matrix_data/UNdata_Export_2008_Germany_Population.csv'
        
    # DEFINE POPULATION PARAMETERS
    year = 2010
    alpha = pop_func.calc_alpha(year, dict_childpop, dict_adultpop)
    d_metro_age_pop = pop_func.calc_csv_metro_age_pop(filename_metropop, alpha)
    ch_travelers_r = 0.0 # fraction of children who travel
    ad_travelers_s = (1 - ch_travelers_r)
    
    # CONTACT MATRIX
    q_c, q_a, p_c, p_a, _, _ = pop_func.calc_p(filename_germ_within_group_contact_data, filename_germ_pop_data, filename_germ_all_contact_data)
    params_to_calc_C = []
    params_to_calc_C.append(p_c)
    params_to_calc_C.append(p_a)
    params_to_calc_C.append(q_c)
    params_to_calc_C.append(q_a)
    params_to_calc_C.append(alpha)
    C = pop_func.calc_contact_matrix_pqa(p_c, p_a, q_c, q_a, alpha)
                          
    # DEFINE DISEASE PARAMETERS
    gamma = 0.5 # recovery rate based on (1/gamma) day infectious period
    beta = 0.03 
    
    # DEFINE TRAVEL PARAMETERS
    theta_susc = 1 # proportion of susceptibles that may travel
    theta_infc = 1 # proportion of infecteds that may travel
    theta_recv = 1 # proportion of recovereds that may travel

    # SET INITIAL SIMULATION CONDITIONS
    num_metro_zeros = 1 # number of metro areas to seed
    num_child_zeros = 1 # number of child patient zeros in metro seed
    num_adult_zeros = 0 # number of adult patient zeros in metro seed
    time_end = 500 # maximum time steps in the simulation
    
    # DEFINE INTERVENTION PARAMETERS
    # import intervention settings from intervention_settings.py
    experiment = settings.experiment
    disease_intervention = settings.disease_intervention
    travel_intervention = settings.travel_intervention
    timing = settings.timing

    intervention_time_params = []
    manip_exp_params = []

    manip_exp_params.append(experiment)
    manip_exp_params.append(disease_intervention)
    manip_exp_params.append(travel_intervention)

    
    # TRUNCATE SIMULATION START 
    model_peak = 191 # time steps
    data_peak = 140 # days - peak occurs about 20 weeks into epi in data
    data_holiday_start = 90 # days
    
    # SET INTERVENTION DURATION: SCHOOL CLOSURE & TRAVEL
    dis_len_before = 7 # days before holiday
    dis_len_after = 7 # days after holiday
    trav_len_before = 7 # days before holiday
    trav_len_after = 7 # days after holiday 
    
    if timing == 'extreme':
        dis_len_before = (dis_len_before * 4) # contact interv lasts 2 months
        dis_len_after = (dis_len_after * 4)
        trav_len_after = (trav_len_after * 4) # travel interv lasts 2 months
    elif timing == 'late_holiday_3wk':
        data_holiday_start = 111 #shifts holiday forward 3 weeks
    elif timing == 'late_holiday_6wk':
        data_holiday_start = 132 #shifts holiday forward 6 weeks
    elif timing == 'late_holiday_9wk':
        data_holiday_start = 153 #shifts holiday forward 9 weeks
        
    # CREATE LIST OF INTERVENTION TIMING PARAMETERS
    intervention_time_params.append(model_peak)
    intervention_time_params.append(data_peak)
    intervention_time_params.append(data_holiday_start)
    intervention_time_params.append(dis_len_before)
    intervention_time_params.append(dis_len_after)
    intervention_time_params.append(trav_len_before)
    intervention_time_params.append(trav_len_after)    

    # CREATE LIST OF EXPERIMENT PARAMETERS
    manip_exp_params.append(beta)  
    manip_exp_params.append(C)
    manip_exp_params.append(ch_travelers_r)
    manip_exp_params.append(ad_travelers_s)  
    manip_exp_params.append(theta_susc)
    manip_exp_params.append(theta_infc)
    manip_exp_params.append(theta_recv)
    
    # CREATE PARAMETER DICTIONARIES TO CALL AT DIFFERENT TIMESTEPS
    base_param_dict = def_orig_params(disease_intervention, travel_intervention, beta, C, ch_travelers_r, ad_travelers_s, theta_susc, theta_infc, theta_recv, baseline_air_network)
    disease_exp_param_dict = def_disease_exp_params(disease_intervention, travel_intervention, beta, C, ch_travelers_r, ad_travelers_s, theta_susc, theta_infc, theta_recv, baseline_air_network) # this dictionary is for timesteps with school closure only
    exp_param_dict = def_exp_params(disease_intervention, travel_intervention, beta, C, ch_travelers_r, ad_travelers_s, theta_susc, theta_infc, theta_recv, baseline_air_network, holiday_air_network) # this dictionary is for timesteps with the travel and closure interventions
    
    
    # RUN EPIDEMIC SIMULATIONS
    average_epidemic_size, std_dev, adult_epi_size, child_epi_size, incidence_time_series_metro_child, incidence_time_series_metro_adult, tot_incidence_time_series_child, tot_incidence_time_series_adult = run_multiple_simulations(beta, gamma, alpha, theta_susc, theta_infc, theta_recv, time_end, num_metro_zeros, num_child_zeros, num_adult_zeros, d_metropop, metro_ids, filename_metropop, ch_travelers_r, ad_travelers_s, params_to_calc_C, manip_exp_params, intervention_time_params, abrv_metro_ids)
 
    
    # PRINT ATTACK RATES
    print "\nAverage Large Epidemic Size = ", round(100*average_epidemic_size,2), '%.\n'
    print "\nAverage Adult Epidemic Size = ", round(100*adult_epi_size,2), '%.\n'
    print "\nAverage Child Epidemic Size = ", round(100*child_epi_size,2), '%.\n'
    
    # WRITE SIMULATION DATA TO FILE
    write_csv_file(incidence_time_series_metro_child, incidence_time_series_metro_adult, tot_incidence_time_series_child, tot_incidence_time_series_adult, disease_intervention, travel_intervention, timing)
